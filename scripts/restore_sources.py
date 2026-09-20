"""Restore exactly the pinned pilot inputs; never extract arbitrary archive paths."""
import hashlib
import json
import tarfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
EVIDENCE = ROOT / "evidence/pilot-001"


def checked(path, content, expected):
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError("Downloaded checksum mismatch: " + path.name)
    path.write_bytes(content)


def main():
    CACHE.mkdir(exist_ok=True)
    manifest = json.loads((EVIDENCE / "source_manifest.json").read_text())
    needed = {row["path"]: row for row in manifest["members"]
              if not (CACHE / (row["model"] + ".json")).exists()
              or hashlib.sha256((CACHE / (row["model"] + ".json")).read_bytes()).hexdigest() != row["sha256"]}
    if needed:
        with requests.get(manifest["url"], stream=True, timeout=90) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                for member in archive:
                    if member.name not in needed:
                        continue
                    row = needed[member.name]
                    if not member.isfile() or member.size != row["size"]:
                        raise RuntimeError("Archive member differs")
                    checked(CACHE / (row["model"] + ".json"), archive.extractfile(member).read(), row["sha256"])
                    del needed[member.name]
                    if not needed:
                        break
        if needed:
            raise RuntimeError("Pinned archive is missing required members")
    mf = json.loads((EVIDENCE / "routellm_manifest.json").read_text())
    path = CACHE / "model.safetensors"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != mf["weights_sha256"]:
        url = f"https://huggingface.co/{mf['checkpoint']}/resolve/{mf['revision']}/model.safetensors"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        checked(path, response.content, mf["weights_sha256"])
    (CACHE / "routellm_manifest.json").write_text(json.dumps(mf, indent=2))
    print("Pinned dataset pair and RouteLLM weights verified")


if __name__ == "__main__":
    main()
