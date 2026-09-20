import hashlib,json
from pathlib import Path
import requests
root=Path(__file__).resolve().parents[1]
meta=requests.get('https://huggingface.co/api/models/routellm/mf_gpt4_augmented',timeout=30);meta.raise_for_status();rev=meta.json()['sha']
r=requests.get(f'https://huggingface.co/routellm/mf_gpt4_augmented/resolve/{rev}/model.safetensors',timeout=40);r.raise_for_status()
(root/'.cache'/'model.safetensors').write_bytes(r.content)
commit=requests.get('https://api.github.com/repos/lm-sys/RouteLLM/commits/main',timeout=30);commit.raise_for_status();sha=commit.json()['sha']
files={}
for name,path in {'mf.py':'routellm/routers/matrix_factorization/model.py','routers.py':'routellm/routers/routers.py','LICENSE.routellm':'LICENSE'}.items():
 d=requests.get(f'https://raw.githubusercontent.com/lm-sys/RouteLLM/{sha}/{path}',timeout=30);d.raise_for_status();(root/'.cache'/name).write_bytes(d.content);files[name]=hashlib.sha256(d.content).hexdigest()
(root/'.cache'/'routellm_manifest.json').write_text(json.dumps({'checkpoint':'routellm/mf_gpt4_augmented','revision':rev,'weights_sha256':hashlib.sha256(r.content).hexdigest(),'source_commit':sha,'source_hashes':files,'scoring_models':['gpt-4-1106-preview','mixtral-8x7b-instruct-v0.1'],'model_ids':[24,36]},indent=2),encoding='utf-8')
print('Pinned RouteLLM source and checkpoint',rev,'weight bytes',len(r.content))
