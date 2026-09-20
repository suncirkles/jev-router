import io, json, os, tarfile, time
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]
def get(url):
    r=requests.get(url, timeout=40); r.raise_for_status(); return r
for name,url in {
 'mf.py':'https://raw.githubusercontent.com/lm-sys/RouteLLM/main/routellm/routers/matrix_factorization/model.py',
 'routers.py':'https://raw.githubusercontent.com/lm-sys/RouteLLM/main/routellm/routers/routers.py',
}.items():
 r=get(url); (ROOT/'.cache'/name).write_text(r.text,encoding='utf-8'); print(name, len(r.content))
r=requests.post('https://openrouter.ai/api/alpha/decisions',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},json={'model':'~typesafe/jev-latest','state':'Rename a local variable in a Python function.','questions':{'difficulty':{'type':'choice','instructions':'How much reasoning does the task require?','criteria':{'simple':'Direct localized edit','complex':'Multi-step algorithm or architecture work'}}}},timeout=40)
print('Jev decisions status',r.status_code)
d=r.json(); (ROOT/'artifacts'/'jev-smoke.json').write_text(json.dumps(d,indent=2),encoding='utf-8'); print({k:d.get(k) for k in ['model','answers','usage','error']})
url='https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz'
with requests.get(url, stream=True,timeout=40) as rr:
 rr.raise_for_status()
 archive=tarfile.open(fileobj=rr.raw,mode='r|gz')
 start=time.monotonic()
 for i,m in enumerate(archive):
  print(m.name,m.size,flush=True)
  if i>=35 or time.monotonic()-start>35: break
