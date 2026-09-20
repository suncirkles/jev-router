import hashlib,json,time,tarfile
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache'; CACHE.mkdir(exist_ok=True)
meta=requests.get('https://huggingface.co/api/datasets/NPULH/LLMRouterBench',timeout=30);meta.raise_for_status()
revision=meta.json()['sha']
(CACHE/'dataset_metadata.json').write_text(json.dumps(meta.json(),indent=2),encoding='utf-8')
url=f'https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/{revision}/bench-release.tar.gz'
start=time.monotonic(); saved=[]; names=[]; last_group=None
with requests.get(url,stream=True,timeout=60) as r:
 r.raise_for_status()
 with tarfile.open(fileobj=r.raw,mode='r|gz') as archive:
  for m in archive:
   names.append({'path':m.name,'size':m.size})
   group=m.name.split('/')[1] if '/' in m.name else m.name
   if group!=last_group:
    print('Archive group:',group,'elapsed',round(time.monotonic()-start),flush=True);last_group=group
   is_code=any(x in group.lower() for x in ['livecode','lcb'])
   candidate=next((x for x in ['gemini-2.5-flash','gpt-5','gpt-5-medium'] if '/'+x+'/' in m.name),None)
   if is_code and candidate and m.isfile() and m.name.endswith('.json'):
    if m.size>100_000_000: raise RuntimeError('Unexpected member size')
    b=archive.extractfile(m).read(); name=candidate+'.json'; (CACHE/name).write_bytes(b)
    saved.append({'model':candidate,'path':m.name,'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)})
    print('Saved',name,len(b),flush=True)
    if len(saved)>=2:break
   if time.monotonic()-start>900: raise TimeoutError('Archive scan exceeded 15 minute limit')
(CACHE/'source_manifest.json').write_text(json.dumps({'dataset':'NPULH/LLMRouterBench','revision':revision,'url':url,'members':saved,'scanned_members':names},indent=2),encoding='utf-8')
print('Finished',saved,flush=True)
