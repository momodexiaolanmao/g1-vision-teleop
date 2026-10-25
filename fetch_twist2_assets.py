"""Fetch pinned MIT TWIST2 assets; validate every Git blob before using it."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).absolute().parent
COMMIT='d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a'
REPO='https://github.com/amazon-far/TWIST2'

def git_hash(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def main():
    tree_path=ROOT/'artifacts/twist2_tree.json'
    if tree_path.exists():
        tree=json.loads(tree_path.read_text(encoding='utf-8'))
    else:
        request=urllib.request.Request(
            f'https://api.github.com/repos/amazon-far/TWIST2/git/trees/{COMMIT}?recursive=1',
            headers={'User-Agent':'Local-G1-MuJoCo-Validation'})
        with urllib.request.urlopen(request,timeout=45) as response:
            tree=json.load(response)
        if tree.get('sha')!=COMMIT or tree.get('truncated'):
            raise ValueError('Expected a complete TWIST2 tree at the pinned commit')
        tree_path.parent.mkdir(parents=True,exist_ok=True)
        tree_path.write_text(json.dumps(tree,indent=2),encoding='utf-8')
    entries={p['path']:p for p in tree['tree'] if p['type']=='blob'}
    records=[]
    def fetch(source,destination):
        item=entries[source]
        target=ROOT/destination
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and git_hash(target.read_bytes())==item['sha']:
            data=target.read_bytes()
        else:
            req=urllib.request.Request(item['url'],headers={'User-Agent':'Local-G1-MuJoCo-Validation'})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req,timeout=45) as response:
                        value=json.load(response)
                    if value.get('encoding')!='base64':
                        raise RuntimeError('Expected GitHub base64 blob')
                    data=base64.b64decode(value['content'])
                    assert len(data)==item['size'] and git_hash(data)==item['sha'],source
                    target.write_bytes(data)
                    break
                except Exception:
                    if attempt==3:
                        raise
                    time.sleep(1)
        records.append({'source':source,'file':destination,'bytes':len(data),
                        'git_blob':item['sha'],'sha256':hashlib.sha256(data).hexdigest()})
        return target
    model=fetch('assets/g1/g1_sim2sim_29dof.xml','robots/unitree_g1_twist2/g1_sim2sim_29dof.xml')
    xml=ET.parse(model)
    mesh_files=[m.attrib['file'] for m in xml.findall('.//mesh')]
    def mesh(filename):
        source='assets/g1/meshes/'+filename
        destination='robots/unitree_g1_twist2/meshes/'+filename
        target=ROOT/destination
        existing=ROOT/'robots/unitree_g1/meshes'/filename
        if existing.exists() and git_hash(existing.read_bytes())==entries[source]['sha']:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(existing.read_bytes())
        fetch(source,destination)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(mesh,mesh_files))
    assets={
        'assets/ckpts/twist2_1017_20k.onnx':'policies/twist2_tracking/policy.onnx',
        'assets/example_motions/0807_yanjie_walk_001.pkl':'fixtures/twist2_walk.pkl',
        'LICENSE':'policies/twist2_tracking/LICENSE',
        'deploy_real/server_low_level_g1_sim.py':'artifacts/twist2_upstream_controller.py',
        'deploy_real/data_utils/params.py':'artifacts/twist2_upstream_params.py',
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item:fetch(*item),assets.items()))
    source={'repository':REPO,'commit':COMMIT,'license':'MIT','assets':records,
            'policy_sha256':hashlib.sha256((ROOT/'policies/twist2_tracking/policy.onnx').read_bytes()).hexdigest()}
    for folder in ('policies/twist2_tracking','robots/unitree_g1_twist2'):
        (ROOT/folder/'SOURCE.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    (ROOT/'robots/unitree_g1_twist2/LICENSE').write_bytes((ROOT/'policies/twist2_tracking/LICENSE').read_bytes())
    print(json.dumps({'files':len(records),'mesh_files':len(mesh_files),'git_blob_validation':'passed',
                      'policy_sha256':source['policy_sha256']},indent=2))

if __name__=='__main__':
    main()
