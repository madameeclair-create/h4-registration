"""严格消融、候选预算和离线覆盖率；复用主实验保存的候选，不重新精化。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import importlib
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import read_csv, write_csv, sha

VARIANTS=['full_h4','without_diversity','forward_only','without_normal',
          'without_density_balance','without_multiscale','budget_top1','budget_top4','budget_top8']


def harmonic(a,b):
    return 2*a*b/(a+b) if a+b>0 else 0.


def score(p,s,t,tx,ty,scale):
    """列为完整分数、单向、去NC、去DB；与论文严格消融的数学定义一致。"""
    x,nx,wx,vx=s;y,ny,wy,vy=t
    d,j=ty.query(x@p[:3,:3].T+p[:3,3],workers=1)
    e,i=tx.query((y-p[:3,3])@p[:3,:3],workers=1)
    ok=(np.abs(np.sum((nx@p[:3,:3].T)*ny[j],1))>=np.cos(np.deg2rad(30)))&vx&vy[j]
    back=(np.abs(np.sum((ny@p[:3,:3])*nx[i],1))>=np.cos(np.deg2rad(30)))&vy&vx[i]
    result=[]
    for tau in np.array([.05,.075,.1,.15,.2])*scale:
        a=np.maximum(0,1-d/tau);b=np.maximum(0,1-e/tau)
        result.append([harmonic(np.mean(a*ok*wx),np.mean(b*back*wy)),np.mean(a*ok*wx),
                       harmonic(np.mean(a*wx),np.mean(b*wy)),harmonic(np.mean(a*ok),np.mean(b*back))])
    return np.asarray(result)


def choose(ids,values,q,metric):
    mat=np.array([values[i][:,metric] for i in ids])
    ranks=np.column_stack([rankdata(-mat[:,j],method='average') for j in range(5)])
    return ids[min(range(len(ids)),key=lambda j:(ranks[j].max(),ranks[j].mean(),-q[ids[j]],ids[j]))]


def metrics(pose,gt,dataset):
    re=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@gt[:3,:3])-1)/2,-1,1))))
    te=float(np.linalg.norm(pose[:3,3]-gt[:3,3])*100)
    passed=re<= (5 if dataset=='KITTI' else 15) and te<= (60 if dataset=='KITTI' else 30)
    return re,te,bool(passed)


def main():
    parser=argparse.ArgumentParser(description='Analyze cached benchmark outputs without new generation/refinement')
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args();folder=args.run_dir.resolve()
    if not (folder/'completion.json').is_file():
        parser.error('A completed run directory is required')
    out=folder/'analysis'
    out.mkdir(exist_ok=False)
    primary=read_csv(folder/'results_raw.csv')
    # Seed coverage rows with every evaluated case, including failed generation.
    coverage={ (('kitti' if c['dataset']=='KITTI' else 'indoor'),c['case_id']):
        {k:c[k] for k in ['case_id','dataset','descriptor','scene','pair_id']} |
        {stage:False for stage in ['original_pool','conventional_top16_before','diverse_top16_before',
                                   'conventional_top16_after','diverse_top16_after']} for c in primary }
    results=[]
    for path in sorted(folder.glob('*/*/case_manifest.csv')):
        block=path.parent;cases=read_csv(path)
        groups=json.loads((block/'shortlists.json').read_text(encoding='utf-8'))
        old={(r['case_id'],r['variant']):r for r in read_csv(block/'results_raw.csv')}
        qrows={(r['case_id'],int(r['candidate_id'])):float(r['correspondence_score'])
               for r in read_csv(block/'candidate_scores.csv')}
        for c in cases:
            domain='kitti' if c['dataset']=='KITTI' else 'indoor';scale=6 if domain=='kitti' else 1
            core=importlib.import_module('h4_registration.core_'+domain)
            cid=c['case_id'];g=groups[cid];poses={};q={};values={}
            for name in ['source','target']:
                if sha(c[name+'_npz'])!=c[name+'_npz_sha256']:
                    raise ValueError('Geometry cache changed: '+cid)
            with np.load(c['source_npz']) as f:s=tuple(f[k] for k in ['points','normals','weights','normal_valid'])
            with np.load(c['target_npz']) as f:t=tuple(f[k] for k in ['points','normals','weights','normal_valid'])
            tx,ty=cKDTree(s[0]),cKDTree(t[0])
            for i in sorted(set(g['dense']+g['diverse'])):
                poses[i]=np.loadtxt(block/'refined'/f'{cid}_h{i}_final.txt')
                q[i]=qrows[cid,i];values[i]=score(poses[i],s,t,tx,ty,scale)
            ds=g['diverse'];dense=g['dense']
            selected={}
            if dense:
                selected={'full_h4':choose(ds,values,q,0),'without_diversity':choose(dense,values,q,0),
                          'forward_only':choose(ds,values,q,1),'without_normal':choose(ds,values,q,2),
                          'without_density_balance':choose(ds,values,q,3),
                          'without_multiscale':max(ds,key=lambda i:values[i][2,0])}
                for k in [1,4,8]:selected['budget_top'+str(k)]=choose(ds[:k],values,q,0)
            # ★ 所有变体选完后才读取GT；覆盖率是独立离线诊断，不参与选择。
            if sha(c['gt_path'])!=c['gt_sha256']:raise ValueError('GT changed')
            gt=np.loadtxt(c['gt_path'])
            for variant in VARIANTS:
                row={k:c[k] for k in ['case_id','dataset','descriptor','scene','pair_id']}
                row.update(variant=variant,status='no_hypothesis',success=False,RE_deg='',TE_cm='',candidate_id='')
                if variant in selected:
                    i=selected[variant];re,te,passed=metrics(poses[i],gt,c['dataset'])
                    row.update(status='completed',success=passed,RE_deg=re,TE_cm=te,candidate_id=i)
                if variant in ['full_h4','without_multiscale']:
                    ref=old[cid,'diverse_scale_rank' if variant=='full_h4' else 'diverse_normal_balanced']
                    if str(row['candidate_id']) != ref['candidate_id']:
                        raise ValueError('Cached full/single-scale selection mismatch: '+cid)
                results.append(row)
            initial=core.decode_cache(c['cache_path'])
            cv=coverage[domain,cid]
            cv['original_pool']=any(metrics(p,gt,c['dataset'])[2] for p in initial)
            for group,label in [(dense,'conventional_top16'),(ds,'diverse_top16')]:
                cv[label+'_before']=any(metrics(initial[i],gt,c['dataset'])[2] for i in group)
                cv[label+'_after']=any(metrics(poses[i],gt,c['dataset'])[2] for i in group)
    existing={(r['dataset'],r['descriptor'],r['case_id']) for r in results}
    for c in primary:
        if (c['dataset'],c['descriptor'],c['case_id']) not in existing:
            for v in VARIANTS:
                results.append(dict(c,variant=v))
    write_csv(out/'ablation_raw.csv',results)
    write_csv(out/'coverage_raw.csv',list(coverage.values()))
    groups=defaultdict(list)
    for r in results:groups[r['dataset'],r['descriptor'],r['variant']].append(r)
    summary=[]
    for (dataset,descriptor,variant),rs in sorted(groups.items()):
        wins=[r for r in rs if str(r['success']).lower()=='true']
        summary.append(dict(dataset=dataset,descriptor=descriptor,variant=variant,cases=len(rs),successes=len(wins),
                            RR_percent=100*len(wins)/len(rs),
                            RE_success_deg=float(np.mean([float(r['RE_deg']) for r in wins])) if wins else '',
                            TE_success_cm=float(np.mean([float(r['TE_cm']) for r in wins])) if wins else ''))
    write_csv(out/'ablation_summary.csv',summary)
    cv_groups=defaultdict(list)
    for r in coverage.values():cv_groups[r['dataset'],r['descriptor']].append(r)
    cv_summary=[]
    for (dataset,descriptor),rs in sorted(cv_groups.items()):
        for stage in ['original_pool','conventional_top16_before','diverse_top16_before','conventional_top16_after','diverse_top16_after']:
            cv_summary.append(dict(dataset=dataset,descriptor=descriptor,stage=stage,cases=len(rs),
                                   coverage_percent=100*sum(r[stage] for r in rs)/len(rs)))
    write_csv(out/'coverage_summary.csv',cv_summary)
    (out/'completion.json').write_text(json.dumps({'complete':True,'cases':len(primary),'ablation_records':len(results),
        'generation_repeated':False,'refinement_repeated':False},indent=2),encoding='utf-8')
    print('Analysis complete:',out)


if __name__=='__main__':main()
