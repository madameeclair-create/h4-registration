"""Input-size timing with the same five stage definitions as the manuscript."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT']:os.environ[key]='1'
import argparse,csv,json,re,struct,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h4_registration import runtime
from h4_registration import core_indoor as geom
from run import read_csv as read,write_csv as write,sha,save_json,execute,manifest
from h4_registration.runtime import native_path as linux
OUT=None

def native(command,log):
    execute(command,log,timeout=650)
    return log.read_text(encoding='utf-8',errors='replace')

def value(text,key):
    found=re.findall(re.escape(key)+r' ([^\r\n]+)',text)
    if len(found)!=1:raise RuntimeError('Missing timing marker: '+key)
    return found[0]

def single(c,rep,golden):
    d=OUT/'runs'/f'{c["case_id"]}_r{rep}';d.mkdir(exist_ok=False);wall=time.perf_counter();st=time.perf_counter();s=geom.preprocess(c['source_path']);t=geom.preprocess(c['target_path']);pre=time.perf_counter()-st
    text=native([linux(runtime.binary('mac_producer_timed')),'3dmatch',linux(c['source_path']),linux(c['target_path']),linux(c['corr_path']),linux(c['labels_path']),linux(OUT/'identity.txt'),linux(d/'mac'),c['descriptor'],linux(d/'provenance.bin')],d/'producer.log')
    phases=[float(x) for x in value(text,'MAC_STAGE_SECONDS').split()];assert len(phases)==4 and np.isfinite(phases).all() and min(phases)>=0
    st=time.perf_counter();poses=geom.decode_cache(d/'provenance.bin');cache=d/'poses.bin'
    with cache.open('wb') as f:f.write(b'MACPOSE1');f.write(struct.pack('<Q',len(poses)));f.write(poses.astype('<f8').tobytes())
    digest=sha(cache)
    if c['case_id'] in golden:assert golden[c['case_id']]['cache']==digest,'repeat candidate pool mismatch'
    (d/'score.tsv').write_text(f'{c["case_id"]}\t{linux(cache)}\t{linux(c["corr_path"])}\n',encoding='utf-8');bridge=time.perf_counter()-st
    sc=float(value(native([linux(runtime.binary('indoor_tools_timed')),'score',linux(d/'score.tsv'),linux(d/'scores')],d/'score.log'),'NATIVE_TOOL_SECONDS'))
    st=time.perf_counter();scores=read(d/'scores'/f'{c["case_id"]}_scores.csv');good=[int(r['candidate_id']) for r in scores if r['finite']=='1' and float(r['q2'])>0];good.sort(key=lambda i:-float(scores[i]['q2']))
    x=s[0];mean=x.mean(0);cov=(x-mean).T@(x-mean)/len(x);eig,u=np.linalg.eigh(cov);root=u@np.diag(np.sqrt(np.maximum(eig,0)))@u.T;actions=np.concatenate([(poses[:,:3,:3]@root).reshape(len(poses),9),poses[:,:3,:3]@mean+poses[:,:3,3]],axis=1);ids=[]
    for i in good:
        if not ids or np.min(np.linalg.norm(actions[ids]-actions[i],axis=1))>=.1:ids.append(i)
        if len(ids)>=16:break
    ds=time.perf_counter()-st;ref=geo=rank=0.;selected=-1;pose_digest='';status='no_hypothesis'
    if ids:
        st=time.perf_counter();inputs=[]
        for i in ids:
            p=d/f'h{i}.txt';np.savetxt(p,poses[i],fmt='%.17g');inputs.append(f'{c["case_id"]}_h{i}\t{linux(c["corr_path"])}\t{linux(p)}\n')
        (d/'refine.tsv').write_text(''.join(inputs),encoding='utf-8');bridge+=time.perf_counter()-st
        ref=float(value(native([linux(runtime.binary('indoor_tools_timed')),'refine',linux(d/'refine.tsv'),linux(d/'refined')],d/'refine.log'),'NATIVE_TOOL_SECONDS'))
        st=time.perf_counter();fitted={r['id']:r for r in read(d/'refined/tool_results.csv')};values={};q={}
        for i in ids:
            pose=np.loadtxt(d/'refined'/f'{c["case_id"]}_h{i}_final.txt');values[i]=geom.geometry(pose,s,t);q[i]=float(fitted[f'{c["case_id"]}_h{i}']['final_score'])
        geo=time.perf_counter()-st;st=time.perf_counter();matrix=np.array([values[i][:,2] for i in ids]);ranks=np.column_stack([rankdata(-matrix[:,j],method='average') for j in range(5)]);selected=ids[min(range(len(ids)),key=lambda j:(ranks[j].max(),ranks[j].mean(),-q[ids[j]],ids[j]))];rank=time.perf_counter()-st;pose_digest=sha(d/'refined'/f'{c["case_id"]}_h{selected}_final.txt');status='completed'
    if c['case_id'] in golden:assert golden[c['case_id']]['selected']==selected and golden[c['case_id']]['pose']==pose_digest,'repeat final pose mismatch'
    else:golden[c['case_id']]={'cache':digest,'selected':selected,'pose':pose_digest}
    hph=[pre,phases[0],phases[1],phases[2]+ds,phases[3]+sc+bridge+ref+geo+rank]
    r={k:c[k] for k in ['case_id','original_case_id','descriptor','scene','pair_id','n_correspondences']};r.update(repeat=rep,status=status,reason='' if ids else 'no finite positive-scoring hypothesis',candidate_count=len(poses),selected_candidate=selected,cache_sha256=digest,final_pose_sha256=pose_digest,MAC_graph_ms=1000*phases[0],MAC_search_ms=1000*phases[1],MAC_selection_ms=1000*phases[2],MAC_pose_ms=1000*phases[3],MAC_total_ms=1000*sum(phases),H4_preprocess_ms=1000*hph[0],H4_graph_ms=1000*hph[1],H4_search_ms=1000*hph[2],H4_selection_ms=1000*hph[3],H4_pose_ms=1000*hph[4],H4_total_ms=1000*sum(hph),external_wall_seconds=time.perf_counter()-wall);(d/'record.json').write_text(json.dumps(r,indent=2));return r

def prepare(args):
    global OUT
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ['inputs','runs']:(OUT/name).mkdir()
    groups=defaultdict(list)
    for c in manifest('indoor'):
        if c['dataset']=='3DMatch':groups[c['descriptor'],c['scene']].append(c)
    rows=[];audit=[];scenes=defaultdict(int)
    for (desc,scene),cases in sorted(groups.items()):
        if args.scene_limit and scenes[desc]>=args.scene_limit:continue
        scenes[desc]+=1
        for c in sorted(cases,key=lambda c:c['pair_id']):
            lines=[x for x in Path(c['corr_path']).read_text().splitlines() if x.strip()]
            # Always select by >=5000 rows even for a reduced timing smoke test.
            eligible=len(lines)>=5000
            audit.append(dict(descriptor=desc,scene=scene,pair_id=c['pair_id'],case_id=c['case_id'],rows=len(lines),selected=eligible))
            if eligible:break
        else:raise RuntimeError('No 5000-correspondence pair: '+scene)
        if sha(c['corr_path'])!=c['corr_sha256']:raise ValueError('Input hash mismatch')
        seed=24080+int(c['case_id']);order=np.random.default_rng(seed).permutation(len(lines))
        for n in args.sizes:
            cid=c['case_id']+'_'+str(n);corr=OUT/'inputs'/(cid+'.txt')
            corr.write_text('\n'.join(lines[i] for i in order[:n])+'\n',encoding='ascii')
            labels=OUT/'inputs'/(cid+'_labels.txt');labels.write_text('0\n'*n,encoding='ascii')
            rows.append(dict(c,original_case_id=c['case_id'],case_id=cid,n_correspondences=n,
                sampling_seed=seed,original_corr_sha256=c['corr_sha256'],corr_path=str(corr),corr_sha256=sha(corr),labels_path=str(labels)))
    write(OUT/'case_manifest.csv',rows);write(OUT/'selection_audit.csv',audit)
    np.savetxt(OUT/'identity.txt',np.eye(4),fmt='%.17g')
    save_json(OUT/'protocol.json',{'sizes':args.sizes,'repetitions':args.repeats,'scene_limit':args.scene_limit,
        'full_manuscript_protocol':args.sizes==[250,500,1000,2500,5000] and args.repeats==3 and args.scene_limit==0,
        'conditions':len(rows),'expected_timed_runs':len(rows)*args.repeats,'seed':'24080 + original case_id',
        'manifest_sha256':sha(OUT/'case_manifest.csv'),
        'tool_hashes':{name:sha(runtime.binary(name)) for name in ['mac_producer_timed','indoor_tools_timed']},
        'timing_definition':'See docs/REPRODUCIBILITY.md; Total is the sum of five unrounded stage durations, not end-to-end wall time.'})
    return rows

def main():
    parser=argparse.ArgumentParser(description='Serial five-stage timing benchmark')
    parser.add_argument('--config')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[250,500,1000,2500,5000])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--scene-limit',type=int,default=0,help='0 = all eight scenes; positive values are smoke tests only')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();runtime.configure(args.config)
    if args.repeats<1 or args.scene_limit<0 or not args.sizes or any(n<3 or n>5000 for n in args.sizes):parser.error('Invalid timing arguments')
    args.sizes=sorted(set(args.sizes))
    for name in ['mac_producer_timed','indoor_tools_timed']:
        if not runtime.binary(name).is_file():parser.error('Build timed tools with python scripts/build_timing.py')
    cases=prepare(args)
    if args.prepare_only:return
    golden={};warmed=set();rows=[]
    for c in cases:
        if c['descriptor'] not in warmed:
            single(c,0,golden);warmed.add(c['descriptor'])
        for rep in range(1,args.repeats+1):
            rows.append(single(c,rep,golden));write(OUT/'results_raw.csv',rows)
            print('Timed',len(rows),'/',len(cases)*args.repeats,flush=True)
    # Average repetitions within a pair, then average pair means across scenes.
    by_pair=defaultdict(list)
    for row in rows:by_pair[row['descriptor'],row['n_correspondences'],row['original_case_id']].append(row)
    means=defaultdict(list)
    for (desc,n,cid),rs in by_pair.items():
        means[desc,n].append({key:float(np.mean([r[key] for r in rs])) for key in rs[0] if key.endswith('_ms')})
    summary=[]
    for (desc,n),rs in sorted(means.items()):
        row=dict(descriptor=desc,n_correspondences=n,pairs=len(rs),repeats=args.repeats)
        row.update({key:float(np.mean([r[key] for r in rs])) for key in rs[0]})
        for key in ['preprocess','graph','search','selection','pose']:
            row['H4_'+key+'_percent']=100*row['H4_'+key+'_ms']/row['H4_total_ms']
        summary.append(row)
    write(OUT/'summary.csv',summary)
    save_json(OUT/'completion.json',{'complete':True,'timed_runs':len(rows),'repeat_equivalence_passed':True})
    print('Timing complete:',OUT)

if __name__=='__main__':main()
