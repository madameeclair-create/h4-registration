# 发布版：保留终稿算法，移除历史开发入口，适配 Linux / Windows+WSL。
# 由 run.py 调用；本文算法创新起点见下方醒目标记。
"""H4：候选多样性、双向整云几何验证、法向密度平衡；无训练。
使用公开输入清单，所有变体复用同一候选池；真值仅在选择完成后读取。
"""
import os
os.environ['OMP_NUM_THREADS']='1'
import csv,hashlib,json,shutil,struct,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
OUT=None
PRE=None
VARIANTS=['top1_fixed','dense_corr','diverse_corr','dense_geometry','diverse_geometry','diverse_normal','diverse_normal_balanced','diverse_scale_rank']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rs):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
from h4_registration.runtime import native_path as linux, command_for

def decode_cache(path):
    data=Path(path).read_bytes();n=struct.unpack_from('<Q',data,8)[0]
    if data[:8]==b'MACPOSE1':return np.frombuffer(data,dtype='<f8',offset=16).reshape(n,4,4).copy()
    if data[:8]!=b'MACPRV01':raise ValueError('cache format')
    output=np.empty((n,4,4));offset=16
    for k in range(n):
        output[k]=np.frombuffer(data,dtype='<f8',count=16,offset=offset).reshape(4,4);offset+=128
        m=struct.unpack_from('<Q',data,offset)[0];offset+=8+24*m
    if offset!=len(data):raise ValueError('bad provenance size')
    return output
# 【新增模块 DB｜论文3.5.1】预处理坐标、估计法向，并计算无向法向密度权重。
# 这里只读取点云，不读取真值；室内体素0.05m，KITTI体素0.30m。
def preprocess(path):
    cloud=o3d.io.read_point_cloud(str(path))
    points=np.asarray(cloud.points)
    if len(points)<3 or not np.isfinite(points).all():raise ValueError('invalid geometry input')
    cloud=cloud.voxel_down_sample(.05)
    # 使用20近邻估计局部表面方向；后续用绝对内积，法向正负翻转不影响评价。
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    points=np.asarray(cloud.points);normals=np.asarray(cloud.normals)
    # 固定坐标字典序后等间隔抽样，避免PLY行顺序影响验证采样。
    order=np.lexsort((points[:,2],points[:,1],points[:,0]))
    if len(order)>4096:order=order[np.linspace(0,len(order)-1,4096,dtype=int)]
    points=points[order].copy();normals=normals[order].copy()
    # 非有限或接近零的法向标记为无效，不能伪装为可靠的方向约束。
    norm=np.linalg.norm(normals,axis=1);valid=np.isfinite(normals).all(1)&(norm>1e-12)
    normals[valid]/=norm[valid,None];normals[~valid]=0
    # 【DB】法向邻域越密，权重越低：1/sqrt(邻域数量)，随后归一化为平均1。
    weights=np.zeros(len(points))
    if valid.any():
        # 同时建立n与-n的邻域，实现无向法向空间密度计数，邻域角度为20度。
        unit=normals[valid];tree=cKDTree(np.vstack([unit,-unit]));radius=np.sqrt(2*(1-np.cos(np.deg2rad(20))))
        count=tree.query_ball_point(unit,radius,return_length=True)
        weights[valid]=1/np.sqrt(count);weights/=weights.mean()
    else:weights[:]=1 # 仅普通几何可用；法向验证仍由valid=False标记。
    return points,normals,weights,valid
# 【BG】调和平均限制只有单方向几何支持很高的候选，不显式预测重叠掩膜。
def harmonic(a,b):return 2*a*b/(a+b) if a+b>0 else 0.
# 【新增模块 BG + NC + DB｜论文3.5】双向近邻、法向一致性与密度平衡评分。
# 输出5个距离尺度×3种评分；完整方法使用第三列，即双向法向密度平衡分数。
def geometry(p,s,t):
    x,nx,wx,vx=s;y,ny,wy,vy=t
    # 前向：源点经过T变换后，查询目标点云最近邻；d为距离，j为目标索引。
    d,j=cKDTree(y).query(x@p[:3,:3].T+p[:3,3],workers=1)
    # 反向：目标点经过T逆变换后查询源点云；两边分别平均，避免单侧支持主导。
    e,i=cKDTree(x).query((y-p[:3,3])@p[:3,:3],workers=1)
    # 【NC】旋转源法向；法向夹角不超过30度且两侧法向有效时才计入几何支持。
    nx_rot=nx@p[:3,:3].T
    ok=(np.abs(np.sum(nx_rot*ny[j],1))>=np.cos(np.deg2rad(30)))&vx&vy[j]
    backward=(np.abs(np.sum((ny@p[:3,:3])*nx[i],1))>=np.cos(np.deg2rad(30)))&vy&vx[i]
    # 对每个尺度tau计算截断线性距离响应，超过tau的点贡献0，仍保留在均值分母。
    result=[]
    for tau in [.05,.075,.1,.15,.2]:
        a=np.maximum(0,1-d/tau);b=np.maximum(0,1-e/tau)
        result.append([harmonic(a.mean(),b.mean()),harmonic((a*ok).mean(),(b*backward).mean()),
                       harmonic(np.mean(a*ok*wx),np.mean(b*backward*wy))])
    return np.asarray(result)
def run():
    if (OUT/'completion.json').exists():raise ValueError('refuse overwrite')
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if sha(OUT/'protocol.json')!=(OUT/'protocol.sha256').read_text().strip():raise ValueError('protocol changed')
    for name,digest in protocol['runtime_hashes'].items():
        if sha(OUT/'runtime'/name)!=digest:raise ValueError('runtime changed')
    cases=rows(OUT/'case_manifest.csv');started=time.perf_counter()
    (OUT/'score_inputs.tsv').write_text(''.join(f"{c['case_id']}\t{linux(c['cache_path'])}\t{linux(c['corr_path'])}\n" for c in cases),encoding='utf-8')
    command=command_for([linux(OUT/'runtime/tools')])
    with (OUT/'score_export.log').open('w',encoding='utf-8') as log:subprocess.run(command+['score',linux(OUT/'score_inputs.tsv'),linux(OUT/'scores')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    # ======================================================================
    # ★ 本文创新代码起点：已有MAC候选及对应点评分 → DS候选多样性筛选 ★
    # 对应论文3.3；此前MAC二阶图、极大团、节点引导和候选SVD均为继承部分。
    # ======================================================================
    shortlist={};tasks=[]
    for c in cases:
        poses=decode_cache(c['cache_path']);scores=rows(OUT/'scores'/(c['case_id']+'_scores.csv'))
        good=[int(r['candidate_id']) for r in scores if r['finite']=='1' and float(r['q2'])>0]
        # 按MAC的对应点MAE型支持分数降序排列；相同分数保留原始候选顺序。
        good.sort(key=lambda i:-float(scores[i]['q2']))
        # 常规Top16为对照；diverse为本文DS保留集合，最多16个，允许少于16个。
        dense=good[:16];diverse=[]
        # 用源点云均值和协方差表示变换作用距离，统一衡量旋转与平移产生的实际位移。
        cloud=np.load(c['source_npz']);x=cloud['points'];mean=x.mean(0);cov=(x-mean).T@(x-mean)/len(x)
        # 半正定协方差开平方；小于0的浮点舍入特征值截断为0，避免无效平方根。
        eig,u=np.linalg.eigh(cov);root=u@np.diag(np.sqrt(np.maximum(eig,0)))@u.T
        # 12维作用表示：[R*sqrt(Sigma)的9个元素, R*mu+t的3个元素]。
        # 两个表示的欧氏距离等于源采样点上变换位移差的RMS，单位为米。
        actions=np.concatenate([(poses[:,:3,:3]@root).reshape(len(poses),9),poses[:,:3,:3]@mean+poses[:,:3,3]],axis=1)
        # 【DS】从高分到低分贪心遍历，只有与所有已保留候选足够不同才加入。
        for i in good:
            if not diverse or np.min(np.linalg.norm(actions[diverse]-actions[i],axis=1))>=.1:diverse.append(i)
            if len(diverse)>=16:break
        shortlist[c['case_id']]={'dense':dense,'diverse':diverse}
        for i in sorted(set(dense+diverse)):
            path=OUT/'initial'/f"{c['case_id']}_h{i}.txt";np.savetxt(path,poses[i],fmt='%.17g')
            tasks.append(f"{c['case_id']}_h{i}\t{linux(c['corr_path'])}\t{linux(path)}\n")
    (OUT/'shortlists.json').write_text(json.dumps(shortlist,indent=2),encoding='utf-8')
    # 【精化｜论文3.4】调用C++工具，在固定对应关系上做加权SVD。
    # 只接受提高对应点评分的更新；此求解思想已有先例，不将SVD本身称为创新。
    (OUT/'refine_inputs.tsv').write_text(''.join(tasks),encoding='utf-8')
    with (OUT/'refinement.log').open('w',encoding='utf-8') as log:subprocess.run(command+['refine',linux(OUT/'refine_inputs.tsv'),linux(OUT/'refined')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    fitted={r['id']:r for r in rows(OUT/'refined/tool_results.csv')};result=[];candidate_rows=[];equivalence=[]
    for c in cases:
        groups=shortlist[c['case_id']]
        cloud_s=np.load(c['source_npz']);cloud_t=np.load(c['target_npz'])
        s=tuple(cloud_s[k] for k in ['points','normals','weights','normal_valid']);t=tuple(cloud_t[k] for k in ['points','normals','weights','normal_valid'])
        values={};poses={};q={}
        for i in sorted(set(groups['dense']+groups['diverse'])):
            ident=f"{c['case_id']}_h{i}";pose=np.loadtxt(OUT/'refined'/(ident+'_final.txt'));poses[i]=pose;q[i]=float(fitted[ident]['final_score'])
            # 【整云评价】使用坐标、法向、密度权重计算五个尺度的几何支持。
            values[i]=geometry(pose,s,t)
            candidate_rows.append({'case_id':c['case_id'],'candidate_id':i,'correspondence_score':q[i],'geometry_score':values[i][2,0],'normal_score':values[i][2,1],'normal_balanced_score':values[i][2,2]})
        selected={}
        if groups['dense']:
            selected['top1_fixed']=groups['dense'][0]
            for name,which,metric in [('dense_corr','dense',-1),('diverse_corr','diverse',-1),('dense_geometry','dense',0),('diverse_geometry','diverse',0),('diverse_normal','diverse',1),('diverse_normal_balanced','diverse',2)]:
                selected[name]=max(groups[which],key=lambda i:q[i] if metric<0 else values[i][2,metric])
            # 【新增模块 MS｜论文3.6】完整方法的最终选择：跨尺度名次融合。
            # 同分使用平均名次；首先最小化最差名次，再最小化平均名次，
            # 仍并列时优先精化后的对应点评分，最后按原始候选编号确定。
            ids=groups['diverse'];matrix=np.asarray([values[i][:,2] for i in ids])
            from scipy.stats import rankdata
            ranks=np.column_stack([rankdata(-matrix[:,j],method='average') for j in range(5)])
            selected['diverse_scale_rank']=ids[min(range(len(ids)),key=lambda j:(ranks[j].max(),ranks[j].mean(),-q[ids[j]],ids[j]))]
        # 真值只在上述所有方法完成选择后读取。
        # ★ 创新算法结束：在此之后才读取真值，仅用于计算RE/TE/RR。
        # 切勿把gt传入前面的筛选、精化或几何评分。
        gt=np.loadtxt(c['gt_path'])
        for variant in VARIANTS:
            row={'case_id':c['case_id'],'descriptor':c['descriptor'],'dataset':c['dataset'],'scene':c['scene'],'pair_id':c['pair_id'],'variant':variant,
                'status':'no_hypothesis','reason':'no positive-scoring finite candidate','success':False,'RE_deg':None,'TE_cm':None,'candidate_id':None,'dense_count':len(groups['dense']),'diverse_count':len(groups['diverse'])}
            if variant in selected:
                i=selected[variant];p=poses[i];re=float(np.degrees(np.arccos(np.clip((np.trace(p[:3,:3].T@gt[:3,:3])-1)/2,-1,1))))
                te=float(np.linalg.norm(p[:3,3]-gt[:3,3])*100);row.update(status='completed',reason='',success=bool(re<=15 and te<=30),RE_deg=re,TE_cm=te,candidate_id=i)
                directory=OUT/'poses'/variant;directory.mkdir(parents=True,exist_ok=True);np.savetxt(directory/(c['case_id']+'.txt'),p,fmt='%.17g')
                if variant=='top1_fixed':
                    old=np.loadtxt(PRE/'engine/fixed_control'/(c['case_id']+'_final.txt'));equivalence.append(float(np.max(np.abs(p-old))))
            result.append(row)
        write(OUT/'results_raw.csv',result);write(OUT/'candidate_scores.csv',candidate_rows)
        print('H4',c['case_id'],c['descriptor'],'diverse',len(groups['diverse']),flush=True)
    # 结果统计与配对分析不属于算法本体；失败案例必须保留在RR分母中。
    summary=[];paired=[]
    for descriptor in ['fpfh','fcgf']:
        for dataset in ['3DMatch','3DLoMatch']:
            for variant in VARIANTS:
                g=[r for r in result if r['descriptor']==descriptor and r['dataset']==dataset and r['variant']==variant];wins=[r for r in g if r['success']]
                if not g:continue # 批次可能只含一个描述子/数据集，不能除以空组数量。
                summary.append({'descriptor':descriptor,'dataset':dataset,'variant':variant,'cases':len(g),'successes':len(wins),'RR_percent':100*len(wins)/len(g),
                   'mean_RE_success_deg':float(np.mean([r['RE_deg'] for r in wins])) if wins else None,'mean_TE_success_cm':float(np.mean([r['TE_cm'] for r in wins])) if wins else None})
                ref='diverse_geometry' if variant in ['diverse_normal','diverse_normal_balanced','diverse_scale_rank'] else 'top1_fixed'
                base={r['case_id']:r for r in result if r['descriptor']==descriptor and r['dataset']==dataset and r['variant']==ref}
                paired.append({'descriptor':descriptor,'dataset':dataset,'variant':variant,'reference':ref,'recovered':sum(r['success'] and not base[r['case_id']]['success'] for r in g),'lost':sum(not r['success'] and base[r['case_id']]['success'] for r in g)})
    write(OUT/'summary.csv',summary);write(OUT/'paired_changes.csv',paired)
    checks={'complete':len(equivalence)==len(cases) and max(equivalence,default=0)==0,'exploratory_only':True,'expected_records':len(cases)*len(VARIANTS),'actual_records':len(result),
      'top1_equivalence_max_error':max(equivalence,default=None),'runtime_seconds':time.perf_counter()-started,'full_run':False}
    (OUT/'completion.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(checks)
    if not checks['complete']:raise RuntimeError('baseline equivalence failed; retained results')
