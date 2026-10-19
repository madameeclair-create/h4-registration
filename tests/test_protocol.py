"""Scientific and release checks, with synthetic inputs and no downloaded data."""
import csv
import importlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from h4_registration import runtime
from run import selected_cases, summarize
from experiments.analyze import choose
from argparse import Namespace


class ProtocolTests(unittest.TestCase):
    def test_manifest_protocol(self):
        runtime.configure()
        groups=selected_cases(Namespace(scope='all',per_group=0))
        self.assertEqual([len(g[3]) for g in groups],[1623,1623,1781,1781,555,555])
        for _,_,_,rows in groups:
            self.assertEqual(len({r['case_id'] for r in rows}),len(rows))
            for r in rows:
                for name in ['source','target','corr','gt']:
                    self.assertEqual(len(r[name+'_sha256']),64)
        demo=selected_cases(Namespace(scope='all',per_group=1))
        self.assertEqual(sum(len(g[3]) for g in demo),6)

    def test_action_distance_matches_direct_rms(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(67,3))+[2,-3,7]
        a=Rotation.from_rotvec([.2,-.3,.1]).as_matrix()
        b=Rotation.from_rotvec([-.1,.5,.3]).as_matrix()
        ta=np.array([.3,.2,-.7]);tb=np.array([-.2,.6,.4])
        mean=x.mean(0);cov=(x-mean).T@(x-mean)/len(x)
        eig,u=np.linalg.eigh(cov);root=u@np.diag(np.sqrt(np.maximum(eig,0)))@u.T
        embedding=np.sqrt(np.linalg.norm((a-b)@root)**2+np.linalg.norm((a-b)@mean+ta-tb)**2)
        direct=np.sqrt(np.mean(np.sum((x@a.T+ta-x@b.T-tb)**2,axis=1)))
        self.assertAlmostEqual(embedding,direct,places=12)

    def test_geometry_is_bidirectional_and_normal_sign_invariant(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(40,3))
        n=rng.normal(size=(40,3));n/=np.linalg.norm(n,axis=1)[:,None]
        r=Rotation.from_rotvec([.1,.2,-.3]).as_matrix();t=np.array([1.,2.,3.])
        pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=t
        source=(x,n,np.ones(40),np.ones(40,dtype=bool))
        target=(x@r.T+t,-n@r.T,np.ones(40),np.ones(40,dtype=bool))
        for domain in ['indoor','kitti']:
            core=importlib.import_module('h4_registration.core_'+domain)
            score=core.geometry(pose,source,target)
            np.testing.assert_allclose(score,np.ones((5,3)),atol=1e-12)
            np.testing.assert_allclose(score,core.geometry(np.linalg.inv(pose),target,source),atol=1e-12)

    def test_cache_decoders_and_pose_order(self):
        poses=np.repeat(np.eye(4)[None],2,axis=0);poses[1,0,3]=.4
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cache.bin'
            p.write_bytes(b'MACPOSE1'+struct.pack('<Q',2)+poses.astype('<f8').tobytes())
            for domain in ['indoor','kitti']:
                core=importlib.import_module('h4_registration.core_'+domain)
                np.testing.assert_array_equal(core.decode_cache(p),poses)

    def test_rank_fusion_uses_worst_then_mean_then_score_then_id(self):
        values={4:np.array([[.9],[.1],[.9],[.9],[.9]]),2:np.full((5,1),.6),7:np.full((5,1),.6)}
        # Candidate 4 has a poor worst rank; 2 and 7 tie on geometry.
        self.assertEqual(choose([4,7,2],values,{4:99,7:4,2:4},0),2)
        self.assertEqual(choose([4,7,2],values,{4:99,7:5,2:4},0),7)

    def test_failure_is_kept_in_recall_denominator(self):
        rows=[dict(dataset='3DMatch',descriptor='fpfh',success=True,RE_deg=1,TE_cm=2),
              dict(dataset='3DMatch',descriptor='fpfh',success=False,RE_deg='',TE_cm='')]
        report=summarize(rows)[0]
        self.assertEqual(report['RR_percent'],50)
        self.assertEqual(report['RE_success_deg'],1)


if __name__=='__main__':unittest.main()
