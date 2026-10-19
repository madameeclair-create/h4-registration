"""Compile timing hooks without changing registration parameters."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h4_registration.runtime import configure,native_path,command_for

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args();cfg=configure(args.config);build=Path(cfg['binary_dir']).parent
    if args.jobs<1:p.error('--jobs must be positive')
    subprocess.run(command_for(['cmake','-S',native_path(ROOT/'cpp'),'-B',native_path(build),
                               '-DCMAKE_BUILD_TYPE=Release','-DH4_BUILD_TIMING=ON']),check=True)
    subprocess.run(command_for(['cmake','--build',native_path(build),'--target','mac_producer_timed',
                               'indoor_tools_timed','--parallel',str(args.jobs)]),check=True)
