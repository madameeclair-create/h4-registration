"""从源码编译五个工具；在仓库根目录运行 python scripts/build.py。"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h4_registration.runtime import configure, command_for, native_path


def main():
    p = argparse.ArgumentParser(description='Build MAC frontend and H4 refinement tools')
    p.add_argument('--config')
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    settings = configure(args.config)
    build = Path(settings['binary_dir']).parent
    subprocess.run(command_for(['cmake', '-S', native_path(ROOT / 'cpp'),
                                '-B', native_path(build), '-DCMAKE_BUILD_TYPE=Release']), check=True)
    subprocess.run(command_for(['cmake', '--build', native_path(build), '--parallel', str(args.jobs)]), check=True)
    names = ['mac_producer', 'indoor_tools', 'kitti_tools', 'indoor_control', 'kitti_control']
    for name in names:
        if not (Path(settings['binary_dir']) / name).is_file():
            raise RuntimeError('Missing build output: ' + name)
    print('Build complete:', settings['binary_dir'])


if __name__ == '__main__':
    main()
