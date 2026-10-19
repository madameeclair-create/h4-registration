"""平台适配层：不参与位姿选择；Linux 原生或 Windows + WSL。"""
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'indoor_data_root': 'data/processed_official',
    'kitti_data_root': 'data/kitti_processed_official',
    'binary_dir': 'build/bin', 'output_dir': 'outputs',
    'batch_size': 16, 'wsl_distribution': 'Ubuntu-24.04',
}
SETTINGS = dict(DEFAULTS)


def configure(config=None):
    """配置路径相对于仓库根目录；用户绝对路径只留在被忽略的本地配置中。"""
    SETTINGS.clear()
    SETTINGS.update(DEFAULTS)
    path = Path(config) if config else ROOT / 'config.local.json'
    if path.is_file():
        SETTINGS.update(json.loads(path.read_text(encoding='utf-8')))
    elif config:
        raise FileNotFoundError(path)
    for name in ['indoor_data_root', 'kitti_data_root', 'binary_dir', 'output_dir']:
        env = os.getenv(name.upper())
        value = Path(env if env else SETTINGS[name]).expanduser()
        SETTINGS[name] = str(value if value.is_absolute() else ROOT / value)
    if int(SETTINGS['batch_size']) < 1:
        raise ValueError('batch_size must be positive')
    return SETTINGS


def native_path(path):
    value = str(Path(path).resolve()).replace('\\', '/')
    if os.name != 'nt':
        return value
    if not re.match(r'^[A-Za-z]:/', value):
        raise ValueError('WSL execution requires a drive-letter path: ' + value)
    return '/mnt/' + value[0].lower() + value[2:]


def command_for(arguments):
    prefix = ['env', 'OMP_NUM_THREADS=1', 'OMP_THREAD_LIMIT=1']
    if os.name == 'nt':
        prefix = ['wsl.exe', '-d', SETTINGS['wsl_distribution'], '--'] + prefix
    return prefix + list(map(str, arguments))


def manifest_path(value):
    for token, key in [('${INDOOR_DATA_ROOT}', 'indoor_data_root'),
                       ('${KITTI_DATA_ROOT}', 'kitti_data_root')]:
        if value.startswith(token + '/'):
            return str(Path(SETTINGS[key]) / value[len(token) + 1:])
    raise ValueError('Unrecognized data root in public manifest: ' + value)


def binary(name):
    return Path(SETTINGS['binary_dir']) / name


@contextmanager
def cloud_input(path):
    """Open3D's Windows PCD reader cannot open some Unicode paths.

    Stage the unchanged bytes under an ASCII-only temporary name when needed;
    the point-cloud parsing and all preprocessing mathematics remain unchanged.
    """
    value=str(path)
    if os.name!='nt' or Path(path).suffix.lower()!='.pcd' or value.isascii():
        yield path
        return
    base=Path(tempfile.gettempdir())
    if not str(base).isascii():
        base=Path(os.environ.get('PUBLIC','C:/Users/Public'))
    if not str(base).isascii():
        raise ValueError('Use an ASCII-only Windows PCD data path or an ASCII TEMP directory')
    with tempfile.TemporaryDirectory(prefix='h4_cloud_',dir=base) as tmp:
        staged=Path(tmp)/'cloud.pcd'
        shutil.copyfile(path,staged)
        yield staged
