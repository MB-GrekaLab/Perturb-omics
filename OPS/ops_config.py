"""Shared helpers for the OPS scripts: configuration, well and plate lists, the array-task number."""
import os
import sys

import yaml


def load(path=None):
    """The configuration: `path`, else $OPS_CONFIG, else config.yaml next to the scripts."""
    path = path or os.environ.get('OPS_CONFIG') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['_path'] = path
    return cfg


def require(cfg, *keys):
    """Stop with a message if any dotted key (e.g. 'sbs.n_cycles') is unset (null) in the configuration."""
    missing = []
    for key in keys:
        value = cfg
        for part in key.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None or value == '':
            missing.append(key)
    if missing:
        sys.exit(f"Please set these values in {cfg['_path']}: {', '.join(missing)}")


def fmt(cfg, template, **kwargs):
    """Fill {root}, {experiment} and any extra keyword in a path or pattern template."""
    return template.format(root=cfg['root_dir'].rstrip('/'), experiment=cfg['experiment'], **kwargs)


def read_lines(cfg, key):
    with open(os.path.join(cfg['root_dir'], cfg[key])) as f:
        return [line.strip() for line in f if line.strip()]


def plates(cfg):
    return read_lines(cfg, 'plates_file')


def wells(cfg):
    return read_lines(cfg, 'wells_file')


def task_id():
    """Array task number (1-based): $SGE_TASK_ID, else the first command-line argument."""
    task = os.environ.get('SGE_TASK_ID')
    if task and task.isdigit():
        return int(task)
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        return int(sys.argv[1])
    sys.exit('No task number: set SGE_TASK_ID or pass it as the first argument.')


def channels(cfg, mode, cycle):
    """Channels (1-based) of one cycle; channel 1 is DAPI. mode = 'SBS' or 'Pheno'."""
    n = cfg['imaging']['channels_per_cycle']
    if mode == 'Pheno':
        n = int((cfg['imaging'].get('extra_channels') or {}).get(cycle, n))
    return list(range(1, n + 1))


def plate_rows(cfg):
    """Row labels of the plate format (96: A-H, 384: A-P, 1536: A-AF), without the edge rows if configured."""
    n = {96: 8, 384: 16, 1536: 32}[int(cfg['plate_format'])]
    letters = [chr(ord('A') + i) for i in range(26)] + ['A' + chr(ord('A') + i) for i in range(6)]
    rows = letters[:n]
    return rows[1:-1] if cfg.get('exclude_edge_rows') else rows
