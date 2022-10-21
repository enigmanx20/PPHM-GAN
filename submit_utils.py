import os, sys
from random import sample
import time, shutil, re

def get_next_run_id(results_dir):
    idx = []
    p = r'^[0-9]+'
    for dirname in os.listdir(results_dir):
        m = re.match(p, dirname)
        if m is not None and os.path.isdir(os.path.join(results_dir, dirname)):
            idx.append( int(m.group()) )
    if len(idx) == 0:
        run_id = 0
    else:
        run_id = max(idx)
    return run_id + 1

def creat_project_dir(results_dir, project_name):
    os.makedirs(results_dir , exist_ok=True)
    run_id = get_next_run_id(results_dir)
    project_dir = os.path.join(results_dir, '{:06}'.format(run_id) + '-' + project_name)
    os.makedirs(project_dir , exist_ok=True)
    log_dir      = os.path.join(project_dir, 'logs')
    sample_dir   = os.path.join(project_dir, 'samples')
    snapshot_dir = os.path.join(project_dir, 'snapshots')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    return project_dir, log_dir, sample_dir, snapshot_dir
               
def populate_project_dir(project_dir):
    scripts = []
    for f in os.listdir('./'):
        if os.path.isfile(f)  and  os.path.splitext(f)[-1] in ['.py', '.sh', '.npz']:
                scripts.append(f)
    for s in scripts:
        shutil.copy2(s, os.path.join(project_dir, s) )
    
    