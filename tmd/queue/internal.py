import os
import subprocess
from tmd.queue.queuefile import get_qf_path
from tmd.queue.queue_util import global_config

def enqueue(config):
    # Assumes appropriate queufile already exists.
    # (allow for queuefile to be written and examined before enqueuing)
    machine = config["machine"]

    gconf = global_config()

    if machine == "__local__":
        if gconf['allow_local']:
            return _run_local(rundir, command, config)
        else:
            raise ValueError("Local operation disallowed in global_config.yaml")
    else:
        pass

def _run_local(config):
    qf_path = get_qf_path(config)

    cwd = os.getcwd()
    os.chdir(config["rundir"])
    subprocess.call([qf_path])
