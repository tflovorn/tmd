import os
import subprocess
from tmd.queue.queuefile import get_qf_path

def enqueue(config):
    # Assumes appropriate queufile already exists.
    # (allow for queuefile to be written and examined before enqueuing)
    machine = config["machine"]

    if machine == "__local__":
        return _run_local(rundir, command, config)
    else:
        pass

def _run_local(config):
    qf_path = get_qf_path(config)

    cwd = os.getcwd()
    os.chdir(config["rundir"])
    subprocess.call([qf_path])
