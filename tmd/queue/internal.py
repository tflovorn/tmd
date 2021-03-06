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
    elif machine == "ls5":
        _enqueue_ls5(config)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def _run_local(config):
    qf_path = get_qf_path(config)
    cwd = os.getcwd()

    if config["calc"] == "wan_setup":
        qf_dir = os.path.join(config["base_path"], config["prefix"], "wannier")
        os.chdir(qf_dir)
        subprocess.call([qf_path])
        os.chdir(cwd)
    else:
        raise ValueError("unsupported config['calc'] for enqueue")

def _enqueue_ls5(config):
    qf_path = get_qf_path(config)
    cwd = os.getcwd()

    if config["calc"] == "wan_setup":
        qf_dir = os.path.join(config["base_path"], config["prefix"], "wannier")
        os.chdir(qf_dir)
        subprocess.call(["sbatch", qf_path])
        os.chdir(cwd)
    elif config["calc"] == "wan_setup_group":
        _run_sbatch_group(config, cwd, "wan_setup")
    elif config["calc"] == "pw_post_group":
        _run_sbatch_group(config, cwd, "pw_post")
    elif config["calc"] == "bands_only_group":
        _run_sbatch_group(config, cwd, "bands_only")
    else:
        raise ValueError("unsupported config['calc'] for enqueue")

def _run_sbatch_group(config, cwd, calc_name):
    os.chdir(config["base_path"])
    qf = "{}_{}_{}".format(config["global_prefix"], calc_name, config["prefix"])
    qf_path = os.path.join(config["base_path"], qf)
    subprocess.call(["sbatch", qf_path])
    os.chdir(cwd)
