import os
import stat
import tmd.pwscf.config
from tmd.queue.queue_util import global_config

def write_queuefile(config):
    machine = config["machine"]

    gconf = global_config()

    if machine == "__local__":
        if gconf['allow_local']:
            _write_queuefile_local(config)
        else:
            raise ValueError("Local operation disallowed in global_config.yaml")
    elif machine == "ls5":
        _write_queuefile_ls5(config)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def get_qf_path(config):
    qf_name = "run_{}".format(config["calc"])
    qf_path = os.path.join(config["base_path"], config["prefix"], "wannier", qf_name)

    return qf_path

def _ls_format_duration(hours, minutes):
    hstr = str(hours)
    if minutes < 10:
        mstr = "0{}".format(str(minutes))
    else:
        mstr = str(minutes)

    return "{}:{}:00".format(hstr, mstr)

def _write_queuefile_ls5(config):
    duration = _ls_format_duration(config["hours"], config["minutes"])
    prefix = config["prefix"]

    qf = ["#!/bin/bash"]
    qf.append("#SBATCH -p {}".format(config["queue"]))
    qf.append("#SBATCH -J {}".format(prefix))
    #qf.append("#SBATCH -o {}.out".format(prefix))
    qf.append("#SBATCH -e {}.err".format(prefix))
    qf.append("#SBATCH -t {}".format(duration))
    qf.append("#SBATCH -N {}".format(str(config["nodes"])))
    qf.append("#SBATCH -n {}".format(str(config["cores"])))
    qf.append("#SBATCH -A {}".format(config["project"]))
    qf.append("")
    qf.append("export OMP_NUM_THREADS=1")

    if config["calc"] == "wan_setup":
        qf.append("ibrun tacc_affinity pw.x -input {}.scf.in > scf.out".format(prefix))
        qf.append("cd ..")
        qf.append("cp -r wannier/* bands")
        qf.append("cd bands")
        qf.append("ibrun tacc_affinity pw.x -input {}.bands.in > bands.out".format(prefix))
        qf.append("ibrun tacc_affinity bands.x -input {}.bands_post.in > bands_post.out".format(prefix))
        if config["wannier"]:
            qf.append("cd ../wannier")
            qf.append("ibrun tacc_affinity pw.x -input {}.nscf.in > nscf.out".format(prefix))
            qf.append("wannier90.x -pp {}".format(prefix))
            qf.append("ibrun tacc_affinity pw2wannier90.x -input {}.pw2wan.in > pw2wan.out".format(prefix))
    elif config["calc"] == "wan_run":
        qf.append("wannier90.x {}".format(prefix))
    else:
        raise ValueError("unrecognized config['calc'] ('wan_setup' and 'wan_run' supported)")

    qf_path = get_qf_path(config)

    with open(qf_path, 'w') as fp:
        qf_str = "\n".join(qf)
        fp.write(qf_str)

    os.chmod(qf_path, stat.S_IRWXU)

def _write_queuefile_local(config):
    qf_path = get_qf_path(config)
    prefix = config["prefix"]
    if "__local_mpi_cmd__" in config:
        mpi = "{} ".format(config["__local_mpi_cmd__"])
    else:
        mpi = ""

    qf = ["#!/bin/bash"]
    if config["calc"] == "wan_setup":
        qf.append("{}pw.x < {}.scf.in > scf.out".format(mpi, prefix))
        qf.append("cd ..")
        qf.append("cp -r wannier/* bands")
        qf.append("cd bands")
        qf.append("{}pw.x < {}.bands.in > bands.out".format(mpi, prefix))
        qf.append("{}bands.x < {}.bands_post.in > bands_post.out".format(mpi, prefix))
        if config["wannier"]:
            qf.append("cd ../wannier")
            qf.append("{}pw.x -input {}.nscf.in > nscf.out".format(mpi, prefix))
            qf.append("wannier90.x -pp {}".format(prefix))
            qf.append("{}pw2wannier90.x -input {}.pw2wan.in > pw2wan.out".format(mpi, prefix))
    elif config["calc"] == "wan_run":
        qf.append("wannier90.x {}".format(prefix))
    else:
        raise ValueError("unrecognized config['calc'] ('wan_setup' and 'wan_run' supported)")

    with open(qf_path, 'w') as fp:
        qf_str = "\n".join(qf)
        fp.write(qf_str)

    os.chmod(qf_path, stat.S_IRWXU)