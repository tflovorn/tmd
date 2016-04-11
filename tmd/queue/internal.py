import os
import subprocess
import tmd.pwscf.config

def enqueue(config):
    write_queuefile(config)

    machine = config["machine"]

    if machine == "__local__":
        return _run_local(rundir, command, config)
    else:
        pass

def write_queuefile(config):
    machine = config["machine"]

    if machine == "__local__":
        _write_queuefile_local(config)
    else:
        pass

def get_qf_path(config):
    qf_name = "{}.run".format(config["prefix"])
    qf_path = os.path(config["rundir"], qf_name)

    return qf_path

def _write_queuefile_local(config):
    if config["command"] == "pw.x" or config["command"] == "pw2wannier90.x":
        file_prefix = tmd.pwscf.config.file_prefix(config)
        cmd = "mpiexec -np {} {} < {}.in > {}.out".format(str(config["cores"]),
                config["command"], file_prefix, file_prefix)
    else:
        #TODO
        pass

    with open(qf_path, 'w') as fp:
        fp.write("#!/bin/bash\n")
        fp.write(cmd)

def _run_local(config):
    qf_path = get_qf_path(config)

    cwd = os.getcwd()
    os.chdir(config["rundir"])
    subprocess.call([qf_path])
