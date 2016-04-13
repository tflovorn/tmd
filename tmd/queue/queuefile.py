import os
import stat
import tmd.pwscf.config

def write_queuefile(config):
    machine = config["machine"]

    if machine == "__local__":
        _write_queuefile_local(config)
    else:
        pass

def get_qf_path(config):
    qf_name = "run_{}".format(config["calc"])
    qf_path = os.path.join(config["base_path"], config["prefix"], "wannier", qf_name)

    return qf_path

def _write_queuefile_local(config):
    qf_path = get_qf_path(config)
    prefix = config["prefix"]

    qf = ["#!/bin/bash"]
    if config["calc"] == "wan_setup":
        qf.append("mpirun pw.x < {}.scf.in > scf.out".format(prefix))
        qf.append("cd ..")
        qf.append("cp -r wannier/* bands")
        qf.append("cd bands")
        qf.append("mpirun pw.x < {}.bands.in > bands.out".format(prefix))
        qf.append("mpirun bands.x < {}.bands_post.in > bands_post.out".format(prefix))
        if config["wannier"]:
            qf.append("cd ../wannier")
            qf.append("mpirun pw.x < {}.nscf.in > nscf.out".format(prefix))
            #TODO - w90 -pp and pw2wan
            pass
    else:
        # TODO - run wannier90
        pass

    with open(qf_path, 'w') as fp:
        qf_str = "\n".join(qf)
        fp.write(qf_str)

    os.chmod(qf_path, stat.S_IRWXU)
