import numpy as np
import os
import yaml
from copy import deepcopy
from tmd.bilayer.material import get_material
from tmd.bilayer.bilayer_util import _base_dir, global_config
from tmd.pwscf.build import build_qe, build_bands, build_pw2wan
from tmd.wannier.build import Winfile
from tmd.queue.queuefile import write_queuefile, write_launcherfiles, write_job_group_files
from tmd.queue.internal import enqueue

def dgrid_inputs(db_path, sym_A, sym_B=None, c_sep=None, num_d_a=None, num_d_b=None, soc=True):
    if sym_B is None:
        d_as = [0.0]
        d_bs = [0.0]
    else:
        d_as = np.linspace(0.0, 1.0, num_d_a, endpoint=False)
        d_bs = np.linspace(0.0, 1.0, num_d_b, endpoint=False)

    inputs = {}

    for d_a in d_as:
        for d_b in d_bs:
            material = get_material(db_path, sym_A, sym_B, c_sep, d_a, d_b, soc)

            inputs[(d_a, d_b)] = {"material": material}
            for calc_type in ["scf", "nscf", "bands"]:
                qe_input = build_qe(material, calc_type)
                inputs[(d_a, d_b)][calc_type] = qe_input

            inputs[(d_a, d_b)]["bands_post"] = build_bands(material)
            inputs[(d_a, d_b)]["pw2wan"] = build_pw2wan(material)
            wan_up, wan_down = Winfile(material)
            # TODO handle collinear spin-polarized case
            inputs[(d_a, d_b)]["wannier"] = wan_up

    return inputs

def write_dgrid(base_path, dgrid):
    for dk, dv in dgrid.items():
        _write_dv(base_path, dv)

def _write_dv(base_path, dv):
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    prefix = dv["material"]["prefix"]
    
    d_dir_path = os.path.join(base_path, prefix)
    if not os.path.exists(d_dir_path):
        os.mkdir(d_dir_path)

    material_path = os.path.join(d_dir_path, "material.yaml")
    material_str = yaml.dump(dv["material"])
    with open(material_path, 'w') as fp:
        fp.write(material_str)

    wannier_dir_path = os.path.join(d_dir_path, "wannier")
    bands_dir_path = os.path.join(d_dir_path, "bands")
    if not os.path.exists(wannier_dir_path):
        os.mkdir(wannier_dir_path)

    if not os.path.exists(bands_dir_path):
        os.mkdir(bands_dir_path)

    scf_path = os.path.join(wannier_dir_path, "{}.scf.in".format(prefix))
    nscf_path = os.path.join(wannier_dir_path, "{}.nscf.in".format(prefix))
    bands_path = os.path.join(bands_dir_path, "{}.bands.in".format(prefix))
    bands_post_path = os.path.join(bands_dir_path, "{}.bands_post.in".format(prefix))
    pw2wan_path = os.path.join(wannier_dir_path, "{}.pw2wan.in".format(prefix))

    with open(scf_path, 'w') as fp:
        fp.write(dv["scf"])

    with open(nscf_path, 'w') as fp:
        fp.write(dv["nscf"])

    with open(bands_path, 'w') as fp:
        fp.write(dv["bands"])

    with open(bands_post_path, 'w') as fp:
        fp.write(dv["bands_post"])

    with open(pw2wan_path, 'w') as fp:
        fp.write(dv["pw2wan"])

    wannier_path = os.path.join(wannier_dir_path, "{}.win".format(prefix))

    with open(wannier_path, 'w') as fp:
        fp.write(dv["wannier"])

def write_dgrid_queuefiles(base_path, dgrid, config):
    prefix_list = []
    for dk, dv in dgrid.items():
        _write_dv_queuefile(base_path, dv, config)
        prefix_list.append(dv["material"]["prefix"])

    prefix_groups = group_jobs(config, prefix_list)
    write_prefix_groups(base_path, config["global_prefix"], prefix_groups)

    config["base_path"] = base_path
    wan_setup_group_config = deepcopy(config)
    wan_setup_group_config["calc"] = "wan_setup"
    write_job_group_files(wan_setup_group_config, prefix_groups)

    pw_post_group_config = deepcopy(config)
    pw_post_group_config["calc"] = "pw_post"
    pw_post_group_config["nodes"] = 1
    pw_post_group_config["cores"] = config["pw_post_cores"]
    write_job_group_files(pw_post_group_config, prefix_groups)

    launcher_config = deepcopy(config)
    launcher_config["prefix_list"] = prefix_list
    launcher_config["calc"] = "wan_run"
    write_launcherfiles(launcher_config)

    return prefix_groups

def _prefix_groups_path(base_path, global_prefix):
    groups_path = os.path.join(base_path, "{}_prefix_groups.yaml".format(global_prefix))
    return groups_path

def write_prefix_groups(base_path, global_prefix, prefix_groups):
    groups_path = _prefix_groups_path(base_path, global_prefix)
    groups_str = yaml.dump(prefix_groups)
    with open(groups_path, 'w') as fp:
        fp.write(groups_str)

def get_prefix_groups(base_path, global_prefix):
    groups_path = _prefix_groups_path(base_path, global_prefix)
    with open(groups_path, 'r') as fp:
        groups_str = fp.read()
        
    prefix_groups = yaml.load(groups_str)
    return prefix_groups

def group_jobs(config, prefix_list):
    max_jobs = config["max_jobs"]

    groups = []
    for i, prefix in enumerate(prefix_list):
        group_id = i % max_jobs
        if len(groups) <= group_id:
            groups.append([])

        groups[group_id].append(prefix)

    return groups

def _write_dv_queuefile(base_path, dv, config):
    config["base_path"] = base_path

    prefix = dv["material"]["prefix"]
    config["prefix"] = prefix

    wan_setup_config = deepcopy(config)
    wan_setup_config["calc"] = "wan_setup"
    write_queuefile(wan_setup_config)

    pw_post_config = deepcopy(config)
    pw_post_config["calc"] = "pw_post"
    pw_post_config["nodes"] = 1
    pw_post_config["cores"] = config["pw_post_cores"]
    write_queuefile(pw_post_config)

    wan_run_config = deepcopy(config)
    wan_run_config["calc"] = "wan_run"
    write_queuefile(wan_run_config)

def submit_dgrid_wan_setup(base_path, config, prefix_groups):
    config["base_path"] = base_path

    for i in range(len(prefix_groups)):
        dv_config = deepcopy(config)
        dv_config["calc"] = "wan_setup_group"
        dv_config["prefix"] = str(i)
        enqueue(dv_config)

def _main():
    base = _base_dir()
    db_path = os.path.join(base, "c2dm.db")
    gconf = global_config()

    #c_sep, num_d_a, num_d_b = 3.0, 2, 2
    c_sep, num_d_a, num_d_b = None, None, None
    soc = True
    #symA, symB = "MoS2", "WS2"
    symA, symB = "MoS2", None
    dgrid = dgrid_inputs(db_path, symA, symB, c_sep, num_d_a, num_d_b, soc)
    base_path = os.path.expandvars(gconf["work_base"])
    write_dgrid(base_path, dgrid)

    #config = {"machine": "__local__", "wannier": True}
    #config = {"machine": "__local__", "wannier": True, "__local_mpi_cmd__": "mpirun"}
    if symA is not None and symB is not None:
        global_prefix = "{}_{}".format(symA, symB)
    elif symB is None:
        global_prefix = symA
    elif symA is None:
        global_prefix = symB
    else:
        raise ValueError("symA and symB are None")

    num_nodes = 4
    num_cores = 24*num_nodes
    config = {"machine": "ls5", "cores": num_cores, "nodes": num_nodes, "queue": "normal",
            "hours": 1, "minutes": 0, "wannier": True, "project": "A-ph9",
            "global_prefix": global_prefix, "max_jobs": 24,
            "pw_post_cores": 24}
    prefix_groups = write_dgrid_queuefiles(base_path, dgrid, config)

    submit_dgrid_wan_setup(base_path, config, prefix_groups)
    
if __name__ == "__main__":
    _main()
