import argparse
import math
import os
from copy import deepcopy
import numpy as np
import yaml
from tmd.bilayer.material import get_material
from tmd.bilayer.bilayer_util import _base_dir, global_config
from tmd.pwscf.build import build_qe, build_bands, build_pw2wan
from tmd.wannier.build import Winfile
from tmd.queue.queuefile import write_queuefile, write_launcherfiles, write_job_group_files
from tmd.queue.internal import enqueue

def dgrid_inputs(db_path, sym_A, sym_B=None, c_bulk=None, num_d_a=None, num_d_b=None, c_sep=None, soc=True, xc="lda", ordering="2H", iprelax=False, pp='nc', bands_only=False):
    if sym_B is None:
        d_as = [0.0]
        d_bs = [0.0]
    else:
        d_as = np.linspace(0.0, 1.0, num_d_a, endpoint=False)
        d_bs = np.linspace(0.0, 1.0, num_d_b, endpoint=False)

    inputs = {}
    atoms_A, atoms_B = None, None

    for d_a in d_as:
        for d_b in d_bs:
            material, atoms_A, atoms_B = get_material(db_path, sym_A, sym_B, c_bulk,
                    d_a, d_b, c_sep, soc, xc, atoms_A, atoms_B, ordering, pp)

            if iprelax or bands_only:
                if iprelax:
                    lat_facs = np.linspace(0.8, 1.2, 21)
                else:
                    lat_facs = [1.0]

                for lat_fac in lat_facs:
                    this_material = deepcopy(material)
                    this_material["latconst"] *= lat_fac
                    if iprelax:
                        if sym_B is None:
                            this_material["prefix"] = "{}_lat_{:.3f}_da_{:.3f}_db_{:.3f}".format(sym_A,
                                    lat_fac, d_a, d_b)
                        else:
                            this_material["prefix"] = "{}_{}_lat_{:.3f}_da_{:.3f}_db_{:.3f}".format(sym_A,
                                    sym_B, lat_fac, d_a, d_b)

                    inputs[(d_a, d_b, lat_fac)] = {"material": this_material}
                    for calc_type in ["scf", "bands"]:
                        qe_input = build_qe(this_material, calc_type)
                        inputs[(d_a, d_b, lat_fac)][calc_type] = qe_input

                    inputs[(d_a, d_b, lat_fac)]["bands_post"] = build_bands(this_material)
            else:
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

    calcs = ["scf", "nscf", "bands", "bands_post", "pw2wan"]
    for calc in calcs:
        if calc not in dv:
            continue

        in_path = os.path.join(wannier_dir_path, "{}.{}.in".format(prefix, calc))
        with open(in_path, 'w') as fp:
            fp.write(dv[calc])

    if "wannier" in dv:
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

    if config["iprelax"] is True or config["bands_only"] is True:
        bands_only_group_config = deepcopy(config)
        bands_only_group_config["calc"] = "bands_only"
        write_job_group_files(bands_only_group_config, prefix_groups)
        return prefix_groups

    wan_setup_group_config = deepcopy(config)
    wan_setup_group_config["calc"] = "wan_setup"
    write_job_group_files(wan_setup_group_config, prefix_groups)

    pw_post_group_config = deepcopy(config)
    pw_post_group_config["calc"] = "pw_post"
    pw_post_group_config["nodes"] = 1
    cores_per_node = int(config["cores"] / config["nodes"])
    pw_post_group_config["cores"] = cores_per_node
    write_job_group_files(pw_post_group_config, prefix_groups)

    launcher_config = deepcopy(config)
    launcher_config["prefix_list"] = prefix_list
    launcher_config["calc"] = "wan_run"
    num_systems = len(prefix_list)
    num_wannier_nodes = math.ceil(num_systems / (4*cores_per_node))
    num_wannier_cores = num_wannier_nodes * cores_per_node
    launcher_config["nodes"] = num_wannier_nodes
    launcher_config["cores"] = num_wannier_cores
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

def get_prefixes(base_path, global_prefix):
    prefix_groups = get_prefix_groups(base_path, global_prefix)
    prefixes = []
    for group in prefix_groups:
        for prefix in group:
            prefixes.append(prefix)

    return prefixes

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

    if config["iprelax"] is True or config["bands_only"] is True:
        _write_queuefile_calc(config, "bands_only")
    else:
        for calc in ["wan_setup", "pw_post", "wan_run"]:
            _write_queuefile_calc(config, calc)

def _write_queuefile_calc(config, calc):
    calc_config = deepcopy(config)
    calc_config["calc"] = calc
    write_queuefile(calc_config)

def submit_dgrid_pw(base_path, config, prefix_groups, calc_name):
    config["base_path"] = base_path

    for i in range(len(prefix_groups)):
        dv_config = deepcopy(config)
        dv_config["calc"] = "{}_group".format(calc_name)
        dv_config["prefix"] = str(i)
        enqueue(dv_config)

def _main():
    parser = argparse.ArgumentParser("Build and run calculation on grid of d's",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", action="store_true",
            help="Run calculation after making inputs")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base to run calculation")
    parser.add_argument("--symA", type=str, default="MoS2",
            help="Atomic composition of bottom layer")
    parser.add_argument("--symB", type=str, default="WS2",
            help="Atomic composition of top layer")
    parser.add_argument("--monolayer", action="store_true",
            help="Use monolayer system instead of bilayer")
    parser.add_argument("--ordering", type=str, default="2H",
            help="Ordering of atoms: '2H' -> BAB/ABA; '2H_top' -> BAB/BAB")
    parser.add_argument("--c_sep", type=float, default=None,
            help="Separation between layers (use value from bilayer of symA if not specified)")
    parser.add_argument("--soc", action="store_true",
            help="Use spin-orbit coupling")
    parser.add_argument("--xc", type=str, default="lda",
            help="Exchange-correlation functional (lda or pbe)")
    parser.add_argument("--num_d_a", type=int, default=3,
            help="Number of d's (shifts) along the a-axis")
    parser.add_argument("--num_d_b", type=int, default=3,
            help="Number of d's (shifts) along the b-axis")
    parser.add_argument("--pp", type=str, default="nc",
            help="Pseudopotential type ('nc' or 'paw')")
    parser.add_argument("--iprelax", action="store_true",
            help="Vary in-plane lattice constant to search for lowest energy")
    parser.add_argument("--bands_only", action="store_true",
            help="Run only scf and bands calc")
    args = parser.parse_args()

    symA, symB = args.symA, args.symB
    if args.monolayer:
        symB = None

    tmd_base = _base_dir()
    db_path = os.path.join(tmd_base, "c2dm.db")
    gconf = global_config()

    if not args.monolayer:
        c_bulk_values = {"MoS2": 12.296, "MoSe2": 12.939}
        c_bulk = c_bulk_values[symA]
    else:
        c_bulk = None

    dgrid = dgrid_inputs(db_path, symA, symB, c_bulk, args.num_d_a, args.num_d_b,
            c_sep=args.c_sep, soc=args.soc, xc=args.xc, ordering=args.ordering, iprelax=args.iprelax,
            pp=args.pp, bands_only=args.bands_only)

    base_path = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        base_path = os.path.join(base_path, args.subdir)

    # Recompiled QE to increase precision in bands.dat output file.
    # (The hard-coded default is 3 decimal places (1 meV) which is too small
    # for gap variation on the scale of 10 meV).
    #
    # Starting with QE 5.4.0,
    # in PP/src/bands.f90 changed:
    # Line 470: 10f8.3 --> 10f13.8
    # Line 476: 10f8.3 --> 10f13.8
    #
    # To compile QE on LS5, need to specify mpich scalapack.
    # With module 'espresso/5.4.0' loaded, compile with:
    # ./configure SCALAPACK_LIBS="-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64"
    # make pp
    if "qe_bands" in gconf:
        qe_bands_dir = os.path.expandvars(gconf["qe_bands"])
        qe_bands_path = os.path.join(qe_bands_dir, "bands.x")
    else:
        qe_bands_path = "bands.x"

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

    num_nodes = 1
    num_cores = 24*num_nodes
    config = {"machine": "ls5", "cores": num_cores, "nodes": num_nodes, "queue": "normal",
            "hours": 4, "minutes": 0, "wannier": True, "project": "A-ph9",
            "global_prefix": global_prefix, "max_jobs": 24,
            "outer_min": -10.0, "outer_max": 7.0,
            "inner_min": -8.0, "inner_max": 3.0,
            "subdir": args.subdir, "iprelax": args.iprelax, "bands_only": args.bands_only,
            "qe_bands": qe_bands_path}
    prefix_groups = write_dgrid_queuefiles(base_path, dgrid, config)

    if args.run:
        if not args.iprelax and not args.bands_only:
            calc_name = "wan_setup"
        else:
            calc_name = "bands_only"

        submit_dgrid_pw(base_path, config, prefix_groups, calc_name)

    
if __name__ == "__main__":
    _main()
