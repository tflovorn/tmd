from argparse import ArgumentParser
import os
from tmd.bilayer.dgrid import get_prefix_groups
from tmd.pwscf.parseScf import fermi_from_scf
from tmd.wannier.wannier_util import global_config
from tmd.wannier.build import Update_Disentanglement

def do_update_dis(base_path, prefix, outer_min, outer_max, inner_min, inner_max):
    wandir = os.path.join(base_path, prefix, "wannier")
    scf_path = os.path.join(wandir, "scf.out")
    E_Fermi = fermi_from_scf(scf_path)

    outer = [outer_min, outer_max]
    inner = [inner_min, inner_max]

    win_path = os.path.join(wandir, "{}.win".format(prefix))
    Update_Disentanglement(win_path, E_Fermi, outer, inner)

def _main():
    parser = ArgumentParser(description="Update disentanglement window in W90 input")
    parser.add_argument('--subdir', type=str, default=None,
            help="Subdirectory under work_base for all job dirs")
    parser.add_argument('--global_prefix', type=str, default=None,
            help="Run update_dis for all prefixes in subdir matching the given global prefix")
    parser.add_argument('prefix', type=str,
            help="Prefix of system to update")
    parser.add_argument('outer_min', type=float,
            help="Distance below E_F to start outer window")
    parser.add_argument('outer_max', type=float,
            help="Distance above E_F to stop outer window")
    parser.add_argument('inner_min', type=float,
            help="Distance below E_F to start inner window")
    parser.add_argument('inner_max', type=float,
            help="Distance above E_F to stop inner window")
    args = parser.parse_args()
    
    gconf = global_config()
    base_path = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        base_path = os.path.join(base_path, args.subdir)

    if args.global_prefix is None:
        do_update_dis(base_path, args.prefix, args.outer_min, args.outer_max,
                args.inner_min, args.inner_max)
    else:
        prefix_groups = get_prefix_groups(base_path, args.global_prefix)
        for group in prefix_groups:
            for prefix in group:
                do_update_dis(base_path, prefix, args.outer_min, args.outer_max,
                        args.inner_min, args.inner_max)

if __name__ == "__main__":
    _main()
