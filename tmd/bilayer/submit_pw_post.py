import os
from copy import deepcopy
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.dgrid import get_prefix_groups
from tmd.queue.internal import enqueue

def submit_dgrid_pw_post(base_path, config, prefix_groups):
    config["base_path"] = base_path

    for i in range(len(prefix_groups)):
        dv_config = deepcopy(config)
        dv_config["calc"] = "pw_post_group"
        dv_config["prefix"] = str(i)
        enqueue(dv_config)

def _main():
    parser = argparse.ArgumentParser("Run postprocessing to set up Wannier90 calculation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base to run calculation")
    parser.add_argument("--global_prefix", type=str, default="MoS2_WS2",
            help="Prefix for calculation")
    args = parser.parse_args()

    gconf = global_config()
    base_path = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        base_path = os.path.join(base_path, subdir)

    calc = "pw_post"
    prefix_groups = get_prefix_groups(base_path, args.global_prefix)

    config = {"machine": "ls5", "cores": 24, "nodes": 1, "queue": "normal",
            "hours": 1, "minutes": 0, "wannier": True, "project": "A-ph9",
            "global_prefix": args.global_prefix, "max_jobs": 24}

    submit_dgrid_pw_post(base_path, config, prefix_groups)

if __name__ == "__main__":
    _main()
