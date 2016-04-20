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
    gconf = global_config()
    base_path = os.path.expandvars(gconf["work_base"])
    global_prefix = "MoS2"
    calc = "pw_post"
    prefix_groups = get_prefix_groups(base_path, global_prefix)

    config = {"machine": "ls5", "cores": 24, "nodes": 1, "queue": "normal",
            "hours": 1, "minutes": 0, "wannier": True, "project": "A-ph9",
            "global_prefix": global_prefix, "max_jobs": 24}

    submit_dgrid_pw_post(base_path, config, prefix_groups)

if __name__ == "__main__":
    _main()
