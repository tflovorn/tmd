import os
import shutil
from argparse import ArgumentParser
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.bilayer_util import global_config

def _main():
    parser = ArgumentParser("wfc cleanup")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument('global_prefix', type=str,
            help="System for which wannier/ .save directories will be removed")
    parser.add_argument('--confirm', action='store_true',
            help="Must specify --confirm to confirm .save removal is desired")
    args = parser.parse_args()

    if not args.confirm:
        return

    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        work = os.path.join(work, args.subdir)

    prefixes = get_prefixes(work, args.global_prefix)

    for prefix in prefixes:
        save_path = os.path.join(work, prefix, "wannier", "{}.save".format(prefix))
        shutil.rmtree(save_path)

if __name__ == "__main__":
    _main()
