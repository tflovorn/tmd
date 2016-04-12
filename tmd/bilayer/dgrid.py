import numpy as np
import os
import yaml
from tmd.bilayer.material import get_material
from tmd.pwscf.build import build_qe, build_bands

def dgrid_inputs(db_path, sym_A, sym_B, c_sep, num_d_a, num_d_b):
    d_as = np.linspace(0.0, 1.0, num_d_a, endpoint=False)
    d_bs = np.linspace(0.0, 1.0, num_d_b, endpoint=False)

    inputs = {}

    for d_a in d_as:
        for d_b in d_bs:
            material = get_material(db_path, sym_A, sym_B, c_sep, d_a, d_b)

            inputs[(d_a, d_b)] = {"material": material}
            for calc_type in ["scf", "nscf", "bands"]:
                qe_input = build_qe(material, calc_type)
                inputs[(d_a, d_b)][calc_type] = qe_input

            inputs[(d_a, d_b)]["bands_post"] = build_bands(material)

    return inputs

def write_dgrid(base_path, dgrid):
    for dk, dv in dgrid.items():
        _write_dv(base_path, dv)

def _write_dv(base_path, dv):
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

    with open(scf_path, 'w') as fp:
        fp.write(dv["scf"])

    with open(nscf_path, 'w') as fp:
        fp.write(dv["nscf"])

    with open(bands_path, 'w') as fp:
        fp.write(dv["bands"])

    with open(bands_post_path, 'w') as fp:
        fp.write(dv["bands_post"])

def _main():
    db_path = "c2dm.db"
    c_sep = 3.0

    dgrid = dgrid_inputs(db_path, "MoS2", "WS2", c_sep, 2, 2)

    base_path = os.path.expandvars("$HOME/tmd_run/MoS2_WS2")
    write_dgrid(base_path, dgrid)
    
    #for dk, dv in dgrid.items():
    #    for k, v in dv.items():
    #        print(dk, k)
    #        print(v)

if __name__ == "__main__":
    _main()
