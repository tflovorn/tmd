import numpy as np
from tmd.bilayer.material import get_material
from tmd.pwscf.build import build_qe

def dgrid_inputs(db_path, sym_A, sym_B, c_sep, num_d_a, num_d_b):
    d_as = np.linspace(0.0, 1.0, num_d_a, endpoint=False)
    d_bs = np.linspace(0.0, 1.0, num_d_b, endpoint=False)

    inputs = {}

    for d_a in d_as:
        for d_b in d_bs:
            material = get_material(db_path, sym_A, sym_B, c_sep, d_a, d_b)
            prefix = "{}_{}_da_{:.3f}_db_{:.3f}".format(sym_A, sym_B, d_a, d_b)

            inputs[(d_a, d_b)] = {}
            for calc_type in ["scf", "nscf", "bands"]:
                qe_input = build_qe(material, prefix, calc_type)
                inputs[(d_a, d_b)][calc_type] = qe_input

    return inputs

def _main():
    db_path = "c2dm.db"
    c_sep = 3.0

    dgrid = dgrid_inputs(db_path, "MoS2", "WS2", c_sep, 2, 2)
    
    for dk, dv in dgrid.items():
        for k, v in dv.items():
            print(dk, k)
            print(v)

if __name__ == "__main__":
    _main()
