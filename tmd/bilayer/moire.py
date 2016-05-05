import os
import itertools
import numpy as np
from tmd.bilayer.dfourier import H_klat_Glat
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.plot_ds import ds_from_prefixes, wrap_cell, sorted_d_group, orbital_index, get_atom_order
from tmd.pwscf.parseScf import D_from_scf

def moire_Gs(D_2D, epsilon, theta):
    '''Returns a list with the minimal set of moire reciprolcal lattice vectors
    required to tile the approximant Brillouin zone.

    Epsilon and theta must be such that the layers are commensurate in
    order for the returned lattice vectors to tile the full approximant
    Brillouin zone.

    D_2D = 2x2 matrix with columns giving the lattice vectors of the
            approximant unit cell;
    epsilon = lattice constant scaling for the moire lattice;
    theta = interlayer rotation for the moire lattice.
    '''
    eps = 1e-9
    #TODO - is note about commensurate requirement correct?
    # Need commensurate in order for moire Gtildes to form a superset of the
    # approximant Gs.
    S = np.array([[epsilon, theta], [-theta, epsilon]])
    R = 2*np.pi*np.linalg.inv(D_2D)
    RinvT = (1/(2*np.pi))*D_2D.T

    # Since Gtilde = SG is a linear transformation, we shouldn't repeat
    # Gtilde values. Can use a list here instead of a set.
    # (Need other properties? Want lattice vectors to be linearly independent
    # which is always the case when forming a lattice of correct dimensionality.)
    Gts = []
    found_all_G1, found_all_G2 = False, False
    G1, G2 = 0, 0
    while not found_all_G1:
        found_all_G2 = False
        G2 = 0
        while not found_all_G2:
            GT = np.array([G1, G2])
            G_cart = np.dot(GT, R).T
            Gtilde_cart = np.dot(S, G_cart)
            Gtilde = np.dot(RinvT, Gtilde_cart)
            Gt1, Gt2 = Gtilde[0], Gtilde[1]

            point_ok = Gt1 >= 0 and Gt1 < 1 and Gt2 >= 0 and Gt2 < 1
            not_near_1 = abs(Gt1 - 1) > eps and abs(Gt2 - 1) > eps

            if point_ok and not_near_1:
                Gts.append([Gt1, Gt2])
            elif G2 == 0:
                found_all_G1, found_all_G2 = True, True
            else:
                found_all_G2 = True
            G2 += 1

        G1 += 1

    return Gts

def _eval_G_cutoff(G1, G2, D_2D, Gcut, Gcut_cart):
    if not Gcut_cart:
        mag = G1**2 + G2**2
    else:
        R = 2*np.pi*np.linalg.inv(D_2D)
        GT_cart = np.dot(GT, R)
        mag = GT_cart[0]**2 + GT_cart[1]**2

    if mag > Gcut and G2 == 0:
        return True, True
    elif mag > Gcut:
        return False, True
    else:
        return False, False

def approximant_Gs(D_2D, Gcut, Gcut_cart):
    Gs = []
    G1, G2 = 0, 0
    finished_G1, finished_G2 = False, False
    while not finished_G1:
        finished_G2 = False
        G2 = 0
        while not finished_G2:
            finished_G1, finished_G2 = _eval_G_cutoff(G1, G2, D_2D, Gcut, Gcut_cart)
            if not finished_G2:
                Gs.append([G1, G2])
                # TODO - changed this from and - check that or is correct here
                if G1 != 0 or G2 != 0:
                    Gs.append([-G1, -G2])
            G2 += 1

        G1 += 1

    return Gs

def moire_Hamiltonian(dps, ktildes, D_2D, epsilon, theta, Gcut, Gcut_cart=False):
    moire_G_vecs = moire_Gs(D_2D, epsilon, theta)
    print(moire_G_vecs)
    approximant_G_vecs = approximant_Gs(D_2D, Gcut, Gcut_cart)
    print(approximant_G_vecs)
    num_Gt = len(moire_G_vecs)
    # Want to take Cartesian product of: (ktildes, moire_Gs, moire_Gs, approximant_Gs).
    # This gives a list of (ktilde, Gtilde, Gtilde_prime, G2) tuples.
    # Then convert this to input to H_A(k:G).
    HA_input_components = list(itertools.product(ktildes, moire_G_vecs, moire_G_vecs, approximant_G_vecs))
    kGs = []

    S = np.array([[epsilon, theta], [-theta, epsilon]])
    Sinv = np.linalg.inv(S)

    for ktilde, Gtilde, Gtilde_prime, G2 in HA_input_components:
        Gt_expanded = list(np.array(Gtilde))
        Gt_expanded.append(0.0)
        k = tuple(np.array(ktilde) + Gt_expanded)

        Gt_arg = np.array(Gtilde_prime) - np.array(Gtilde) + np.array(G2)
        G = tuple(np.dot(Sinv, Gt_arg))
        kGs.append([k, G])

    Hks = H_klat_Glat(dps, kGs)
    num_bands = Hks[0].shape[0]

    Hk_moires = []
    num_bands_moire = num_bands * num_Gt
    for i in range(len(ktildes)):
        Hk_moires.append(np.zeros([num_bands_moire, num_bands_moire], dtype=np.complex128))

    for Hk, (ktilde, Gtilde, Gtilde_prime, G2) in zip(Hks, HA_input_components):
        Gt_index = moire_G_vecs.index(Gtilde)
        Gtp_index = moire_G_vecs.index(Gtilde_prime)
        k_index = ktildes.index(ktilde)

        Gt_start, Gt_stop = Gt_index*num_bands, (Gt_index + 1)*num_bands
        Gtp_start, Gtp_stop = Gtp_index*num_bands, (Gtp_index + 1)*num_bands
        # Add contribution to this H(k) [Gt, Gtp] block corresponding to G2.
        Hk_moires[k_index][Gt_start:Gt_stop, Gtp_start:Gtp_stop] += Hk

    return Hk_moires

def _main():
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    
    global_prefix = "MoS2_WS2"
    prefixes = get_prefixes(work, global_prefix)
    ds = ds_from_prefixes(prefixes)
    ds, prefixes = wrap_cell(ds, prefixes)
    dps = sorted_d_group(ds, prefixes)

    scf_0_path = os.path.join(work, prefixes[0], "wannier", "scf.out")
    D = D_from_scf(scf_0_path)
    D_2D = D[0:2, 0:2]

    ktildes = [(1/3, 1/3, 0)]
    epsilon = 1/3
    theta = 0.0
    Gcut = 30
    Gcut_cart = False

    Hk_moires = moire_Hamiltonian(dps, ktildes, D_2D, epsilon, theta, Gcut, Gcut_cart)

if __name__ == "__main__":
    _main()
