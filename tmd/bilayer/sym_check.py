import argparse
import numpy as np
import yaml

def Rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

# d_recip and Rot_center_recip are column vectors;
# returns partner column vector
def rotate_and_shift(D, Rot, Rot_center_recip, d_recip):
    Rot_center_cart = np.dot(D, Rot_center_recip.T)
    d_cart = np.dot(D, d_recip.T)
    p = np.dot(Rot, d_cart - Rot_center_cart) + Rot_center_cart
    p_recip = np.dot(np.linalg.inv(D), p)
    p_in_cell = np.array([p_recip[0] % 1, p_recip[1] % 1])
    return p_in_cell.T

def find_d(gap_data, d):
    eps = 2e-3
    for check_d, gaps in gap_data:
        if abs(check_d[0] - d[0]) < eps and abs(check_d[1] - d[1]) < eps:
            return gaps

    return None

def sym_check(D, gap_data):
    all_gap_diffs = []
    R2 = Rotation(2*np.pi/3)
    R4 = Rotation(4*np.pi/3)
    A, B, C = np.array([0.0, 0.0]), np.array([1/3, 2/3]), np.array([2/3, 1/3])
    Rot_center_recip = A
    for d, gaps in gap_data:
        d_partner_2 = rotate_and_shift(D, R2, Rot_center_recip, np.array(d))
        d_partner_4 = rotate_and_shift(D, R4, Rot_center_recip, np.array(d))
        gaps_partner_2 = find_d(gap_data, d_partner_2)
        gaps_partner_4 = find_d(gap_data, d_partner_4)
        if gaps_partner_2 is None or gaps_partner_4 is None:
            raise ValueError("failed to find partner")

        gap_diffs = list(map(abs, [gaps["0/0"] - gaps_partner_2["0/0"],
                gaps["0/0"] - gaps_partner_4["0/0"],
                gaps_partner_2["0/0"] - gaps_partner_4["0/0"]]))
        all_gap_diffs.extend(gap_diffs)

        #print(d, d_partner_2, d_partner_4, gap_diffs)

    print(max(all_gap_diffs))
    print(sum(all_gap_diffs)/len(all_gap_diffs))

def _main():
    parser = argparse.ArgumentParser("check C3 symmetry of d")
    parser.add_argument("gap_file_path", type=str,
            help="Path to YAML file with gap data as emitted by gaps.py")
    args = parser.parse_args()

    D = np.array([[1/2, 1/2],
                 [-np.sqrt(3)/2, np.sqrt(3)/2]])

    with open(args.gap_file_path, 'r') as fp:
        gap_data = yaml.load(fp.read())

    sym_check(D, gap_data)

if __name__ == "__main__":
    _main()
