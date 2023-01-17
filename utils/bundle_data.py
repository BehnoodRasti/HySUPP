"""
Script to create a single .mat files from multiple .mat files to be used in the toolbox

Usage:
$ python -m utils.bundle_data --name Sim1 -Y Y_clean -E E -A XT [-D EE]

It assumes that Sim1 is located in the $PROJECT_ROOT/data folder

"""

import numpy as np
import scipy.io as sio

import os
import argparse

as_matlab = lambda key: f"{key}.mat"


def main(args):

    base_dir = f"./data/{args.name}/"

    Y = sio.loadmat(os.path.join(base_dir, as_matlab(args.hsi)))[args.hsi]
    E = sio.loadmat(os.path.join(base_dir, as_matlab(args.endmembers)))[args.endmembers]
    A = sio.loadmat(os.path.join(base_dir, as_matlab(args.abundances)))[args.abundances]
    if args.dictionary is not None:
        D = sio.loadmat(os.path.join(base_dir, as_matlab(args.dictionary)))[
            args.dictionary
        ]

    if args.height is not None:
        H = args.H
    if args.width is not None:
        W = args.W

    assert len(E.shape) == 2
    if E.shape[0] < E.shape[1]:  # p x L
        E = E.T  # L x p

    L, p = E.shape

    if len(Y.shape) == 3:
        if Y.shape[0] == L:
            if args.height is not None:
                assert Y.shape[1] == H
            if args.width is not None:
                assert Y.shape[2] == W
            H, W = Y.shape[1], Y.shape[2]
            N = H * W
            Y = Y.reshape(L, N)
        elif Y.shape[2] == L:
            if args.height is not None:
                assert Y.shape[1] == H
            if args.width is not None:
                assert Y.shape[2] == W
            H, W = Y.shape[0], Y.shape[1]
            N = H * W
            Y = Y.reshape(N, L).T
        else:
            raise ValueError("Corner case not handled...")

    elif len(Y.shape) == 2:
        if Y.shape[0] != L:  # N x L
            N = Y.shape[0]
            Y = Y.T  # L x N

    else:
        raise ValueError("Invalid shape for Y...")

    breakpoint()

    if len(A.shape) == 3:
        if A.shape[0] == p:
            assert A.shape[1] * A.shape[2] == N
            A = A.reshape(p, A.shape[1] * A.shape[2])
        elif A.shape[2] == p:
            A = A.transpose((2, 0, 1))
            A = A.reshape(p, N)
        else:
            raise ValueError("Corner case not handled...")

    elif len(A.shape) == 2:
        if A.shape[0] != p:  # N x p
            A = A.T  # p x N

    else:
        raise ValueError("Invalid shape for A...")

    data = {
        "Y": Y,
        "E": E,
        "A": A,
        "H": H,
        "W": W,
        "p": p,
        "L": L,
    }

    if args.dictionary is not None:
        data["D"] = D
        data["M"] = D.shape[0] if D.shape[1] == L else D.shape[1]

    sio.savemat(f"./data/{args.name}.mat", data)


if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description="Data bundler")
    parser.add_argument("--name", required=True)
    parser.add_argument("--hsi", "-Y", required=True)
    parser.add_argument("--endmembers", "-E", required=True)
    parser.add_argument("--abundances", "-A", required=True)
    parser.add_argument("--dictionary", "-D", required=False, default=None)
    parser.add_argument("--height", "-H", required=False, default=None)
    parser.add_argument("--width", "-W", required=False, default=None)

    args = parser.parse_args()

    main(args)
