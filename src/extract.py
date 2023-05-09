"""
Endmembers extraction techniques main source file
"""
import logging

from hydra.utils import instantiate

from src.model.extractors import SISAL, VCA, SiVM
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as sio
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    logger.debug("Endmember extraction --- start...")

    # Instantiate objects
    extractors = [
        VCA(),
        SiVM(),
        # SISAL(),
    ]

    hsi = instantiate(
        cfg.data,
        noise=None,
    )

    Y, p, H, W, N, L = hsi.get_endmembers_extraction_input()

    # Rescale Y
    Y = (Y - Y.min()) / (Y.max() - Y.min())

    Y_img = Y.reshape(L, H, W)

    rows_division = 4
    cols_division = 4
    nH = H // rows_division
    nW = W // cols_division

    candidates = []

    for ii in range(rows_division):
        for jj in range(cols_division):

            # Sub-images creation
            sub_img = Y_img[:, nH * ii : nH * (ii + 1), nW * jj : nW * (jj + 1)]
            sub_img_flattened = sub_img.reshape(L, -1)

            # Endmembers extraction
            for extractor in extractors:
                candidates.append(
                    extractor.extract_endmembers(
                        Y=sub_img_flattened,  # Use sub-images here
                        p=p,
                        seed=0,
                        snr_input=100,
                    )
                )
    # Aggregate endmembers
    candidates = np.hstack(candidates)  # (L, p x rows_div x cols_div x nb_extractors)
    kmeans_input = candidates.T  # (n_samples, n_features)

    # Use k-means
    kmeans = KMeans(n_clusters=p).fit(kmeans_input)
    indices = kmeans.labels_

    # Dictionary creation
    logger.info(f"Dictionary shape => {candidates.shape}")

    color_dict = {0: "b", 1: "g", 2: "r", 3: "k"}
    for cc in range(len(indices)):
        plt.plot(kmeans_input[cc], c=color_dict[indices[cc]])
    plt.show()

    M = len(indices)

    breakpoint()
    sio.savemat(
        to_absolute_path(f"data/{cfg.data.name}-{M}.mat"),
        {"D": candidates, "index": indices},
    )

    logger.debug("Endmembers extraction --- end...")
