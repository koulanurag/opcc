"""
Thanks to "Aayam Shrestha" (https://github.com/idigitopia)
"""

import numpy as np
from sklearn.neighbors import KDTree as RawKDTree
from tqdm import tqdm


def iter_batch(iterable, n=1):
    _len = len(iterable)
    for ndx in range(0, _len, n):
        yield iterable[ndx : min(ndx + n, _len)]


def v_iter(iterator, verbose, message=""):
    """
    Returns a verbose iterator i.e. tqdm enabled iterator if verbose is True.
    It also attaches the passed message to the iterator.
    """
    if verbose:
        vb_iterator = tqdm(iterator)
        vb_iterator.set_description(message)
    else:
        vb_iterator = iterator

    return vb_iterator


# KD Tree helper function
class KDTree:
    """
    Class to contain all the KD Tree related logics.
    - Builds the index and inverseIndex for the vectors passed as the
      vocabulary for knn
    - can get k/1 NN or k/1 NN of a batch of passed query vectors.
    """

    def __init__(self, all_vectors):
        self.s2i, self.i2s = self._gen_vocab(all_vectors)
        self.KDtree = RawKDTree(np.array(list(self.s2i.keys())))

    def get_knn(self, s, k):
        return self.get_knn_batch(np.array([s]), k)[0]

    def get_nn(self, s):
        return list(self.get_knn_batch(np.array([s]), 1)[0])[0]

    def get_nn_batch(self, s_batch):
        return [list(knnD)[0] for knnD in self.get_knn_batch(s_batch, 1)]

    def get_nn_sub_batch(self, s_batch):
        return [list(knnD)[0] for knnD in self.get_knn_sub_batch(s_batch, 1)]

    def get_knn_idxs(self, s, k):
        return self.get_knn_idxs_batch(np.array([s]), k)[0]

    def get_nn_idx(self, s):
        return list(self.get_knn_idxs_batch(np.array([s]), 1)[0])[0]

    def get_nn_idx_batch(self, s_batch):
        return [list(knnD)[0] for knnD in self.get_knn_idxs_batch(s_batch, 1)]

    def get_nn_idx_sub_batch(self, s_batch):
        return [list(knnD)[0] for knnD in self.get_knn_idxs_sub_batch(s_batch, 1)]

    def _gen_vocab(self, all_vectors):
        """
        generate index mappings and inverse index mappings.
        """

        s2i = {tuple(s): i for i, s in enumerate(all_vectors)}
        i2s = {i: tuple(s) for i, s in enumerate(all_vectors)}
        return s2i, i2s

    def get_knn_batch(self, s_batch, k):
        """
        input: a list of query vectors.
        output: a list of k-nn tuples for each query vector.
        """

        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        get_nn_dict = lambda dists, idxs: {
            self.i2s[int(idx)]: dist for dist, idx in zip(dists, idxs)
        }
        nn_dict_list = [
            get_nn_dict(dists, idxs) for dists, idxs in zip(dists_b, idxs_b)
        ]
        return nn_dict_list

    def get_knn_idxs_batch(self, s_batch, k):
        """
        input: a list of query vectors.
        output: a list of k-nn idxs for each query vector.
        """

        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        get_nn_dict = lambda dists, idxs: {idx: dist for dist, idx in zip(dists, idxs)}
        nn_dict_list = [
            get_nn_dict(dists, idxs) for dists, idxs in zip(dists_b, idxs_b)
        ]
        return nn_dict_list

    # Get knn with smaller batch sizes. | useful when passing large batches.
    def get_knn_sub_batch(self, s_batch, k, batch_size=256, verbose=True, message=None):
        """
        Get knn with smaller batch sizes. It's useful when passing large
        batches.

        input: a large list of query vectors.
        output: a large list of k-nn tuples for each query vector.
        """

        nn_dict_list = []
        for small_batch in v_iter(
            iter_batch(s_batch, batch_size), verbose, message or "getting NN"
        ):
            nn_dict_list.extend(self.get_knn_batch(small_batch, k))
        return nn_dict_list

    def get_knn_idxs_sub_batch(
        self, s_batch, k, batch_size=256, verbose=True, message=None
    ):
        nn_dict_list = []
        for small_batch in v_iter(
            iter_batch(s_batch, batch_size), verbose, message or "getting NN Idxs"
        ):
            nn_dict_list.extend(self.get_knn_idxs_batch(small_batch, k))
        return nn_dict_list

    @staticmethod
    def normalize_distances(knn_dist_dict, delta=None):
        delta = delta
        all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
        all_knn_probs = {
            nn: knn_kernel / sum(all_knn_kernels.values())
            for nn, knn_kernel in all_knn_kernels.items()
        }
        return all_knn_probs
