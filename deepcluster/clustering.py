# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): index of data
    #     Returns:
    #         tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
    #     """
    #     path, pseudolabel = self.imgs[index]
    #     img = pil_loader(path)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img, pseudolabel

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        # import pdb; pdb.set_trace()

        def _sample(path):
            if isinstance(path, str):
                img = pil_loader(path)
            else:
                img = path
            
            if self.transform is not None:
                img = self.transform(img)
            return img

        img = np.stack([_sample(p) for p in path])

        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=64, mat=None):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    if mat is None:
        mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
        mat.train(npdata)

    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata, mat


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


def cluster_assign(images_lists, dataset, transform):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    # normalize = transforms.Normalize(mean=[0.29501004, 0.34140844, 0.3667595 ],
    #                                 std=[0.16179572, 0.1323428 , 0.1213659 ])

    # normalize = transforms.Normalize(mean=mean, std=std)

    # t = transforms.Compose([
    #     transforms.Resize([128, 128]),
    #     # transforms.RandomResizedCrop(128, scale=(0.8, 0.9), ratio=(1, 1),),
    #     # transforms.Lambda(
    #     #     # lambda crops: torch.stack([ToTensor()(crop) for crop in crops])
    #     #     lambda x: transforms.functional.crop(x, 128/2, 128/4, 128/2, 128/2)
    #     # ),
    #     transforms.Resize([128, 128]),
    #     transforms.ToTensor(),
    #     normalize])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # t = transforms.Compose([transforms.RandomResizedCrop(224),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, transform)


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.nredo = 10

    clus.max_points_per_centroid = int(3*(n_data/nmb_clusters)) #10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # import pdb; pdb.set_trace()

    # perform the training
    clus.train(x, index)
    
    D, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)

    if verbose:
        print('k-means loss evolution: {0}'.format(losses))
    
    # import pdb; pdb.set_trace()

    return [int(n[0]) for n in I], [float(n[0]) for n in D], losses[-1], clus, index, flat_config


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]

import sys
sys.path.append('../')
sys.path.append('../mixture')
import mixture as mix
import mixture.models as mix_models

class GMM:
    def __init__(self, k, group=10):
        self.k = k
        self.group = group

    def get_clus(self, verbose):
        clus = mix_models.GaussianMixture(n_components=self.k, covariance_type='full',
            verbose=verbose, verbose_interval=1, tol=0.01)
        return clus

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb, self.mat = preprocess_features(data)

        # cluster the data
        clus = self.get_clus(verbose)
        clus.fit(xb, group=self.group if self.group > 1 else None)

        clus.centroids = clus.means_
        D = clus.predict_proba(xb, group=self.group if self.group > 1 else None)
        
        # import pdb; pdb.set_trace()
        loss = clus.score(xb)
        I, D = D.argmax(axis=1), D.max(axis=1)

        self.clus = clus

        self.images_lists = [[] for i in range(self.k)]
        self.images_dists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)
            self.images_dists[I[i]].append((i, D[i]))

        if verbose:
            print('gmm time: {0:.0f} s'.format(time.time() - end))

        return loss

class BGMM(GMM):
    def get_clus(self):
        clus = mix_models.BayesianGaussianMixture(n_components=self.k,
            covariance_type='full', weight_concentration_prior=1e3,
            weight_concentration_prior_type='dirichlet_process', tol=0.1)
        return clus

class Kmeans:
    def __init__(self, k, group=10):
        self.k = k
        self.group = group

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb, self.mat = preprocess_features(data)

        # cluster the data

        # I, D, loss, self.clus, self.index, self.flat_config = run_kmeans(xb, self.k, verbose)
        # import pdb; pdb.set_trace()


        clus = mix_models.k_means_.KMeans(n_clusters=self.k, n_init=1,
            precompute_distances=True, max_iter=50,
            algorithm='full',
            group=self.group if self.group > 1 else None).fit(xb)
        D = clus.transform(xb)
        loss = clus.inertia_
        clus.centroids = clus.cluster_centers_
        I, D = D.argmin(axis=1), -D.min(axis=1)

        self.clus = clus

        # import pdb; pdb.set_trace()
        # clus = kmeans.KMeans(n_clusters=self.n_components, n_init=1,
        #                            random_state=random_state, algorithm='full',
        #                            group=self.group).fit(xb)
                                   
        
        self.images_lists = [[] for i in range(self.k)]
        self.images_dists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)
            self.images_dists[I[i]].append((i, D[i]))

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


##############################################################################################


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwith of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if (i == 200 - 1):
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                vi = best_vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC():
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwith of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True, group=10):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons
        self.group = group

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        xb, mat = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0
