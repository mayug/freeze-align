'''Code taken from https://github.com/mayug/0-shot-llm-vision/blob/main/src/utils.py'''

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch
import math
from scipy.optimize import quadratic_assignment
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def centering(K, device):
    n = K.shape[0]
    unit = torch.ones(n, n).to(device)
    I = torch.eye(n).to(device)
    H = I - unit / n

    return torch.matmul(torch.matmul(H, K), H)

def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, device, sigma):
    return torch.trace(torch.matmul(centering(rbf(X, sigma), device), centering(rbf(Y, sigma), device)))

def kernel_CKA(X, Y, device, sigma=None):
    hsic = kernel_HSIC(X, Y, device, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, device, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, device, sigma))

    return hsic / (var1 * var2)

def linear_HSIC(X, Y, device):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X, device) * centering(L_Y, device))

def linear_CKA(X, Y, device):
    hsic = linear_HSIC(X, Y, device)
    var1 = torch.sqrt(linear_HSIC(X, X, device))
    var2 = torch.sqrt(linear_HSIC(Y, Y, device))

    return hsic / (var1 * var2)

def centered_scaled_corr(X, device):
    L_X = torch.matmul(X, X.T)
    var1 = torch.sqrt(linear_HSIC(X, X, device))
    C = centering(L_X, device)
    C = C / (var1)
    return C    


def centered_scaled_corr_rbf(X, device, sigma=None):
    rbf_X = rbf(X, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, device, sigma))
    C = centering(rbf_X, device)
    C = C / (var1)
    return C

def get_data_sep(seed1, seed2, 
                 base_samples, 
                 query_samples,
                 source_base_large,
                 target_base_large,
                 source_query_large,
                 target_query_large,
                 source_base_cluster,
                 target_base_cluster,
                 clustering_mode=0,
                 same = False,
                 stretch = False):

        source_total = torch.cat([source_base_large, source_query_large], dim=0)
        target_total = torch.cat([target_base_large, target_query_large], dim=0)

        if stretch:
            source_total = stretch_representations(source_total)
            target_total = stretch_representations(target_total)

        source_base_large = source_total[:source_base_large.shape[0]]
        source_query_large = source_total[source_base_large.shape[0]:]

        target_base_large = target_total[:target_base_large.shape[0]]
        target_query_large = target_total[target_base_large.shape[0]:]

        if clustering_mode == 0:
            np.random.seed(seed1)

            choose = np.random.choice(source_base_large.shape[0], base_samples, replace=False)

            source_base = source_base_large[choose]
            target_base = target_base_large[choose]
        elif clustering_mode == 1: # source clustering

            choose = source_base_cluster

            source_base = source_base_large[choose]
            target_base = target_base_large[choose]
        elif clustering_mode == 2: # source clustering

            choose = target_base_cluster

            source_base = source_base_large[choose]
            target_base = target_base_large[choose]

        if not same:
            np.random.seed(seed2)
            choose = np.random.choice(source_query_large.shape[0], query_samples, replace=False)
            source_query = source_query_large[choose]
            target_query = target_query_large[choose]
        else:
            mask = np.ones(source_query_large.shape[0], dtype=bool)
            mask[choose] = False

            np.random.seed(seed2)
            query_choose = np.random.choice(source_query_large.shape[0] - base_samples, query_samples, replace=False)
            
            source_query = source_query_large[mask][query_choose]
            target_query = target_query_large[mask][query_choose]

        return source_base, target_base, source_query, target_query

class KernelCKA:
    def __init__(self, n, device):
        self.H = self.get_centering_matrix(n, device)

    def kernel_HSIC(self, X, Y, sigma=None):
         return torch.sum(self.centering(rbf(X, sigma))*self.centering(rbf(Y, sigma)))
    
    def centering(self, K):
        return torch.matmul(torch.matmul(self.H, K), self.H)
    
    def get_centering_matrix(self, n, device):
        unit = torch.ones(n, n).to(device)
        I = torch.eye(n).to(device)
        H = I - unit / n
        return H

    def calculate(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        # print(hsic)
        # asd
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

def linear_HSIC(X, Y, device):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X, device) * centering(L_Y, device))

class LinearCKA:
    def __init__(self, n, device):
        self.H = self.get_centering_matrix(n, device)

    def linear_HSIC(self, X, Y, sigma=None):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X)*self.centering(L_Y))
    
    def centering(self, K):
        return torch.matmul(torch.matmul(self.H, K), self.H)
    
    def get_centering_matrix(self, n, device):
        unit = torch.ones(n, n).to(device)
        I = torch.eye(n).to(device)
        H = I - unit / n
        return H

    def calculate(self, X, Y, sigma=None):
        hsic = self.linear_HSIC(X, Y, sigma)
        # print(hsic)
        # asd
        var1 = torch.sqrt(self.linear_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


def linear_baseline(source_base, target_base, source_query, target_query, device):
    source_base = source_base.cpu().numpy()
    target_base = target_base.cpu().numpy()
    source_query = source_query.cpu().numpy()
    target_query = target_query.cpu().numpy()
    
    model = LinearRegression(copy_X=True, fit_intercept=False)
    model.fit(source_base, target_base)

    source_query_projected = model.predict(source_query)

    source_query_projected_norm = source_query_projected / np.linalg.norm(source_query_projected, axis=1, keepdims=True)
    target_query_norm = target_query / np.linalg.norm(target_query, axis=1, keepdims=True)

    graph = cosine_similarity(source_query_projected_norm, target_query_norm)
    return torch.tensor(graph)

def cos_baseline(source_base, target_base, source_query, target_query, device):
    source_base = source_base.cpu().numpy()
    target_base = target_base.cpu().numpy()
    source_query = source_query.cpu().numpy()
    target_query = target_query.cpu().numpy()
    
    source_query_norm = source_query / np.linalg.norm(source_query, axis=1, keepdims=True)
    target_query_norm = target_query / np.linalg.norm(target_query, axis=1, keepdims=True)

    graph = cosine_similarity(source_query_norm, target_query_norm)
    return torch.tensor(graph)

def linear_local_CKA(source_base, target_base, source_query, target_query, device):
    cka = LinearCKA(target_base.shape[0]+1, device)
    graph = []
    for source_ind in tqdm(range(source_query.shape[0])):
        source = source_query[source_ind:source_ind+1, :]
        row = []
        for target_ind in range(target_query.shape[0]):
            target = target_query[target_ind:target_ind+1, :]
    
            row.append(cka.calculate(torch.cat((source, source_base), axis = 0), torch.cat((target, target_base), axis = 0)))
        graph.append(row)
    return torch.tensor(graph)

def kernel_local_CKA(source_base, target_base, source_query, target_query, device):
    cka = KernelCKA(target_base.shape[0]+1, device)
    
    graph = []
    
    for source_ind in tqdm(range(source_query.shape[0])):
        source = source_query[source_ind:source_ind+1, :]
        row = []
        for target_ind in range(target_query.shape[0]):
            target = target_query[target_ind:target_ind+1, :]
    
            row.append(cka.calculate(torch.cat((source, source_base), axis = 0), torch.cat((target, target_base), axis = 0)))
        graph.append(row)
    return torch.tensor(graph)

def get_retrieval(graph):
    if graph is None:
        return None, None, None
        
    top_1 = 0
    top_5 = 0
    top_10 = 0
    total = 0

    for i in tqdm(range(len(graph))):
        row = graph[i]
        ind_row = sorted(list(range(len(graph))), key = lambda x: -row[x])
    
        if i in ind_row[:1]:
            top_1 += 1
    
        if i in ind_row[:5]:
            top_5 += 1
    
        if i in ind_row[:10]:
            top_10 += 1
    
        total += 1
    
    return top_1 / total, top_5 / total, top_10 / total

def get_classes(graph, classes, target_query):        
    top_1 = 0
    top_5 = 0
    top_10 = 0
    total = 0

    class_to_ind = {tuple(classes[i].cpu().numpy()):i for i in range(classes.shape[0])} 

    for i in tqdm(range(len(graph))):
        row = graph[i]
        class_embed = target_query[i]
        class_ind = class_to_ind[tuple(class_embed.cpu().numpy())]
        ind_row = sorted(list(range(len(graph[0]))), key = lambda x: -row[x])
    
        if class_ind in ind_row[:1]:
            top_1 += 1
    
        if class_ind in ind_row[:5]:
            top_5 += 1
    
        if class_ind in ind_row[:10]:
            top_10 += 1
    
        total += 1
    
    return top_1 / total, top_5 / total, top_10 / total

def qap_matching(source_base, target_base, source_query, target_query, graph, device):
    base_samples = source_base.shape[0]
    query_samples = source_query.shape[0]
    
    torch.manual_seed(0)
    shuffle = torch.randperm(query_samples)
    source_shuffled = source_query
    target_shuffled = target_query[shuffle]

    source_total = torch.cat([source_base, source_shuffled], dim=0)
    target_total = torch.cat([target_base, target_shuffled], dim=0)
    
    A = centered_scaled_corr_rbf(source_total, device).cpu().numpy()
    B = centered_scaled_corr_rbf(target_total, device).cpu().numpy()
    
    # first use faq multiple times and get the best result
    n_runs = 500
    options = {"P0": "randomized", "maximize": True, 
            "partial_match": np.array([np.arange(base_samples), np.arange(base_samples)]).T,
            "tol":1e-5}
    
    out_list = [quadratic_assignment(A, B, options=options)
            for i in tqdm(range(n_runs), total=n_runs)]
    len(out_list)
    
    
    out_list_sorted = sorted(out_list, key=lambda x: x.fun, reverse=True)
    out = out_list_sorted[0]
    
    new_col_ind = out.col_ind[base_samples:]-base_samples
    return sum(new_col_ind[shuffle] == np.arange(source_query.shape[0]))/len(new_col_ind)

def linear_matching(source_base, target_base, source_query, target_query, graph, device):
    
    torch.manual_seed(0)
    shuffle = torch.randperm(graph.shape[1])
    
    graph_shuffled = graph[:, shuffle]
    row_ind, col_ind = linear_sum_assignment(graph_shuffled, maximize=True)
    return sum(col_ind[shuffle] == row_ind)/len(col_ind)

def stretch_representations(repr):
    var = repr.var(axis=0)
    stretch = torch.diag(1/torch.sqrt(var))
    repr = torch.matmul(repr, stretch)
    return repr

def get_relative_representation(q, prior):
    # l2 normalize q and prior
    
    q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
    prior_norm = prior / np.linalg.norm(prior, axis=1, keepdims=True)
    
    # compute cosine similarity
    sims = np.dot(q_norm, prior_norm.T)
    
    print(sims.shape)
    
    return sims

def relative_baseline(source_base, target_base, source_query, target_query, device):
    source_base = source_base.cpu().numpy()
    target_base = target_base.cpu().numpy()
    source_query = source_query.cpu().numpy()
    target_query = target_query.cpu().numpy()

    source_repr = get_relative_representation(source_query, source_base)
    target_repr = get_relative_representation(target_query, target_base)

    # find the best match for each image_repr_shuffled using cosine similarity

    # normalize image_relative_repr and text_relative_repr  

    source_repr = source_repr / np.linalg.norm(source_repr, axis=1, keepdims=True)
    target_repr = target_repr / np.linalg.norm(target_repr, axis=1, keepdims=True)

    graph = cosine_similarity(source_repr, target_repr)
    return torch.tensor(graph)