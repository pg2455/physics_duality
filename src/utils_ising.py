import numpy as np
from scipy.ndimage import convolve
import torch
import itertools
import torch.nn.functional as F
import math
from copy import deepcopy
from typing import Union

device = torch.device("cpu")

get_dual = lambda beta: 0.5 * math.asinh(math.sinh(2*beta) ** -1)

def get_plaquette(lattice, x, y):
    """Returns a product of spins formed by lattice sites forming a square with lower-left corner at x,y"""
    N = lattice.shape[0]
    # right * up * diag * (x,y)
    return lattice[(x+1)%N, y] * lattice[x, (y+1)%N] * lattice[(x+1)%N, (y+1)%N] * lattice[x, y]


# run mcmc
def run_mcmc_step(lattice, beta:Union[int, np.ndarray]=1.0, rng=None):
    """Single iteration of MCMC"""

    plaquette_beta = None
    # determine hamiltonian form
    if isinstance(beta, int):
        beta = beta 
    elif isinstance(beta, np.ndarray):
        if beta.shape[0] == 2:
            beta, plaquette_beta = beta[0], beta[1]
        elif beta.shape[0] == 1:
            beta = beta[0]
            plaquette_beta = 0
        else:
            raise ValueError(f"Unrecognized beta vector shape: {beta.shape}")
    else:
        raise ValueError(f"Unrecognized type of beta vector: {type(beta)}")

    N = lattice.shape[0]
    x = np.random.randint(0, N) if rng is None else rng.integers(0, N)
    y = np.random.randint(0, N) if rng is None else rng.integers(0, N)

    neighbours = lattice[(x+1)%N, y] + lattice[(x-1)%N, y] + lattice[x, (y+1)%N] + lattice[x, (y-1)%N]
    deltaE = 2 * lattice[x, y] * neighbours * beta
    
    deltaEPlaq = 0
    if plaquette_beta:
        # there are 4 squares associated with a lattice site
        neighbourPlaquettes = get_plaquette(lattice, x, y) 
        neighbourPlaquettes += get_plaquette(lattice, (x-1)%N, y) 
        neighbourPlaquettes += get_plaquette(lattice, (x-1)%N, (y-1)%N) 
        neighbourPlaquettes += get_plaquette(lattice, x, (y-1)%N) 
        deltaEPlaq = 2 * neighbourPlaquettes * plaquette_beta

    Ups = np.exp(- deltaE - deltaEPlaq)

    # flip the spin according to metropolis:
    accept = (np.random.rand()<Ups) if rng is None else (rng.random()<Ups)
    if accept:
        lattice[x,y] *= -1

    return lattice


def run_mcmc(N=4, beta:Union[int, np.ndarray]=1.0, n_samples=10, warmup_steps=-1,sample_rate=1,lattice=None, rng=None):
    """MCMC"""
    # note that it lets you specify a lattice for the start of the run
    # if not specified just create it
    if lattice is None:
        lattice = np.ones([N, N])

    # Warm up steps
    if warmup_steps < 0:
        warmup_steps = 500 * N * N
    
    lattices = []
    # okay note that the number of steps is equal to the number of samples you want
    # multipled by the sample rate . 
    for i in range(warmup_steps + (n_samples*sample_rate)):
        lattice = run_mcmc_step(lattice, beta, rng=rng)

        if i >= warmup_steps and (i % sample_rate)==0:
            lattices.append(lattice.copy())  
            
    return np.array(lattices)


def compute_energy(lattice, beta=1.0):
    """Computes energy of lattice using convolutions."""
    
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    
    return - beta * np.mean(convolve(lattice, kernel,  mode='wrap') * lattice * 0.25)


def compute_mean_potential(lattices):
    """Computes average of link potentials across lattices."""    
    lattices = np.array(lattices)
    kernel = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])
    x = convolve(lattices, kernel[None, ], mode='wrap') * lattices * 0.25 
    return x.mean((1,2)).mean()


def lattice_adjacency_matrix(lattice=None, rows=None, cols=None):
    """Returns the adjacency matrix corresponding to lattice (or if rows and columns are specified)."""
    if lattice is not None:
        rows, cols = lattice.shape

    adjacency_matrix = np.zeros((rows * cols, rows * cols), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            
            # index in flattened lattice
            idx_flat = i * cols + j
            
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # neighboring cell for i, j in lattice
                ni, nj = i + di, j + dj
                
                ni = ni % rows
                nj = nj % cols
                n_idx_flat = ni * cols + nj
                
                adjacency_matrix[idx_flat, n_idx_flat] = 1
                adjacency_matrix[n_idx_flat, idx_flat] = 1
                                
    return adjacency_matrix


def unflatten_idx(idx, rows, cols):
    return idx // rows, idx % cols


def get_edges(lattice=None, rows=None, cols=None):
    """Returns to and from lattice indices corresponding to each link."""
    A = lattice_adjacency_matrix(lattice, rows, cols)
    
    if lattice is not None:
        rows, cols = lattice.shape

    u, v = A.nonzero()
    u, v = list(zip(*[(i,j) for i, j in zip(u,v) if i<j])) # deduplicate; A is symmetric
    i = [unflatten_idx(x, rows, cols) for x in u]
    j = [unflatten_idx(x, rows, cols) for x in v]
    
    return np.array(i), np.array(j)


def get_s_ij(lattices, i, j):
    """Returns s_i, s_j pairs for each lattice in lattices."""
    s_i = lattices[:, i[:, 0], i[:, 1], None]
    s_j = lattices[:, j[:, 0], j[:, 1], None]
    return np.concatenate([s_i, s_j], axis=-1)

# Nabil moved this sample_stats function here and changed it slightly to take in a starting lattice
def sample_stats(N, beta:Union[int, np.ndarray], num_samples, i, j, model=None,warmup_steps=-1,sample_rate=-1, lattice=None):
    """
    Samples lattices and return their stats.
    Note that if you provide the lattice to start, it probably makes sense to set warmup_steps to zero. 
    """

    # if rate is not specified, use the following defaults: 
    if warmup_steps < 0:
        warmup_steps = 50*N*N
    if sample_rate < 0:
        sample_rate = 10*N*N

    #print(f"Samping at (beta,N) = ({beta},{N}) with warmup_steps = {warmup_steps} and sample_rate {sample_rate}.")
    lattices_n = run_mcmc(N, beta, n_samples=num_samples,warmup_steps = warmup_steps,sample_rate=sample_rate,lattice=lattice)
    s_ij = get_s_ij(lattices_n, i, j)
    o = s_ij[:, :, 0] * s_ij[:, :, 1]
    o = torch.tensor(o, dtype=torch.float32).to(device)
    H_model = -o.sum(-1).detach().cpu().numpy()
    avg_H_model = H_model.mean()

    # this applies the NN; not doing this for now
    if model is not None:
        o = model(o[..., None], beta).squeeze()

    return o, H_model, avg_H_model


# the functions that follow are simply buildup to create a permutation matrix that acts on the space 
# of link configurations, mapping the original link to its dual. 

def canonical_orientation(i,j,N):
    # this function takes in i and j and returns as an *ordered* pair i,j so that the link goes *from* i to j, i.e.
    # so that i is lower than j mod 1. 

    if (i[0]==(j[0]+1)%N) or (i[1]==(j[1]+1)%N):
        return j,i
    return i,j

"""
assert(canonical_orientation([0,0],[0,1],N)==canonical_orientation([0,1],[0,0],N))
"""

def dual_edge_from_edge(i,j,N):
    # this takes in an edge in the form i,j 
    # note that both i and j are pairs of coordinates which differ by a translation of one lattice site in 
    # either the x or y direction. 

    # first need to canonically orient it
    i,j = canonical_orientation(i,j,N)

    # find the translation vector:
    vec = j-i
    
    # make it the dual vector
    dual_vec = -np.flip(vec,axis=0)
    i_dual = i
    j_dual = (i_dual + dual_vec)%N

    # always returns it in a canonical orientation
    # the components of j will necessarily be lower. 
    return j_dual,i_dual

def dual_edge_permutation(N):
    # returns the dual edge permutation matrix in the basis used by Prateek
    # output matrix perm[enumeration of links in dual frame, enumeration of links in primal frame]
    i_primals,j_primals = get_edges(rows=N, cols=N)

    # okay, now we have them -- let's now construct the permutation matrix
    perm = np.zeros((len(i_primals),len(i_primals)))

    for idx_primal, (i,j) in enumerate(zip(i_primals,j_primals)):
        i_dual, j_dual = dual_edge_from_edge(i,j,N)
        idx_dual = find_link(i_dual, j_dual,N)
        perm[idx_dual,idx_primal] = 1
        
        
    return perm

# this finds a given link in i,j
def find_link(i_target, j_target,N):
    i,j = get_edges(rows=N, cols=N)

    for idx, (i_primal, j_primal) in enumerate(zip(i,j)):
        if all(np.array_equal(a, b) for a, b in zip(canonical_orientation(i_primal,j_primal,N),canonical_orientation(i_target,j_target,N))):
            return idx
        
    return None

def translate_basis(N,axis,delta = 1):
    # this is a function which returns a matrix that translates the whole basis we use for links either once in the x or y direction 
    # (accordinge to axis)
    # shifts it by an amount delta

    shift_vec = np.zeros(2)
    shift_vec[axis]=delta

    i,j = get_edges(rows=N, cols=N)

    mat = np.zeros((len(i),len(i)))
    # now we shift each of the i and j
    for idx_orig, (i_orig, j_orig) in enumerate(zip(i,j)):
        i_new, j_new = (i_orig + shift_vec) % N, (j_orig+shift_vec) % N
        # so i_new, j_new are the two things shifted by 1. 
        # now create the matrix by finding the appropriate link
        
        idx_new = find_link(i_new, j_new, N)
        mat[idx_new, idx_orig] = 1
    
    return mat

def generate_masks(N):

    # okay, now I will seek to enumerate everything as in Prateek's basis
    # the idea here is that we will create *masks*
    # a given mask will contain all orbits of the symmetry group acting on a single feature
    # the symmetry group here meaning translations. 

    # so, the first moment is very simple -- we just sum over all of the G_{ij}s, and we do not really need this fancy structure

    # the other features will be a bit more complicated
    # note the first index is the basis element entry, the second two are labeled by the action of the symmetry group
    num_features = 13
    i,j = get_edges(rows=N, cols=N)

    masks = np.zeros((2*N*N,N,N,num_features))
    masks[find_link([0,0],[0,1],N),0,0,0] = 1 # horizontal link
    masks[find_link([0,0],[1,0],N),0,0,1] = 1 # vertical link

    # now the other ones; note we need those which are topologically equivalent
    # because this is what the duality mapping will ultimately learn!

    masks[find_link([0,0],[0,1],N),0,0,2] = 1
    masks[find_link([0,1],[1,1],N),0,0,2] = 1 # L-shape 1

    masks[find_link([0,0],[0,1],N),0,0,3] = 1
    masks[find_link([0,1],[(N-1),1],N),0,0,3] = 1 # L- shape 2


    masks[find_link([0,0],[0,1],N),0,0,4] = 1
    masks[find_link([0,0],[1,0],N),0,0,4] = 1 # L- shape 3

    masks[find_link([0,0],[(N-1),0],N),0,0,5] = 1
    masks[find_link([0,0],[0,1],N),0,0,5] = 1  # L- shape 4

    # and finally the last two (a square)
    masks[find_link([0,0],[0,1],N),0,0,6] = 1
    masks[find_link([1,0],[1,1],N),0,0,6] = 1 # parallel lines 


    masks[find_link([0,0],[1,0],N),0,0,7] = 1
    masks[find_link([0,1],[1,1],N),0,0,7] = 1 # other parallel lines

    # okay, now a U-shape
    masks[find_link([0,0],[1,0],N),0,0,8] = 1
    masks[find_link([1,0],[1,1],N),0,0,8] = 1
    masks[find_link([1,1],[0,1],N),0,0,8] = 1 # U shape missing the line from 0,1 to 0,0

    # now the other U shapes 
    masks[find_link([0,0],[1,0],N),0,0,9] = 1
    masks[find_link([1,0],[1,1],N),0,0,9] = 1
    masks[find_link([0,1],[0,0],N),0,0,9] = 1 # U shape missing the line from 1,1 to 0,1

    masks[find_link([0,0],[1,0],N),0,0,10] = 1
    masks[find_link([1,1],[0,1],N),0,0,10] = 1
    masks[find_link([0,1],[0,0],N),0,0,10] = 1 # U shape missing the line from 1,0 to 1,1

    masks[find_link([1,0],[1,1],N),0,0,11] = 1
    masks[find_link([1,1],[0,1],N),0,0,11] = 1
    masks[find_link([0,1],[0,0],N),0,0,11] = 1 # U shape missing the line from 0,0 to 1,0


    masks[find_link([0,0],[1,0],N),0,0,12] = 1
    masks[find_link([1,0],[1,1],N),0,0,12] = 1
    masks[find_link([1,1],[0,1],N),0,0,12] = 1
    masks[find_link([0,1],[0,0],N),0,0,12] = 1 # this is a square!

    for x in range(N):
        for y in range(N):
            print(x,y)
            mat = np.matmul(translate_basis(N,axis=0,delta=x),translate_basis(N,axis=1,delta=y))

            # translate all of the features 
            masks[:,x,y,:] = np.matmul(mat,masks[:,0,0,:])
    
    return torch.tensor(masks).permute((1,2,3,0)).float()


# this evaluates a few features (those constructed in the masks above)
# on the dataset so we can see how they chnage 

def feature_eval(masks,o):
    features1 = torch.prod(o[:,None,None,None,:] ** masks[None,:],-1)
    # this takes the spatial average
    avg_features1 = features1.mean((1,2))
    
    # now these are all (num_samples * num_features) -- finally, sum over the samples
    return avg_features1.mean(0)

def feature_samplewise(masks,o):
    # this is the same as above but does not sum over the samples (written for legacy reasons)
    features1 = torch.prod(o[:,None,None,None,:] ** masks[None,:],-1)
    # this takes the spatial average
    avg_features1 = features1.mean((1,2))
    
    # now these are all (num_samples * num_features) -- finally, sum over the samples
    return avg_features1


# functions added by Prateek 


def convolve_filter_product(matrix, filter, pad):
    """Returns a sum of product of terms filtered on the matrix
    Args:
        matrix (B, N, N): vertical or horizontal link matrix
        filter (K, L): filter for convolution

    Returns:
        (B, )
    """
    N = matrix.shape[-1]
    K, L = filter.shape
    if pad == 'column':
        padding = (0, 1, 0, 0) # first two indices are for the last index i.e., column
    elif pad == 'row':
        padding = (0, 0, 0, 1)
    padded = F.pad(matrix.view(-1, 1, N, N), pad=padding, mode='circular')
    unfolded = F.unfold(padded, kernel_size=(K, L))
    weighted_product = unfolded*filter.view(-1, K*L, 1)
    return torch.prod(weighted_product, dim=-2).sum(dim=-1)


def get_plaquette_term(edge_matrices):
    """Returns the sum of plaquette terms."""
    vertical_link_matrix = edge_matrices[:, 1::2] # these links go vertical. neighbors are horizontally opposite links
    horizontal_link_matrix = edge_matrices[:, ::2] # these links go horizontal. neighbors are vertically opposite links

    # these vertical links are horizontally paried
    horizontal_filter = torch.tensor([[1, 1]])
    horizontal_paired_link_product_sum = convolve_filter_product(vertical_link_matrix, horizontal_filter, pad='column')

    # these horizontal links are vertically paried
    vertical_filter = torch.tensor([[1], [1]])
    vertical_paired_link_product_sum = convolve_filter_product(horizontal_link_matrix, vertical_filter, pad='row')

    plaquette_term = 0.5 * (horizontal_paired_link_product_sum + vertical_paired_link_product_sum)
    return plaquette_term


def sample_stats_with_edge(N, beta, num_samples, i, j, edge_i, edge_j, model=None, warmup_steps=-1, sample_rate=-1, rng=None, device=None):
    """
    Samples lattices and return their stats.
    """
    if device is None:
        try:
            device = torch.device('cpu') if model is None else  next(model.parameters()).device
        except StopIteration: # when model has no parameters
            device = torch.device('cpu')

    if warmup_steps < 0:
        warmup_steps = 50 * N * N
    
    if sample_rate < 0:
        sample_rate = 10 * N * N

    lattices_n = run_mcmc(N, beta, n_samples=num_samples, warmup_steps=warmup_steps, sample_rate=sample_rate, rng=rng)
    s_ij = get_s_ij(lattices_n, i, j)
    o = s_ij[:, :, 0] * s_ij[:, :, 1]
    o = torch.tensor(o, dtype=torch.float32).to(device)
    H_model = -o.sum(-1).detach().cpu().numpy() # DONT COMPUTE THIS AFTER CALLING MODEL ON o

    if model is not None:
        o = model(o[..., None], beta).squeeze()

    edge_matrices = np.empty((lattices_n.shape[0], 2*N, N))
    edge_matrices[:, edge_i, edge_j] = s_ij[:, :, 0] * s_ij[:, :, 1]
    edge_matrices = torch.tensor(edge_matrices, dtype=torch.float32).to(device)

    # 
    if beta.shape[0] == 2:
        paquette_term = -get_plaquette_term(edge_matrices) # (B, )
        H_terms = np.vstack([H_model, paquette_term]).transpose(1, 0) # (B, 2)
        avg_H_terms = H_terms.mean(0) # (1, 2)
    elif beta.shape[0] == 1:
        H_terms = H_model[:, None]
        avg_H_terms = H_terms.mean(0)
    else:
        raise ValueError(f"Unrecognized shape of beta vector: {beta.shape}")

    return o, H_terms, avg_H_terms, edge_matrices


def sample_stats_edge_wrapper(N, beta, i, j, edge_i, edge_j, num_samples, sample_rate, rng=None):
    return sample_stats_with_edge(N, beta, num_samples, i, j, edge_i, edge_j, None, sample_rate=sample_rate, rng=rng, device=torch.device('cpu'))


def get_edge_idxs(N, i, j):
    """Returns 2NxN matrix of edges."""
    edge_matrix_i, edge_matrix_j= [], []
    for (i1, j1), (i2, j2) in zip(i, j):
        # both points are the edges
        if (i1 == 0 and i2 == N-1) or (i1 == N-1 and i2 == 0):
            idx_i = 2*N-1
        else:
            idx_i = i1 + i2
        
        # both points are at the extremes
        if (j1 == 0 and j2 == N-1) or (j1 == N-1 and j2 == 0):
            idx_j = N-1
        else:
            idx_j = (j1 + j2) // 2
            
        edge_matrix_i.append(idx_i)
        edge_matrix_j.append(idx_j)
    
    return edge_matrix_i, edge_matrix_j


def get_features(edge_matrices, filters):
    """
    Returns features formed by weighted product of edge_matrices where the weights are provided by filters.

    Args:
        edge_matrices (P, num_samples x 2N x N): Represents the arrangement of observables (links in lattices). 
        filters (num_filters x M x M): Square filters with values to weight the elements of matrices. 

    Returns:
        features (P, num_filters): Each feature is a result of convolving a filter on lattices and averaging across the lattice and the samples
    """
    num_filters, K, L = filters.shape

    P = 1
    if len(edge_matrices.shape) == 4:
        P, num_samples, N, M = edge_matrices.shape
        edge_matrices = edge_matrices.view(-1, N, M)  # Combine P and num_samples dimensions
    else:
        num_samples, N, M = edge_matrices.shape
  
    # pad the matrix to account for toroidal boundary condition
    matrices = torch.cat([edge_matrices, edge_matrices[:, :, :L-1]], dim=-1)
    matrices = torch.cat([matrices, matrices[:, :K-1, :]], dim=-2)

    _, N, M = matrices.shape

    # Unfold the matrix to num_samples x 1 x K*L x P, where P is the number of windows possible with the filter of size KxL
    unfolded = F.unfold(matrices.view(-1, 1, N, M), kernel_size=(K, L))

    # expand the dimension and repeat to apply different filters in each expanded dimension
    unfolded = unfolded.view(P*num_samples, 1, K*L, -1).repeat(1, num_filters, 1, 1) # num_samples x num_filters x K*L x P

    # multiply with the filters
    weighted_product = unfolded * filters.view(1, num_filters, K*L, 1) # num_samples x num_filters x K*L x P

    # replace zeros by 1 so that the product is  not affected by the padding
    weighted_product = torch.where(weighted_product == 0, 1, weighted_product)

    # take the mean across the lattice and num_samples
    features = torch.prod(weighted_product, dim=-2).mean(dim=-1).reshape(P, num_samples, num_filters).mean(dim=1) # P x num_filters
    return features


def get_all_features(edge_matrices, all_filters):
    all_feats = []
    for filters in all_filters:
        all_feats.append(get_features(edge_matrices, filters))
    return torch.cat(all_feats, dim=-1)


def get_all_filters():
    """
    Generates all square filters iteratively until 3x3.

    Note/TODO: One can generate all possible filters in 2NxN and filter out the repetitions. 
    """

    # create combinations
    all_2x2 = list(itertools.product([0, 1], repeat=4))
    filters_2x2 = [torch.tensor(matrix, dtype=torch.float).reshape(2, 2) for matrix in all_2x2 if sum(matrix) > 1]

    # expand 2x2 to 3x3 to avoid repetitions
    # left, right, top, bottom
    pad_combinations = [(0, 1, 1, 0), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 0, 1)]
    all_2x2_padded = [F.pad(matrix, padding) for matrix in filters_2x2 for padding in pad_combinations]

    unique_3x3 = []
    all_3x3 = list(itertools.product([0, 1], repeat=9))
    filters_3x3 = [torch.tensor(matrix, dtype=torch.float).reshape(3, 3) for matrix in all_3x3]
    excluded = 0
    for matrix in filters_3x3:
        if matrix.sum() <= 1:
            excluded += 1
            continue 

        if all(not torch.equal(matrix, other) for other in all_2x2_padded):
            unique_3x3.append(matrix)
        else:
            excluded += 1

    # 
    filters_1x1 = torch.tensor([1], dtype=torch.float).view(1, 1, -1)

    filters_2x2 = [matrix.reshape(1, 2, 2) for matrix in filters_2x2]
    filters_2x2 = torch.cat(filters_2x2)

    filters_3x3 = [matrix.reshape(1, 3, 3) for matrix in unique_3x3]
    filters_3x3 = torch.cat(filters_3x3)

    return [filters_1x1, filters_2x2, filters_3x3]


# Functions for plaqutte mapping 

def get_edge_idxs_with_map(N, i, j):
    """Returns 2NxN matrix of edges."""
    edge_matrix_i, edge_matrix_j= [], []
    all_indices = {}
    for (i1, j1), (i2, j2) in zip(i, j):
        # both points are the edges
        if (i1 == 0 and i2 == N-1) or (i1 == N-1 and i2 == 0):
            idx_i = 2*N-1
        else:
            idx_i = i1 + i2
        
        # both points are at the extremes
        if (j1 == 0 and j2 == N-1) or (j1 == N-1 and j2 == 0):
            idx_j = N-1
        else:
            idx_j = (j1 + j2) // 2
            
        edge_matrix_i.append(idx_i)
        edge_matrix_j.append(idx_j)
        all_indices[frozenset({(i1, j1), (i2, j2)})] = (idx_i, idx_j)

    
    return edge_matrix_i, edge_matrix_j, all_indices


def get_edge_matrix_indexers_for_mapping(N, i, j, ij_to_edge_matrix_idxs):
    """Returns indices to index edge_matrix so that the neighbor of every link is retrieved in an order-preserving manner.
    See plaquette-July6-map-generalization.ipynb to check it in detail.
    """
    raw_mapping, edge_matrix_mapping = {}, {}
    indexing_link_to_edge_matrix_mapping = []
    indexing_link_to_neihboring_sites_mapping = []
    vertical_link_indicator = []
    for (i1, j1), (i2, j2) in zip(i, j):
        
        # vertical link
        if j1 == j2:
            assert i1 < i2, "First site is at the top"
            # start clockwise from the top 
            #  2 1 0     
            #    | 
            #  3 4 5
            #
            top_neighbors = [(i1, (j1+1) % N), ((i1-1) % N, j1), (i1, (j1-1)%N)]
            bottom_neighbors = [(i2, (j2-1)%N), ((i2+1)%N, j2), (i2, (j2+1)%N)]
            
            top_bottom_reverse = False
            if (i1 == 0 and i2 == N-1):
                # at boundaries, top becomes bottom, we swap them at the end but here we only ensure the vertical links are correctly recorded
                top_bottom_reverse = True

                x, y = top_neighbors[1]
                top_neighbors[1] = (i1+1, y) # shift the vertical link up by 1 when the vertical link is considered at the boundary

                x, y = bottom_neighbors[1]
                bottom_neighbors[1] = (i2-1, y) # shift the vertical link down by 1 when the vertical link is considered at the boundary

                bottom_neighbors = bottom_neighbors[::-1] # reverse the order for bottom neighbors to keep it clockwise 
                top_neighbors = top_neighbors[::-1] # same for top neighbors 

            if not top_bottom_reverse:
                raw_mapping[((i1, j1), (i2, j2))] = [((i1, j1), x) for x in top_neighbors] + [((i2, j2), x) for x in bottom_neighbors]
                edge_matrix_mapping[((i1, j1), (i2, j2))] = [ij_to_edge_matrix_idxs[frozenset({(i1, j1), x})] for x in top_neighbors]
                edge_matrix_mapping[((i1, j1), (i2, j2))] += [ij_to_edge_matrix_idxs[frozenset({(i2, j2), x})] for x in bottom_neighbors]
            else:
                raw_mapping[((i1, j1), (i2, j2))] = [((i2, j2), x) for x in bottom_neighbors] + [((i1, j1), x) for x in top_neighbors]
                edge_matrix_mapping[((i1, j1), (i2, j2))] = [ij_to_edge_matrix_idxs[frozenset({(i2, j2), x})] for x in bottom_neighbors]
                edge_matrix_mapping[((i1, j1), (i2, j2))] += [ij_to_edge_matrix_idxs[frozenset({(i1, j1), x})] for x in top_neighbors]

        # horizontal link
        if i1 == i2:
            assert j1 < j2, "First site is assumed on the left"
            # start clockwise from the right 
            #   3    2        2   3
            #  4  --- 1 -->  1 --- 4
            #   5    0        0   5
            right_neighbors = [((i2+1)%N, j2), (i2, (j2+1)%N), ((i2-1)%N, j2)]
            left_neighbors = [((i1-1)%N, j1), (i1, (j1-1)%N), ((i1+1)%N, j1)]

            right_left_reverse = False
            if (j1 == 0 and j2 == N-1):
                # at boundaries, left becomes right, we swap at the end.
                right_left_reverse = True

                x, y = right_neighbors[1]
                right_neighbors[1] = (x, j2-1)

                x, y = left_neighbors[1]
                left_neighbors[1] = (x, j1+1)

                right_neighbors = right_neighbors[::-1]
                left_neighbors = left_neighbors[::-1]

            if not right_left_reverse:
                raw_mapping[((i1, j1), (i2, j2))] = [((i2, j2), x) for x in right_neighbors] + [((i1, j1), x) for x in left_neighbors]
                edge_matrix_mapping[((i1, j1), (i2, j2))] = [ij_to_edge_matrix_idxs[frozenset({(i2, j2), x})] for x in right_neighbors]       
                edge_matrix_mapping[((i1, j1), (i2, j2))] += [ij_to_edge_matrix_idxs[frozenset({(i1, j1), x})] for x in left_neighbors]
            else:
                raw_mapping[((i1, j1), (i2, j2))] = [((i1, j1), x) for x in left_neighbors] + [((i2, j2), x) for x in right_neighbors]
                edge_matrix_mapping[((i1, j1), (i2, j2))] = [ij_to_edge_matrix_idxs[frozenset({(i1, j1), x})] for x in left_neighbors]
                edge_matrix_mapping[((i1, j1), (i2, j2))] += [ij_to_edge_matrix_idxs[frozenset({(i2, j2), x})] for x in right_neighbors]
            
            # reflect vertically 
            edge_matrix_mapping[((i1, j1), (i2, j2))] = edge_matrix_mapping[((i1, j1), (i2, j2))][::-1]

        indexing_link_to_edge_matrix_mapping.append(
            edge_matrix_mapping[((i1, j1), (i2, j2))] + [ ij_to_edge_matrix_idxs[frozenset({(i1, j1), (i2, j2)})] ]
        )

        indexing_link_to_neihboring_sites_mapping.append(
            raw_mapping[((i1, j1), (i2, j2))]  + [ ((i1, j1), (i2, j2)) ]
        )
        
        vertical_link_indicator.append(int(j1 == j2))
    
    return indexing_link_to_edge_matrix_mapping, torch.tensor(vertical_link_indicator)


def compute_convolved_features(batch_edge_matrices, masks, return_batch_mean=False):
    batch_size = batch_edge_matrices.shape[0] 
    N = batch_edge_matrices.shape[-1]
    X, Y = masks.shape[-2:]

    reshaped_masks = masks.reshape(-1, X*Y, 1) # 1 x NUM_MASKS x NUM_SITES_IN_KERNEL x 1
    
    # Prepare edge matrices
    batch_padded = F.pad(
            batch_edge_matrices.view(-1, 1, 2*N, N), 
            pad=(0, Y-1, 0, X-2), 
            mode='circular'
    ) # B * NUM_SAMPLES x 2*N x N
    
    # Extract and filter features
    batch_unfolded = F.unfold(
        batch_padded,
        kernel_size=(X, Y),
        padding=0,
        stride=(2, 1)
    )  # B * NUM_SAMPLES x NUM_SITES_IN_KERNEL x NUM_WINDOWS

    batch_filtered = (batch_unfolded[:, None] ** reshaped_masks[None]).prod(dim=-2)
    batch_feature = batch_filtered.mean(dim=-1)

    if return_batch_mean:
        return  batch_feature.reshape(batch_size, masks.shape[0]).mean(0)

    return batch_feature.reshape(batch_size, masks.shape[0])


class HorizontalLink:
    @staticmethod
    def get_left_relevant(links):
        return links[:3]

    @staticmethod
    def get_right_relevant(links):
        return links[3:-1]

    @staticmethod
    def get_top_relevant(links):
        return links[1:3] + links[3:5] + links[-1:]

    @staticmethod
    def get_bottom_relevant(links):
        return links[:2] + links[4:]

class VerticalLink:
    @staticmethod
    def get_left_relevant(links):
        return links[1:5] + links[-1:] # 5 links

    @staticmethod
    def get_right_relevant(links):
        return links[:2] + links[4:] # 5 links

    @staticmethod
    def get_top_relevant(links):
        return links[:3] # 3 links

    @staticmethod
    def get_bottom_relevant(links):
        return links[3:6]

def get_features_with_second_order_links(N, i, j, edge_i, edge_j, ij_to_edge_matrix_idxs, indexing_link_to_edge_matrix_mapping):
    reverse_ij_to_edge_matrix_idxs = {v: k for k, v in ij_to_edge_matrix_idxs.items()}
    zipped_edge_idxs = list(zip(edge_i, edge_j))
    all_features = []
    for idx, _ in enumerate(zip(i, j)):
        # print("Current edge: ", edge_i[idx], edge_j[idx])
        edge_idx = (edge_i[idx], edge_j[idx])
        current_link = reverse_ij_to_edge_matrix_idxs[edge_idx]

        left_link = (edge_idx[0], (edge_idx[1] - 1) % N )
        right_link = (edge_idx[0], (edge_idx[1] + 1) % N )
        top_link = ((edge_idx[0] - 2) % (2*N), edge_idx[1])
        bottom_link = ((edge_idx[0] + 2) % (2*N), edge_idx[1])

        # print("Edge idxs:", edge_idx, left_link, top_link, right_link, bottom_link)
        current_features = deepcopy(indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(edge_idx)])

        if edge_idx[0] % 2 == 0: # it's a horizontal link
            # left
            featured_left_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(left_link)]
            current_features += HorizontalLink.get_left_relevant(featured_left_links)

            # top
            featured_top_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(top_link)]
            current_features += HorizontalLink.get_top_relevant(featured_top_links)
            
            # right
            featured_right_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(right_link)]
            current_features += HorizontalLink.get_right_relevant(featured_right_links)

            # bottom
            featured_bottom_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(bottom_link)]
            current_features += HorizontalLink.get_bottom_relevant(featured_bottom_links)
        else:
            # top
            featured_top_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(top_link)]
            current_features += VerticalLink.get_top_relevant(featured_top_links)

            # left
            featured_left_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(left_link)]
            current_features += VerticalLink.get_left_relevant(featured_left_links)

            # bottom
            featured_bottom_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(bottom_link)]
            current_features += VerticalLink.get_bottom_relevant(featured_bottom_links)

            # right
            featured_right_links = indexing_link_to_edge_matrix_mapping[zipped_edge_idxs.index(right_link)]
            current_features += VerticalLink.get_right_relevant(featured_right_links)

        # print("Features: ", current_features)

        assert len(set(current_features)) == len(current_features), "Duplicate features"
        all_features.append(current_features)
    
    return all_features