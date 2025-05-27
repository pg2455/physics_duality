import numpy as np
import sys
sys.path.append("../src/")
import utils_ising


# the functions that follow are simply buildup to create a permutation matrix that acts on the space 
# of link configurations, mapping the original link to its dual. 

def canonical_orientation(i,j,N):
    # this function takes in i and j and returns as an *ordered* pair i,j so that the link goes *from* i to j, i.e.
    # so that i is lower than j mod 1. 

    if (i[0]==(j[0]+1)%N) or (i[1]==(j[1]+1)%N):
        return j,i
    return i,j


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
    i_primals,j_primals = utils_ising.get_edges(rows=N, cols=N)

    # okay, now we have them -- let's now construct the permutation matrix
    perm = np.zeros((len(i_primals),len(i_primals)))

    for idx_primal, (i,j) in enumerate(zip(i_primals,j_primals)):
        i_dual, j_dual = dual_edge_from_edge(i,j,N)
        idx_dual = find_link(i_dual, j_dual,N)
        perm[idx_dual,idx_primal] = 1
        
        
    return perm


# this finds a given link in i,j
def find_link(i_target, j_target,N):
    i,j = utils_ising.get_edges(rows=N, cols=N)

    for idx, (i_primal, j_primal) in enumerate(zip(i,j)):
        if all(np.array_equal(a, b) for a, b in zip(canonical_orientation(i_primal,j_primal,N),canonical_orientation(i_target,j_target,N))):
            return idx
        
    return None
