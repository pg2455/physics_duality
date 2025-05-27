import numpy as np
from tqdm import tqdm
from scipy.ndimage import shift
import torch
from utils_ising import translate_basis, canonical_orientation, get_edges


def generate_translational_mats(N):
    # this generates the translational matrices
    # this is the most time consuming part of constructing the various masks
    # so do it just once! 
    
    mat = np.zeros((N,N,2*(N*N),2*(N*N)))
    for x in range(N):
        for y in range(N):
            mat[x,y,:,:] = np.matmul(translate_basis(N,axis=0,delta=x),translate_basis(N,axis=1,delta=y))
    
    return mat


def related_by_translation(links1, links2):
    """
    Input: two numpy arrays, each of which are (x, y, link #)
    We just check if one can be obtained from the other by translation in either x or y
    """

    # this creates a shift vector with enough entries no matter how many extra channels we have
    shiftvec = np.zeros(len(links1.shape))
    size = links1.shape[0]
    for x in range(-size,size):
        for y in range(-size,size):
            #if np.array_equal(np.roll(links1,shift=(x,y),axis=(0,1)),links2):
            shiftvec[:2] = (x,y)
            if np.array_equal(shift(links1,shift=shiftvec,cval=0,mode='constant').astype(np.uint8),links2.astype(np.uint8)):
                return True
    return False


def is_gauge_equivalent(links1_in,links2_in):
    """ 
    Input: two numpy arrays, each of which are (x, y, link #)
    We now check if one can be obtained from another by adding a set of closed loops
    We do this by finding the ``gauge-invariant'' data, i.e. the number of links leaving each point mod 2. 
    """
    
    if np.array_equal(divergence_links(links1_in),divergence_links(links2_in)):
        return True
    return False


def divergence_links(links1_in):
    # this computes the ``divergence'', i.e. a binary variable which counts the number of links entering and
    # leaving the site (mod 2)

    # embed it into a slightly larger grid to deal w/ boundary effects
    links1 = np.zeros((3,3,2)).astype(np.uint8)
    links1[:2,:2,:] = links1_in

    # first, the links leaving the site
    div1 = (links1[:,:,0]+links1[:,:,1]) % 2
    
    # second, those coming ``in''
    div1 = (div1 + shift(links1[:,:,0],shift=(1,0),cval=0,mode='constant').astype(np.uint8)) % 2
    div1 = (div1 + shift(links1[:,:,1],shift=(0,1),cval=0,mode='constant').astype(np.uint8)) % 2

    return div1


def binary_representation_array(number, size):
    binary_array = []
    
    for i in range(size):
        # Extract the i-th bit from the right
        bit = (number >> i) & 1
        # Prepend the bit to the list (reversing the order)
        binary_array.insert(0, bit)
    
    return binary_array


# this does the same thing but faster because it doesnt need to figure out the base i and j, which should be passed to it. 
def find_link_fast(i_target, j_target,i, j, N):

    for idx, (i_primal, j_primal) in enumerate(zip(i,j)):
        if all(np.array_equal(a, b) for a, b in zip(canonical_orientation(i_primal,j_primal,N),canonical_orientation(i_target,j_target,N))):
            return idx
        
    return None


def generate_masks_all(N,num_sites=2,max_features=None,mats=None):
    # this algorithmically generates all the masks within a certain window size
    
    links = np.zeros((num_sites,num_sites,2))

    # this stores the links as a grid. 
    # links[x,y,0] refers to a link leaving the site (x,y) and going to (x+1,y).
    # links[x,y,1] refers to a link leaving the site (x,y) and going to (x,y+1).
    mask_rep = []
    # now we set all of these to 1 or 0 systematically. 
    # this generates two independent binary numbers for the links in the x direction and in the y direction
    for rep_1 in range(2**(num_sites**2)):
        binary_array_1 = np.array(binary_representation_array(rep_1,num_sites**2))
        links[:,:,0] = binary_array_1.reshape(num_sites,num_sites)
        for rep_2 in range(2**(num_sites**2)):
            binary_array_2 = np.array(binary_representation_array(rep_2,num_sites**2))
            links[:,:,1] = binary_array_2.reshape(num_sites,num_sites)
            mask_rep.append(links.copy())
    
    # sort this list by number of links
    mask_rep = sorted(mask_rep,key=np.sum)

    # so now mask_rep contains a list of arrays, each of which have a representation of the 2^(num_sites^2) possibilities
    # of nonzero links. 

    # now we will include only those which are not translations (up to gauge-equivalence) of any other elements
    unique_masks = []
    # the very first one here is the empty set of links, which we need not consider
    print("Removing gauge-equivalent masks")
    for mask_considering in tqdm(mask_rep[1:]):
        keep = True
        for mask_comparing in unique_masks:
            #if related_by_translation(mask_comparing, mask_considering) or is_gauge_equivalent(mask_comparing, mask_considering):
            if related_by_translation(divergence_links(mask_comparing), divergence_links(mask_considering)):
                keep = False
        if keep:
            unique_masks.append(mask_considering.copy())
 
    # now, truncate to a given number of masks
    if max_features is not None:
        unique_masks = unique_masks[:(max_features-1)]

    sq = np.zeros((num_sites,num_sites,2))
    sq[0,0,1] = 1
    sq[0,0,0] = 1
    sq[0,1,0] = 1
    sq[1,0,1] = 1
    unique_masks.append(sq)
    
    # okay, now let's convert these into link variables
    # masks_output will store them in Prateek's basis which we normally use for link variables. 
    i,j = get_edges(rows=N, cols=N)


    masks_output = np.zeros((2*N*N,N,N,len(unique_masks)))
    for idx, unique_mask in enumerate(unique_masks):
        for x in range(num_sites):
            for y in range(num_sites):
                if unique_mask[x,y,0] == 1:
                    # link pointing in the x direction
                    masks_output[find_link_fast([x,y],[x+1,y],i,j,N),0,0,idx] = 1
                if unique_mask[x,y,1] == 1:
                    # link pointing in the y direction
                    masks_output[find_link_fast([x,y],[x,y+1],i,j,N),0,0,idx] = 1

    # and now we do the usual thing to translate the basis
    print("Creating translational images")
    for x in tqdm(range(N)):
        for y in range(N):
            #print(x,y)
            #mat = np.matmul(translate_basis(N,axis=0,delta=x),translate_basis(N,axis=1,delta=y))

            # translate all of the features 
            masks_output[:,x,y,:] = np.matmul(mats[x,y],masks_output[:,0,0,:])
    
    return torch.tensor(masks_output).permute((1,2,3,0)).float()
