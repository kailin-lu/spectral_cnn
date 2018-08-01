import numpy as np
import itertools

def _crop_spectrum(y, filter_height, filter_width): 
    """
    Crops input image y to the center filter size  
    
    Args: 
        y: Numpy array of dimension [batch, channel, height, width] 
        output_size: output image dimension of [height, width] 
    """
    image_height = y.shape[2]
    image_width = y.shape[3]
    
    
    height_i = int(np.ceil((image_height - filter_height) / 2))
    height_j = image_height - height_i 
    
    width_i = int(np.ceil((image_width - filter_width) / 2))
    width_j = image_width - width_i
    
    return y[:,:,height_i:height_j,width_i:width_j] 

    
def _treat_corner_cases(y): 
    """
    Returns map obeying conjugate symmetry, treated indices S  
    
    Args 
        y: Numpy array input of dimensions [b, c, h, w] 
    """
    M = y.shape[2]
    N = y.shape[3] 
    index = list() 
    
    if M % 2 == 0: 
        M = M // 2
    if N % 2 == 0: 
        N = N // 2 
        
    MN_indices = itertools.product(range(M), range(N)) 
    # For all elements within the special indices, set the imaginary part to 0  
    for mn in MN_indices:
        y[:,:,mn[0],mn[1]] = np.real(y[:,:,mn[0],mn[1]])  
        index.append(mn) 
    return y, index


def _remove_redundancy(y): 
    """
    Args
        :y Numpy gradient map of dimension [b, c, h, w]
    """
    z, S = _treat_corner_cases(y) 
    I = [] 
    M = y.shape[2] 
    N = y.shape[3] 
    
    MN_indices = itertools.product(range(M-1), range(int(N/2))) 
    
    for mn in MN_indices:
        if mn not in S: 
            if mn not in I: 
                z[:,:,mn[0],mn[1]] = 2 * z[:,:,mn[0],mn[1]]
                I.append(mn) 
            else: 
                z[:,:,mn[0],mn[1]] = 0 
    return z.astype(float)


def _pad_spectrum(z, orig_shape): 
    """
    Returns z padded with 0s with dim [b, c, final_height, final_width] 
    """
    current_height = z.shape[2] 
    current_width = z.shape[3] 
    
    final_height = orig_shape
    final_width = orig_shape
    
    pad_top = int(np.ceil((final_height - current_height) / 2)) 
    pad_bottom = int(np.floor((final_height - current_height) / 2))
    pad_left = int(np.ceil((final_width - current_width) / 2))
    pad_right = int(np.floor((final_width - current_width) / 2)) 
    return np.pad(z, 
                  pad_width=((0,0),(0,0),(pad_top,pad_bottom),(pad_left,pad_right)), 
                  mode='constant', constant_values=0) 
 
    
def _recover_map(z): 
    """
    Returns full gradient map with recovered redundancy 
    """
    z, index = _treat_corner_cases(z)
    M = z.shape[2] 
    N = z.shape[3]
    MN_indices = list(itertools.product(range(M-1),range(int(np.floor(N/2)))))
    I = [] 
    
    for mn in MN_indices:
        if mn not in index: 
            if mn not in I: 
                z[:,:,mn[0],mn[1]] = .5 * z[:,:,mn[0],mn[1]] 
                a = (M-mn[0]) % M
                b = (N-mn[1]) % N 
                z[:,:,a,b] = z[:,:,mn[0],mn[1]] 
                I.append((mn[0],mn[1])) 
                I.append((a,b))
            else: 
                z[:,:,mn[0],mn[1]] = 0 
    return z.astype(float) 
                    
               
                    

    
    