import numpy as np
import collections

def generate_p_star(N,a=2,normalized=False,symmetric=True):

    if symmetric:

        num_elems_upper_tri = 0.5 * N * (N+1)
        x = np.random.zipf(a=a, size=int(num_elems_upper_tri))

        tri = np.zeros((N, N))
        tri[np.triu_indices(N, 0)] = x

        X = tri + tri.T - np.diag(np.diag(tri))

    else:

        X = np.random.zipf(a=a, size=int(N*N)).reshape(N,N)

    p_zipf_unnorm = lambda k: k**(-a)
    p_star = np.zeros([N, N])
    for i in range(N):
      for j in range(N):
        p_star[i,j] = p_zipf_unnorm(X[i,j])

    noise = np.abs(np.random.normal(loc=0,scale=1,size=int(N*N)).reshape(int(N),int(N))) * 0.0001
    p_orig = p_star
    p_star = p_star + noise

    if normalized == True: 
        return p_star / np.sum(p_star), p_orig, noise, X

    return X, noise

def sample_from_2d(p,N,return_p_hat=True):

    flattened_probs = p.flatten()

    if np.sum(flattened_probs != 1):
        flattened_probs = flattened_probs / np.sum(flattened_probs)

    # Sample an index from the flattened probability array
    index = np.random.choice(len(flattened_probs), p=flattened_probs, size = N)

    # Convert the index back into a 2D index
    row_index = index // p.shape[1]
    col_index = index % p.shape[1]    

    counts = collections.Counter(list(zip(row_index,col_index)))

    samples = np.zeros_like(p)

    for key,value in counts.items():
        samples[key[0],key[1]] = value

    if return_p_hat:
        samples = samples/samples.sum()

    return samples

def get_pmi_matrix(p,ppmi=True):

    D = p.sum()

    counts_context = np.sum(p,axis=0,keepdims=True)

    counts_word = np.sum(p,axis=1,keepdims=True)

    mat = np.log( (p * D) / (counts_context * counts_word))

    if ppmi:
        return np.maximum(mat,0)

    return mat

def convert_to_ppmi(pmi):

    return np.maximum(pmi,0)
