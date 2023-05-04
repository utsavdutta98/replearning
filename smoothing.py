import collections
import numpy as np
from scipy.interpolate import UnivariateSpline

def good_turing_estimator(samples):

    N = samples.sum()
    counts = dict(collections.Counter(samples.flatten()))

    x = [x[0] for x in sorted(counts.items())]
    y = [x[1] for x in sorted(counts.items())]

    spline = UnivariateSpline(x, y)

    def f(x):

        if x in counts:
            c_x = counts[x]
        else:
            c_x = spline(x)

        if x+1 in counts:
            c_x_1 = counts[x+1]
        else:
            c_x_1 = spline(x+1)

        return ((x+1)/N) * (c_x_1/c_x)

    gt = np.vectorize(f)(samples)

    return gt


def MLE_estimator(samples):
    
    return samples/samples.sum()

def svd_smoothing(samples,threshold=0.95,idx=None,normalize=True):

    u,s,vh = np.linalg.svd(samples)

    if idx is None:
        var = np.cumsum(s**2/(s**2).sum())
        idx = np.where(var>threshold)[0][0]

    if idx == 0:
        idx = 1

    reconstructed_samples = u[:,:idx] @ np.diag(s[:idx]) @ vh[:idx,:]

    if normalize:
        return reconstructed_samples/reconstructed_samples.sum()

    return reconstructed_samples

def interpolation_smoothing(samples,l):

    N,N = samples.shape

    MLE = samples/samples.sum()

    interpolated = l * MLE + (1-l) * (1/N)

    return interpolated


def add_k_smoothing(samples,k):

    return (samples + k)/(samples.sum() + k*samples.shape[0]**2)

def kneser_ney_smoothing(samples,d=0.75):

    N = samples.sum()

    counts_context = np.sum(samples,axis=0,keepdims=True)
    counts_word = np.sum(samples,axis=1,keepdims=True)

    l = (d * np.sum(np.sign(samples),axis=0,keepdims=True))/counts_context

    p_word_given_context = (np.maximum(samples - d,0) / counts_context) + l * (counts_word / samples.sum())
    p_context = counts_context/N

    return p_word_given_context * p_context

def glove_importance_sampling(samples,alpha=0.75):

    max_val = np.max(samples)
    X = (samples/max_val) ** alpha

    return X/X.sum()

def eigenvalue_weighting(samples,p,idx=None,normalize=True):

    u,s,vh = np.linalg.svd(samples)

    if idx is None:
        var = np.cumsum(s**2/(s**2).sum())
        idx = np.where(var>0.95)[0][0]

    if idx == 0:
        idx = 1

    reconstructed_samples = u[:,:idx] @ ((np.diag(s[:idx])) ** p) @ vh[:idx,:]

    if normalize:
        return reconstructed_samples/reconstructed_samples.sum()

    return reconstructed_samples

def cds_smoothing(samples,alpha=0.75):

    p_hat = samples/samples.sum()
    p_hat_w = samples.sum(axis=1,keepdims=True)/samples.sum()
    a = samples.sum(axis=0,keepdims=True)
    alpha = 0.75
    p_hat_c = (a ** alpha) / ((a**alpha).sum())

    cds = np.log(p_hat/(p_hat_w * p_hat_c))
    
    return cds

def dirichlet_smoothing(samples,l=1e-3):

    N = samples.sum()

    counts_context = np.sum(samples,axis=0,keepdims=True)
    counts_word = np.sum(samples,axis=1,keepdims=True)

    p_lambda_w_c = (samples + l) / (N + l * samples.shape[0] * samples.shape[1])
    p_lambda_w  = (counts_word + l * N) / (N + l * samples.shape[0] * samples.shape[1])
    p_lambda_c  = (counts_context + l * N) / (N + l * samples.shape[0] * samples.shape[1])

    return np.log( p_lambda_w_c / (p_lambda_w * p_lambda_c ) )
