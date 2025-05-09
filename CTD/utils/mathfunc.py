import numpy as np
import torch
from scipy import linalg
from mpmath import *

mp.dps = 32
one = mpf(1)
mp.pretty = True

def f(x):
    return sqrt(one-x)

a = taylor(f, 0, 10)
pade_p, pade_q = pade(a, 5, 5)
a = torch.from_numpy(np.array(a).astype(float))
pade_p = torch.from_numpy(np.array(pade_p).astype(float))
pade_q = torch.from_numpy(np.array(pade_q).astype(float))

def matrix_pade_approximant(p: torch.Tensor, I: torch.Tensor):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    #There are 4 options to compute the MPA: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(q_sqrt, p_sqrt)
    #return torch.linalg.solve(q_sqrt.cpu(), p_sqrt.cpu()).cuda()
    #return torch.linalg.inv(q_sqrt).mm(p_sqrt)
    #return torch.linalg.inv(q_sqrt.cpu()).cuda().bmm(p_sqrt)

def sqrtm_torch(M: torch.Tensor):
    normM = torch.norm(M,dim=[1,2]).reshape(M.size(0),1,1)
    I = torch.eye(M.size(1), requires_grad=False, device=M.device).reshape(1,M.size(1),M.size(1)).repeat(M.size(0),1,1)
    #This is for MTP calculation
    #M_sqrt = matrix_taylor_polynomial(M/normM,I)
    M_sqrt = matrix_pade_approximant(M / normM, I)
    M_sqrt = M_sqrt * torch.sqrt(normM)
    return M_sqrt

def torch_cov(x: torch.Tensor,
              rowvar: bool=False,
              bias: bool=False,
              ddof=None,
              aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def calculate_frechet_distance_cpu(mu1: np.ndarray,
                                   sigma1: np.ndarray,
                                   mu2: np.ndarray,
                                   sigma2: np.ndarray,
                                   eps: float=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    results = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return results

def calculate_frechet_distance_gpu(mu1: torch.Tensor,
                                   sigma1: torch.Tensor,
                                   mu2: torch.Tensor,
                                   sigma2: torch.Tensor,
                                   eps: float=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = torch.atleast_1d(mu1)
    mu2 = torch.atleast_1d(mu2)

    sigma1 = torch.atleast_2d(sigma1)
    sigma2 = torch.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = sqrtm_torch((sigma1 @ sigma2).unsqueeze(0)).squeeze()

    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = sqrtm_torch(((sigma1 + offset) @ (sigma2 + offset)).unsqueeze(0)).squeeze()

    tr_covmean = torch.trace(covmean)

    results = diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return results

def calculate_frechet_distance(mu1: torch.Tensor,
                               sigma1: torch.Tensor,
                               mu2: torch.Tensor,
                               sigma2: torch.Tensor):
    if mu1.device.type == 'cuda':
        dist = calculate_frechet_distance_gpu(mu1, sigma1, mu2, sigma2)
    else:
        dist = calculate_frechet_distance_cpu(mu1, sigma1, mu2, sigma2)
    return dist

def calculate_statistics_cpu(x: np.ndarray):
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma

def calculate_statistics_gpu(x: torch.Tensor):
    mu = torch.mean(x, dim=0)
    sigma = torch_cov(x)
    return mu, sigma

def calculate_statistics(x: torch.Tensor):
    if x.device.type == 'cuda':
        mu, sigma = calculate_statistics_gpu(x)
    else:
        mu, sigma = calculate_statistics_cpu(x.numpy())
        mu = torch.from_numpy(mu)
        sigma = torch.from_numpy(sigma)
    return mu, sigma

if __name__ == "__main__":
    import time
    x = torch.randn(1024, 1024, device='cuda')
    t1 = time.time()
    r1 = sqrtm_torch(x.unsqueeze(0)).squeeze().cpu().numpy()
    t2 = time.time()
    r2 = linalg.sqrtm(x.cpu().numpy())
    t3 = time.time()
    diff = np.abs(r2 - r1)
    print(t2-t1, t3-t2, diff.mean(), diff.std())
