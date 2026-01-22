import numpy as np
import tifffile as tiff

def calc_snr(n_image: "np.ndarray", dn_image: "np.ndarray"):

    num = np.sum(np.square(n_image))
    denom = np.sum(np.square(dn_image - n_image))

    snr = 10 * np.log10(num / denom)

    return snr

def calc_cov(n_image: "np.ndarray", dn_image: "np.ndarray",
             noisy_mean: float, dn_mean: float):
    
    cov = np.sum(((n_image - noisy_mean) * (dn_image - dn_mean)) / (n_image.shape[0] - 1))

    return cov

def calc_ssim(n_image: "np.ndarray", dn_image: "np.ndarray"):

    c1 = (0.01 * 65535) ** 2
    c2 = (0.03 * 65535) ** 2

    n_mean, dn_mean = np.mean(n_image), np.mean(dn_image)
    n_var, dn_var = np.var(n_image), np.var(dn_image)
    covar = calc_cov(n_image, dn_image, n_mean, dn_mean)

    ssim_num = (2 * n_mean * dn_mean + c1) * (2 * covar + c2)
    ssim_denom = (n_mean ** 2 + dn_mean ** 2 + c1) * (n_var ** 2 + dn_var ** 2 + c2)

    return ssim_num / ssim_denom