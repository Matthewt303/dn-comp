import numpy as np
import tifffile as tiff
from typing import Generator

##--------FILE IO--------##

def load_image_stack(path: str) -> "np.ndarray":

    with tiff.TiffFile(path) as tif:
        memmap_im_stack = tif.asarray(out="memmap")
    
    return memmap_im_stack.astype(np.float32)

def load_stack_slices(n_path: str, dn_path: str) -> Generator[
    "np.ndarray", None, None]:

    """
    Use if memory is concern.
    """

    n_stack_frames = load_image_stack(n_path)
    dn_stack_frames = load_image_stack(dn_path)

    yield zip(n_stack_frames, dn_stack_frames)

##--------CALCULATIONS--------##

def calc_snr(n_image: "np.ndarray", dn_image: "np.ndarray") -> float:

    num = np.sum(np.square(n_image))
    denom = np.sum(np.square(dn_image - n_image))

    snr = 10 * np.log10(num / denom)

    return snr

def calc_cov(n_image: "np.ndarray", dn_image: "np.ndarray",
             noisy_mean: float, dn_mean: float) -> float:
    
    cov = np.sum(((n_image - noisy_mean) * (dn_image - dn_mean)) / (n_image.shape[0] - 1))

    return cov

def calc_ssim(n_image: "np.ndarray", dn_image: "np.ndarray") -> float:

    c1 = (0.01 * 65535) ** 2
    c2 = (0.03 * 65535) ** 2

    n_mean, dn_mean = np.mean(n_image), np.mean(dn_image)
    n_var, dn_var = np.var(n_image), np.var(dn_image)
    covar = calc_cov(n_image, dn_image, n_mean, dn_mean)

    ssim_num = (2 * n_mean * dn_mean + c1) * (2 * covar + c2)
    ssim_denom = (n_mean ** 2 + dn_mean ** 2 + c1) * (n_var ** 2 + dn_var ** 2 + c2)

    return ssim_num / ssim_denom

def calc_metrics(n_stack: "np.ndarray", dn_stack: "np.ndarray") -> "np.ndarray":

    psnr_and_ssim = np.zeros((n_stack.shape[0], 2), dtype=np.float32)

    return psnr_and_ssim

##--------PROGRAM START--------##

def main():

    noisy_im_path = ""
    denoised_im_path = ""