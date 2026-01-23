import numpy as np
from numba import jit, prange
import tifffile as tiff
from typing import Generator
import time

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

@jit(nopython=True, nogil=True, cache=False)
def calc_snr(n_image: "np.ndarray", dn_image: "np.ndarray") -> float:

    num = np.sum(np.square(n_image))
    denom = np.sum(np.square(dn_image - n_image))

    snr = 10 * np.log10(num / denom)

    return snr

@jit(nopython=True, nogil=True, cache=False)
def calc_cov(n_image: "np.ndarray", dn_image: "np.ndarray",
             noisy_mean: float, dn_mean: float) -> float:
    
    cov = np.sum(((n_image - noisy_mean) * (dn_image - dn_mean)) / (n_image.shape[0] - 1))

    return cov

@jit(nopython=True, nogil=True, cache=False)
def calc_ssim(n_image: "np.ndarray", dn_image: "np.ndarray") -> float:

    c1 = (0.01 * 65535) ** 2
    c2 = (0.03 * 65535) ** 2

    n_mean, dn_mean = np.mean(n_image), np.mean(dn_image)
    n_var, dn_var = np.var(n_image), np.var(dn_image)
    covar = calc_cov(n_image, dn_image, n_mean, dn_mean)

    ssim_num = (2 * n_mean * dn_mean + c1) * (2 * covar + c2)
    ssim_denom = (n_mean ** 2 + dn_mean ** 2 + c1) * (n_var ** 2 + dn_var ** 2 + c2)

    return ssim_num / ssim_denom

@jit(nopython=True, nogil=True, cache=False, parallel=True)
def calc_metrics(n_stack: "np.ndarray", dn_stack: "np.ndarray") -> "np.ndarray":

    psnr_and_ssim = np.zeros((n_stack.shape[0], 2), dtype=np.float32)

    for i in prange(n_stack.shape[0]):

        noisy_im = n_stack[i, :, :]
        denoised_im = dn_stack[i, :, :]
    
        psnr_and_ssim[i, 0] = calc_snr(noisy_im, denoised_im)
        psnr_and_ssim[i, 1] = calc_ssim(noisy_im, denoised_im)

    return psnr_and_ssim

##--------PROGRAM START--------##

def main():

    noisy_im_path = ""
    denoised_im_path = ""

    n_stack = load_image_stack(noisy_im_path)
    dn_stack = load_image_stack(denoised_im_path)

    start = time.time()
    metrics = calc_metrics(n_stack, dn_stack)
    end = time.time()

    print(str(end - start))
    print(metrics[0:6, :])

if __name__ == "__main__":
    main()