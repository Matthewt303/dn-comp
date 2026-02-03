import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import matplotlib as mpl
import os

def load_snr_ssim_file(file: str) -> "np.ndarray":

    snr_ssim_data = pd.read_csv(
        file, sep=",", header=None, engine="pyarrow", skiprows=1
    )

    return np.array(snr_ssim_data).astype(np.float32)

def poly_2nd(x: "np.ndarray", a: float, b: float, c: float):

    return a * x**2 + b * x + c

def poly_3rd(x: "np.ndarray", a: float, b: float, c: float, d: float):

    return a*x**3 + b*x**2 + c*x + d

def fit(func, x: "np.ndarray", y: "np.ndarray"):

    popt, pcov = curve_fit(func, x, y)

    return popt, pcov

def plot_data(data: "np.ndarray", fit: "np.ndarray",
              frames: "np.ndarray") -> None:
    
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    cmap = mpl.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=500)
    #colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

    ax.plot(frames, data, 'o', ls='-', linewidth=2.5,
                 alpha=0.05, color='darkred',  label="data")
    ax.plot(frames, fit, ls='-', linewidth=2.5,
                 alpha=0.75, color='darkblue',  label="fit")
    
    leg = plt.legend(bbox_to_anchor=(0.5, 1.175), loc='upper center', ncol=2)

    for line in leg.get_lines():
        line.set_linewidth(3.5)

    for text in leg.get_texts():
        text.set_fontsize(28)
    
    ax.set_xlim(left=0, right=np.max(frames) + 100)

    ax.tick_params(axis='y', which='major', length=10, direction='in')
    ax.tick_params(axis='y', which='minor', length=5, direction='in')
    ax.tick_params(axis='x', which='major', length=10, direction='in')
    ax.tick_params(axis='x', which='minor', length=5, direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('Frame', labelpad=6, fontsize=32)
    ax.set_ylabel("SNR (dB)", labelpad=2, fontsize=32)

    output_folder = "C:/users/mxq76232/Downloads/poly_test"

    plt.savefig(os.path.join(output_folder, "d2_plot_poly3.svg"))
    plt.savefig(os.path.join(output_folder, "d2_plot_poly3.png"))

def main():

    file = "C:/users/mxq76232/Downloads/poly_test/1_tubulin_self_train.csv"
    file2 = "C:/users/mxq76232/Downloads/poly_test/d2_tub_self.csv"
    data = load_snr_ssim_file(file)
    data2 = load_snr_ssim_file(file2)

    snr = data[:20000, 0]
    snr2 = data2[:20000, 0]
    frames = 1 + np.arange(20000)

    popt, pcov = fit(poly_3rd, frames, snr)
    popt2, pcov2 = fit(poly_3rd, frames, snr2) 

    pred_snr = poly_3rd(frames, popt[0], popt[1], popt[2], popt[3])

    mse = np.mean((snr2 - pred_snr)**2)
    print(popt)
    print(popt2)

    plot_data(snr2, pred_snr, frames)

if __name__ == "__main__":
    main()