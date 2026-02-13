import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import os

def collate_files(folder: str) -> list[str]:

    files = os.listdir(folder)

    snr_ssim_files = [
        os.path.join(folder, file) for file in files
        if file.endswith(".csv")
    ]

    return sorted(snr_ssim_files)

def load_snr_ssim_file(file: str) -> "np.ndarray":

    snr_ssim_data = pd.read_csv(
        file, sep=",", header=None, engine="pyarrow", skiprows=1
    )

    return np.array(snr_ssim_data).astype(np.float32)

def plot_data(data: list["np.ndarray"], conditions: list[str],
             output_folder: str, frame_range: int, param: str) -> None:
    
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    cmap = mpl.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=500)

    # Axis labels

    if param == "SNR":
        y_unit = " (dB)"
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    else:
        y_unit = ""
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    #colors = cmap(np.linspace(0, 1, len(conditions)))
    colors = ["darkblue", "darkred", "darkviolet"]

    frame_axis = 1 + np.arange(frame_range)

    for i in range(len(data)):

        data_20k = data[i][:frame_range]

        ax.plot(frame_axis, data_20k, ls='-', linewidth=4.5,
                 alpha=0.7, color=colors[i],  label=conditions[i])
    
    leg = plt.legend(bbox_to_anchor=(0.5, 1.125), loc='upper center', ncol=3)

    for line in leg.get_lines():
        line.set_linewidth(3.5)
        line.set_alpha(1.0)

    for text in leg.get_texts():
        text.set_fontsize(28)
    
    ax.set_xlim(left=0, right=np.max(frame_axis) + 100)

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
    ax.set_ylabel(param + y_unit, labelpad=2, fontsize=32)

    plt.savefig(os.path.join(output_folder, param + "_plot.svg"))
    plt.savefig(os.path.join(output_folder, param + "_plot.png"))

def save_stats(mean_stats: "np.ndarray", conditions: list[str],
               dataset_num: int, output_folder: str) -> None:
    
    dataset_num_list = [str(dataset_num)] * mean_stats.shape[0]
    
    dataframe = pd.DataFrame(mean_stats, columns=["SNR (dN)", "SSIM"])

    dataframe.insert(0, "Conditions", conditions)
    dataframe.insert(0, "Dataset", dataset_num_list)

    dataframe.to_csv(os.path.join(output_folder, "mean_stats.csv"), index=False)

def main():

    ##--------PATHS-------##

    input_folder = "C:/Users/mxq76232/Documents/PhD/Project_work/dn_paper_data/d3_snr_ssim_meas/tub"
    output_folder = "C:/Users/mxq76232/Documents/PhD/Project_work/dn_paper_data/d3_snr_ssim_meas/tub_results"
    
    ##--------INITIALIZE-------##
    
    conditions = [
        #"MT, self-trained",
        "NPC, MT-trained",
        #"NPC, self-trained",
        "ACE-2, MT-trained"
        #"ACE-2, self-trained"
    ]

    dataset_num = 3

    frame_range = 20000

    all_snr_data = []
    all_ssim_data = []

    mean_stats = np.zeros((len(conditions), 2))

    ##--------PROGRAM START-------##

    all_files = collate_files(input_folder)

    for i, file in enumerate(all_files):

        data = load_snr_ssim_file(file)

        mean_stats[i, :] = np.mean(data, axis=0)

        all_snr_data.append(data[:, 0])
        all_ssim_data.append(data[:, 1])
    
    plot_data(all_snr_data, conditions, output_folder, frame_range, param="SNR")
    plot_data(all_ssim_data, conditions, output_folder, frame_range, param="SSIM")

    save_stats(mean_stats, conditions, dataset_num, output_folder)

if __name__ == "__main__":
    main()