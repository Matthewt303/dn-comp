import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import seaborn as sns
import os

def collate_stats(folder: str) -> list[str]:

    folder_files = os.listdir(folder)

    stat_files = [
        os.path.join(folder, file)
        for file in folder_files if file.endswith(".csv")
    ]

    return sorted(stat_files)

def load_stats(file: str) -> "np.ndarray":

    stats = pd.read_csv(file, sep=",", usecols=[0, 2, 3], 
                        header=None, engine="pyarrow", skiprows=1)

    return np.array(stats).astype(np.float32) 

def combine_stats(self_stats: list["np.ndarray"],
                  tub_stats: list["np.ndarray"],
                  cond_A: str,
                  cond_B: str,
                  out: str) -> "pd.DataFrame":
    
    self_trained_array = np.vstack(self_stats).reshape(-1, 3)
    tub_trained_array = np.vstack(tub_stats).reshape(-1, 3)

    self_designation = [cond_A] * self_trained_array.shape[0]
    tub_designation = [cond_B] * tub_trained_array.shape[0]

    all_designations = self_designation + tub_designation

    combined_array = np.vstack((self_trained_array, tub_trained_array))

    combined_df = pd.DataFrame(combined_array, columns=["Dataset", "SNR (dB)", "SSIM"])

    combined_df.insert(0, "Training Method", all_designations)

    combined_df.to_csv(os.path.join(out, "all_results.csv"), index=False)

    return combined_df

def calc_means(all_data: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame"]:

    repeats = sorted(set(all_data["Dataset"]))
    training_conds = sorted(set(all_data["Training Method"]))

    all_mean_snr = []
    all_mean_ssim = []

    for i in repeats:

        for cond in training_conds:

            subset = all_data.loc[(all_data["Dataset"] == i) &
                                  (all_data["Training Method"] == cond)]
            
            mean_snr = subset["SNR (dB)"].mean()
            mean_ssim = subset["SSIM"].mean()

            all_mean_snr.append(mean_snr)
            all_mean_ssim.append(mean_ssim)
    
    overall_conds = training_conds * len(repeats)
    overall_repeats = sorted(repeats * len(training_conds))

    mean_snr = pd.DataFrame(all_mean_snr, columns=["Mean SNR (dB)"])
    mean_ssim = pd.DataFrame(all_mean_ssim, columns=["Mean SSIM"])

    mean_snr.insert(0, "Dataset", overall_repeats)
    mean_snr.insert(0, "Training Method", overall_conds)

    mean_ssim.insert(0, "Dataset", overall_repeats)
    mean_ssim.insert(0, "Training Method", overall_conds)

    return mean_snr, mean_ssim
    
def export_legend(legend, filename: str):
    legend.get_frame().set_linewidth(0.0)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def plot_snr(all_data: "pd.DataFrame", output_folder: str) -> None:

    plt.ioff()

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 42

    sns.set_style("ticks")

    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True

    fig, ax = plt.subplots(figsize=(16, 16), dpi=500)

    dataset_total = len(set(all_data["Dataset"]))
    mean_snrs, _ = calc_means(all_data)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    graph = sns.stripplot(
        x=all_data.columns[0],
        y=all_data[all_data.columns[2]],
        data=all_data,
        hue = "Dataset",
        alpha=0.7,
        palette=["#0077BB", "#009988", "#EE7733"],
        size=9
    )

    sns.pointplot(
        data=all_data,
        x=all_data.columns[0],
        y=all_data[all_data.columns[2]],
        errorbar="sd",
        markers="_",
        linestyles="none",
        capsize=0.2,
        linewidth=4.0,
        color="black",
    )

    sns.pointplot(
        data=mean_snrs,
        x="Training Method",
        y="Mean SNR (dB)",
        errorbar=None,
        linestyles='none',
        hue='Dataset',
        markers=['*', 'D', '^'],
        palette=["#0077BB", "#009988", "#EE7733"],
        markersize=13.5) 

    graph.tick_params(labelsize=42, pad=4)
    graph.legend_.remove()

    leg = plt.legend(title="Dataset", bbox_to_anchor=(0.5, 1.195), loc='upper center', ncols=3,
                     title_fontsize=26, facecolor='white', frameon=False)
    
    leg.get_texts()[0].set_text('1')
    leg.get_texts()[1].set_text('2')
    leg.get_texts()[2].set_text('3')
    leg.get_texts()[3].set_text('Mean 1')
    leg.get_texts()[4].set_text('Mean 2')
    leg.get_texts()[5].set_text('Mean 3')

    for line in leg.get_lines():
        line.set_linewidth(3.5)
        line.set_alpha(1.0)
    
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    for text in leg.get_texts():
        text.set_fontsize(24)

    export_legend(leg, os.path.join(output_folder, "snr_plot_legend.svg"))

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis="y", which="major", length=6, direction="in")
    ax.tick_params(axis="y", which="minor", length=3, direction="in")
    ax.tick_params(axis="x", which="major", length=6, direction="in")
    ax.tick_params(axis="x", which="minor", length=3, direction="in")

    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["top"].set_linewidth(2.0)
    ax.spines["right"].set_linewidth(2.0)
    ax.spines["left"].set_linewidth(2.0)

    ax.set_xlabel("Training", labelpad=8, fontsize=42)
    ax.set_ylabel(all_data.columns[2], labelpad=12, fontsize=42)

    ax.get_legend().remove()

    plt.savefig(os.path.join(output_folder, "all_snr_plot.png"))
    plt.savefig(os.path.join(output_folder, "all_snr_plot.pdf"))
    plt.savefig(os.path.join(output_folder, "all_snr_plot.svg"))

def plot_ssim(all_data: "pd.DataFrame", output_folder: str) -> None:

    plt.ioff()

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 42

    sns.set_style("ticks")

    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True

    fig, ax = plt.subplots(figsize=(16, 16), dpi=500)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    dataset_total = len(set(all_data["Dataset"]))
    _, mean_ssims = calc_means(all_data)

    graph = sns.stripplot(
        x=all_data.columns[0],
        y=all_data[all_data.columns[3]],
        data=all_data,
        hue = "Dataset",
        alpha=0.5,
        palette=["#0077BB", "#009988", "#EE7733"],
        size=14
    )

    sns.pointplot(
        data=all_data,
        x=all_data.columns[0],
        y=all_data[all_data.columns[3]],
        errorbar="sd",
        markers="_",
        linestyles="none",
        capsize=0.2,
        linewidth=5.0,
        color="black",
    )

    sns.pointplot(
        data=mean_ssims,
        x="Training Method",
        y="Mean SSIM",
        errorbar=None,
        linestyles='none',
        hue='Dataset',
        markers=['*', 'D', '^'],
        palette=["#0077BB", "#009988", "#EE7733"],
        markersize=15.5) 

    graph.tick_params(labelsize=42, pad=4)
    graph.legend_.remove()

    leg = plt.legend(title="Dataset", bbox_to_anchor=(0.5, 1.155), loc='upper center', ncols=3,
                     title_fontsize=26, facecolor='white', frameon=False)
    leg.get_texts()[0].set_text('1')
    leg.get_texts()[1].set_text('2')
    leg.get_texts()[2].set_text('3')
    leg.get_texts()[3].set_text('Mean 1')
    leg.get_texts()[4].set_text('Mean 2')
    leg.get_texts()[5].set_text('Mean 3')
    leg.get_frame().set_linewidth(0.0)

    for line in leg.get_lines():
        line.set_linewidth(3.5)
        line.set_alpha(1.0)
    
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    for text in leg.get_texts():
        text.set_fontsize(24)

    export_legend(leg, os.path.join(output_folder, "ssim_plot_legend.svg"))

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis="y", which="major", length=6, direction="in")
    ax.tick_params(axis="y", which="minor", length=3, direction="in")
    ax.tick_params(axis="x", which="major", length=6, direction="in")
    ax.tick_params(axis="x", which="minor", length=3, direction="in")

    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    #ax.ticklabel_format(axis='y', style='plain')

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["top"].set_linewidth(2.0)
    ax.spines["right"].set_linewidth(2.0)
    ax.spines["left"].set_linewidth(2.0)

    ax.set_xlabel("Training", labelpad=8, fontsize=42)
    ax.set_ylabel(all_data.columns[3], labelpad=12, fontsize=42)

    ax.get_legend().remove()

    plt.savefig(os.path.join(output_folder, "all_ssim_plot.png"))
    plt.savefig(os.path.join(output_folder, "all_ssim_plot.pdf"))
    plt.savefig(os.path.join(output_folder, "all_ssim_plot.svg"))

def main():

    ##--------PATHS-------##

    self_trained = "C:/Users/mxq76232/Documents/PhD/Project_work/dn_paper_data/all_self_snr_ssim/self_train"
    tub_trained = "C:/Users/mxq76232/Documents/PhD/Project_work/dn_paper_data/all_self_snr_ssim/tub_train"
    out_folder = "C:/Users/mxq76232/Documents/PhD/Project_work/dn_paper_data/all_self_snr_ssim/all_snr_ssim_results"

    ##--------INITIALIZATION-------##

    # Mapping:
    # Self-trained -> EMCCD
    # Tub-trained -> sCMOS

    condition_a = "Self-trained"
    condition_b = "MT-trained"

    tub_train_stats = []
    self_train_stats = []

    ##--------PROGRAM START-------##

    tub_files = collate_stats(tub_trained)
    self_files = collate_stats(self_trained)

    for tub_file, self_file in zip(tub_files, self_files):

        tub_stats = load_stats(tub_file)
        self_stats = load_stats(self_file)

        tub_train_stats.append(tub_stats)
        self_train_stats.append(self_stats)
    
    all_stats = combine_stats(self_train_stats, tub_train_stats,
                              condition_a, condition_b, out_folder)
    
    all_mean_snrs, all_mean_ssims = calc_means(all_stats)

    all_mean_snrs.to_csv(os.path.join(out_folder, "all_mean_snrs.csv"), index=False)
    all_mean_ssims.to_csv(os.path.join(out_folder, "all_mean_ssims.csv"), index=False)

    plot_snr(all_stats, out_folder)
    plot_ssim(all_stats, out_folder)
    
if __name__ == "__main__":
    main()