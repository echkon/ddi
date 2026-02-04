import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import matplotlib.ticker as ticker
import os

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATTERN = os.path.join(BASE_DIR, "simulation_data_Bc_*.npz")

DRUG_LIST = ["DEX", "FPV", "HCQ", "LPV", "MOV", "NTZ", "PAX", "RBV", "RDV"]


# ==============================================================================
# PLOTTING UTILITIES
# ==============================================================================


def apply_paper_style(ax, title, xlabel, ylabel):
    # ax.set_title(title, fontsize=16, pad=15, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=26, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=26, labelpad=8)

    ax.tick_params(axis="both", which="major", labelsize=12, width=2.0, length=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    ax.grid(True, linestyle=":", alpha=0.001)


# ============================================================
# BITSTRING DECODER
# ============================================================


def decode_drugs(bitstring, drug_list):
    selected = []
    n = len(bitstring)

    for i, char in enumerate(bitstring):
        if char == "1":
            qubit_idx = n - 1 - i
            if qubit_idx < len(drug_list):
                selected.append(drug_list[qubit_idx])
            else:
                selected.append(f"Unknown_q{qubit_idx}")

    return ", ".join(selected) if selected else "No Drugs Selected"


# ============================================================
# TOP-10 HISTOGRAM
# ============================================================


def plot_top10_histogram(bitstrings, probs, A_val, filename):
    sorted_indices = np.argsort(probs)[::-1]

    top_k = 10
    top_indices = sorted_indices[:top_k]

    top_bitstrings = [bitstrings[i] for i in top_indices]
    top_probs = probs[top_indices]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        top_bitstrings,
        top_probs,
        color="steelblue",
        edgecolor="black",
        width=0.6,
        alpha=0.8,
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = plt.gca()
    title = f"Top {top_k} Probabilities (B={A_val})"
    apply_paper_style(ax, title, "Bitstrings", "Probability")

    ax.set_ylim(0.0, 1.0)
    ax.grid(False)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=11,
        family="monospace",
    )

    plt.tight_layout()

    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()

    print(f"   Saved Top 10 Histogram: {filename}")


# ============================================================
# COMBINED ENERGY PLOT
# ============================================================


def plot_combined_energies(energy_dict, gs_dict, times_dict, filename):
    plt.rcParams.update({"mathtext.fontset": "stix"})
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    sorted_keys = sorted(energy_dict.keys(), key=lambda x: float(x))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(sorted_keys))]

    for B_val, color in zip(sorted_keys, colors):
        energies = energy_dict[B_val]

        if B_val in times_dict:
            t_axis = times_dict[B_val]
        else:
            t_axis = np.arange(len(energies))

        plt.plot(
            t_axis,
            energies,
            label=rf"$\gamma={B_val}$",
            linewidth=4.0,
            color=color,
            alpha=1.0,
        )

        if B_val in gs_dict:
            plt.axhline(
                y=gs_dict[B_val],
                color=color,
                linestyle="--",
                linewidth=2.5,
                alpha=1.0,
                label="_nolegend_",
            )

    apply_paper_style(ax, "FALQON Energy Evolution", "Time (t)", "Energy")

    ax.set_ylim(-2.8, 5.5)
    ax.tick_params(axis="both", which="major", labelsize=30)

    ax.set_xlabel(r"$\mathit{t}$", fontsize=45)
    ax.set_ylabel(r"$\mathit{E(t)}$", fontsize=45)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.legend(fontsize=30, loc="upper right", bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()

    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()

    print(f"Saved combined energy plot: {filename}")


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

if __name__ == "__main__":
    print("\n=== START PLOTTING FROM .NPZ FILES (Bc) ===")

    found_files = glob.glob(FILE_PATTERN)
    if not found_files:
        print(f"Error: No data files found matching pattern: {FILE_PATTERN}")
        exit()

    print(f"Found {len(found_files)} data files.")

    all_energies_history = {}
    all_ground_states = {}
    all_times = {}

    for sim_file in found_files:
        match = re.search(r"Bc_(\d+(?:_\d+)?)", sim_file)
        if not match:
            continue

        B_str_raw = match.group(1)
        B_val_str = B_str_raw.replace("_", ".")

        print(f"\n>> Processing B = {B_val_str} (File: {sim_file})")

        try:
            data = np.load(sim_file)

            energies = data["energies"]
            probs = data["probs"]
            bitstrings = data["bitstrings"]

            all_times[B_val_str] = (
                data["times"] if "times" in data else np.arange(len(energies))
            )

            if "ground_state_energy" in data:
                gs_val = data["ground_state_energy"]
                all_ground_states[B_val_str] = (
                    gs_val.item() if hasattr(gs_val, "item") else gs_val
                )

            all_energies_history[B_val_str] = energies

            hist_filename = os.path.join(BASE_DIR, f"Hist_Top10_Bc_{B_str_raw}.eps")
            plot_top10_histogram(bitstrings, probs, B_val_str, hist_filename)

        except Exception as e:
            print(f"Error while reading {sim_file}: {e}")

    print("\n--- PLOTTING COMBINED ENERGY GRAPH ---")

    if all_energies_history:
        combined_filename = os.path.join(BASE_DIR, "Combined_FALQON_Energy_All_Bc.eps")
        plot_combined_energies(
            all_energies_history,
            all_ground_states,
            all_times,
            combined_filename,
        )
    else:
        print("No data available for combined plot.")

    print("\n=== FINISHED ===")
