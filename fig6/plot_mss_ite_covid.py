import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import os

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATTERN = os.path.join(BASE_DIR, "simulation_data_Ac_*.npz")

DRUG_LIST = ["DEX", "FPV", "HCQ", "LPV", "MOV", "NTZ", "PAX", "RBV", "RDV"]

# ==============================================================================
# PLOTTING UTILITIES
# ==============================================================================


def apply_paper_style(ax, title, xlabel, ylabel):
    # ax.set_title(title, fontsize=16, pad=15, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=24, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=24, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.6)


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
    title = f"Top {top_k} Probabilities (A={A_val})"
    apply_paper_style(ax, title, "Bitstrings", "Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=11, family="monospace"
    )
    plt.tight_layout()
    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()
    print(f"   Saved Top 10 Histogram: {filename}")


def plot_combined_energies(energy_dict, gs_dict, times_dict, filename):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    sorted_keys = sorted(energy_dict.keys(), key=lambda x: float(x))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(sorted_keys))]

    for A_val, color in zip(sorted_keys, colors):
        energies = energy_dict[A_val]

        if A_val in times_dict:
            t_axis = times_dict[A_val]
        else:
            t_axis = np.arange(len(energies))

        plt.plot(
            t_axis,
            energies,
            label=rf"$\alpha={A_val}$",
            linewidth=2.0,
            color=color,
            alpha=0.9,
        )

        if A_val in gs_dict:
            gs_energy = gs_dict[A_val]
            plt.axhline(
                y=gs_energy,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="_nolegend_",
            )
    ax.set_ylim(-6.5, -3.0)

    apply_paper_style(
        ax,
        "FALQON Energy Evolution for Different Penalty Factors (A)",
        "Time (t)",
        "Energy",
    )
    plt.legend(fontsize=24, loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()
    print(f"Saved combined plot: {filename}")


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

if __name__ == "__main__":
    print("\n=== START PLOTTING FROM .NPZ FILES ===")

    found_files = glob.glob(FILE_PATTERN)
    if not found_files:
        print(f"Error: No data files found matching pattern: {FILE_PATTERN}")
        exit()

    print(f"Found {len(found_files)} data files.")

    all_energies_history = {}
    all_ground_states = {}
    all_times = {}

    for sim_file in found_files:
        match = re.search(r"Ac_(\d+(?:_\d+)?)", sim_file)

        if match:
            A_str_raw = match.group(1)
            A_val_str = A_str_raw.replace("_", ".")
        else:
            continue

        print(f"\n>> Processing A = {A_val_str} (File: {sim_file})")

        try:
            data = np.load(sim_file)
            energies = data["energies"]
            probs = data["probs"]
            bitstrings = data["bitstrings"]

            if "times" in data:
                all_times[A_val_str] = data["times"]
            else:
                print("Warning: 'times' not found. Using index.")
                all_times[A_val_str] = np.arange(len(energies))

            if "ground_state_energy" in data:
                gs_val = data["ground_state_energy"]
                if hasattr(gs_val, "item"):
                    gs_val = gs_val.item()
                all_ground_states[A_val_str] = gs_val

            all_energies_history[A_val_str] = energies

            print(f"   [Decoding Top 3 Results for A={A_val_str}]")
            sorted_idx = np.argsort(probs)[::-1]

            for k in range(min(3, len(sorted_idx))):
                idx = sorted_idx[k]
                bs = bitstrings[idx]
                prob_val = probs[idx]
                decoded_names = decode_drugs(bs, DRUG_LIST)

                print(f"   #{k+1}: Bitstring {bs} (Prob: {prob_val:.4f})")
                print(f"       -> Drugs: [{decoded_names}]")

            print("-" * 40)

            hist_filename = os.path.join(BASE_DIR, f"Hist_Top10_Ac_{A_str_raw}.eps")
            plot_top10_histogram(bitstrings, probs, A_val_str, hist_filename)

        except Exception as e:
            print(f"Error while reading {sim_file}: {e}")

    print("\n--- PLOTTING COMBINED ENERGY GRAPH ---")
    if all_energies_history:
        combined_filename = os.path.join(BASE_DIR, "Combined_FALQON_Energy_All_Ac.eps")
        plot_combined_energies(
            all_energies_history,
            all_ground_states,
            all_times,
            combined_filename,
        )
    else:
        print("No data available for combined plot.")

    print("\n=== FINISHED ===")
