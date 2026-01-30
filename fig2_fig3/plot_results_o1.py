import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATTERN = os.path.join(BASE_DIR, "simulation_data_A_*.npz")


def apply_paper_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, pad=15, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.6)


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
    ax.grid(False)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=11, family="monospace"
    )
    plt.tight_layout()
    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()
    print(f" Saved Top 10 Histogram: {filename}")


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
            label=f"FALQON Energy (A={A_val})",
            linewidth=1.5,
            color=color,
            alpha=0.9,
        )

        if A_val in gs_dict:
            gs_energy = gs_dict[A_val]
            plt.axhline(
                y=gs_energy,
                color=color,
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label="_nolegend_",
            )

    ax.set_ylim(-3.1, -2)

    apply_paper_style(
        ax,
        "FALQON Energy Evolution for Different Penalty Factors (A)",
        "Time (t)",
        "Energy",
    )
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    print("\n=== Start plotting===")

    found_files = glob.glob(FILE_PATTERN)
    if not found_files:
        print(f"No data has been found: {FILE_PATTERN}")
        exit()

    print(f"Found {len(found_files)}.")

    all_energies_history = {}
    all_ground_states = {}
    all_times = {}

    for sim_file in found_files:
        match = re.search(r"simulation_data_A_(\d+(?:_\d+)?)", sim_file)

        if match:
            A_str_raw = match.group(1)
            A_val_str = A_str_raw.replace("_", ".")
        else:
            print(f"Warning: Cannot read A from file: {sim_file}")
            continue

        print(f"\n>> Proccessing A = {A_val_str} (File: {sim_file})")

        try:
            data = np.load(sim_file)
            energies = data["energies"]
            probs = data["probs"]
            bitstrings = data["bitstrings"]

            if "times" in data:
                all_times[A_val_str] = data["times"]
            else:
                print("Warning: Cannot find 'times'. Using index.")
                all_times[A_val_str] = np.arange(len(energies))

            if "ground_state_energy" in data:
                gs_val = data["ground_state_energy"]
                if hasattr(gs_val, "item"):
                    gs_val = gs_val.item()
                all_ground_states[A_val_str] = gs_val

            all_energies_history[A_val_str] = energies

            hist_filename = os.path.join(BASE_DIR, f"Hist_Top10_A_{A_str_raw}.eps")
            plot_top10_histogram(bitstrings, probs, A_val_str, hist_filename)

        except Exception as e:
            print(f"Cannot read {sim_file}: {e}")

    print("\n--- Plotting Combined Graph ---")
    if all_energies_history:
        combined_filename = os.path.join(BASE_DIR, "Combined_FALQON_Energy_All_A.eps")
        plot_combined_energies(
            all_energies_history,
            all_ground_states,
            all_times,
            combined_filename,
        )
    else:
        print("Cannot find data to plot.")

    print("\n=== DONE ===")
