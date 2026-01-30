import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import os


FILE_PATTERN = "simulation_data_ki_*.npz"


def apply_paper_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, pad=15, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.grid(True, linestyle=":", alpha=0.6)


def plot_top10_histogram(bitstrings, probs, k_val, filename):
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
    title = f"Top {top_k} Probabilities (k={k_val})"
    apply_paper_style(ax, title, "Bitstrings", "Probability")
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=11, family="monospace"
    )
    plt.tight_layout()
    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()
    print(f"   Da luu Top 10 Histogram: {filename}")


def plot_combined_energies(energy_dict, gs_dict, times_dict, filename):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    sorted_keys = sorted(energy_dict.keys(), key=lambda x: float(x))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(sorted_keys))]

    for k_val, color in zip(sorted_keys, colors):
        energies = energy_dict[k_val]

        if k_val in times_dict:
            t_axis = times_dict[k_val]
        else:
            t_axis = np.linspace(0, 10, len(energies))

        plt.plot(
            t_axis,
            energies,
            label=f"FALQON Energy (k={k_val})",
            linewidth=1.5,
            color=color,
            alpha=0.9,
        )

        if k_val in gs_dict:
            gs_energy = gs_dict[k_val]
            plt.axhline(
                y=gs_energy,
                color=color,
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label="_nolegend_",
            )

    apply_paper_style(
        ax,
        "FALQON Energy Evolution for Different Size Factors (k)",
        "Time (t)",
        "Energy",
    )
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, format="eps", dpi=300)
    plt.savefig(filename.replace(".eps", ".png"), format="png", dpi=300)
    plt.close()
    print(f"Da luu bieu do TONG HOP: {filename}")


if __name__ == "__main__":
    print("\n=== BAT DAU VE BIEU DO TU CAC FILE .NPZ (SCO - k) ===")

    found_files = glob.glob(FILE_PATTERN)
    if not found_files:
        print(f"Loi: Khong tim thay file: {FILE_PATTERN}")
        exit()

    print(f"Tim thay {len(found_files)} file du lieu.")

    all_energies_history = {}
    all_ground_states = {}
    all_times = {}

    for sim_file in found_files:
        match = re.search(r"ki_(\d+(?:_\d+)?)", sim_file, re.IGNORECASE)
        if match:
            k_val_str = match.group(1).replace("_", ".")
        else:
            continue

        print(f"\n>> Dang xu ly k = {k_val_str} (File: {sim_file})")

        try:
            data = np.load(sim_file)
            energies = data["energies"]
            probs = data["probs"]
            bitstrings = data["bitstrings"]

            if "times" in data:
                all_times[k_val_str] = data["times"]
            else:
                print("Warning: Khong tim thay 'times'. Su dung index mac dinh (0-10).")
                all_times[k_val_str] = np.linspace(0, 10, len(energies))

            if "ground_state_energy" in data:
                gs_val = data["ground_state_energy"]
                if hasattr(gs_val, "item"):
                    gs_val = gs_val.item()
                all_ground_states[k_val_str] = gs_val

            all_energies_history[k_val_str] = energies

            hist_filename = f"Hist_Top10_k_{k_val_str.replace('.', '_')}.eps"
            plot_top10_histogram(bitstrings, probs, k_val_str, hist_filename)

        except Exception as e:
            print(f"Loi khi doc file {sim_file}: {e}")

    # Vẽ biểu đồ tổng hợp
    print("\n--- DANG VE BIEU DO TONG HOP ---")
    if all_energies_history:
        plot_combined_energies(
            all_energies_history,
            all_ground_states,
            all_times,
            "Combined_FALQON_Energy_All_k.eps",
        )
    else:
        print("Khong du du lieu de ve bieu do tong hop.")

    print("\n=== HOAN TAT ===")
