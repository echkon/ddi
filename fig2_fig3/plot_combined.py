import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os


FILE_PATTERN_NO_ITE = "simulation_data_A_*.npz"
FILE_PATTERN_WITH_ITE = "simulation_data_Ai_*.npz"


def apply_paper_style(ax, title, xlabel, ylabel):
    # ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=24, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=24, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.6)


def extract_data(file_pattern, regex_pattern):
    data_dict = {}
    ground_states = {}
    time_axes = {}
    found_files = glob.glob(file_pattern)

    for sim_file in found_files:
        match = re.search(regex_pattern, sim_file)
        if match:
            A_val_str = match.group(1).replace("_", ".")
        else:
            continue

        try:
            data = np.load(sim_file)
            data_dict[A_val_str] = data["energies"]

            if "times" in data:
                time_axes[A_val_str] = data["times"]
            else:
                time_axes[A_val_str] = np.arange(len(data["energies"]))

            if "ground_state_energy" in data:
                gs = data["ground_state_energy"]
                if hasattr(gs, "item"):
                    gs = gs.item()
                ground_states[A_val_str] = gs
        except Exception as e:
            print(f"Lỗi đọc file {sim_file}: {e}")

    return data_dict, ground_states, time_axes


if __name__ == "__main__":
    print("--- ĐANG TỔNG HỢP DỮ LIỆU ---")

    # 1. Đọc dữ liệu
    energies_A, gs_A, times_A = extract_data(
        FILE_PATTERN_NO_ITE, r"simulation_data_A_(\d+(?:_\d+)?)\.npz"
    )
    energies_Ai, gs_Ai, times_Ai = extract_data(
        FILE_PATTERN_WITH_ITE, r"simulation_data_Ai_(\d+(?:_\d+)?)\.npz"
    )

    # Tìm A chung
    common_A = sorted(list(set(energies_A.keys()) | set(energies_Ai.keys())), key=float)
    if not common_A:
        print("Không tìm thấy dữ liệu.")
        exit()

    # 2. Khởi tạo biểu đồ (1 Hình duy nhất)
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(common_A))]

    handles_f, labels_f = [], []  # FALQON (Standard)
    handles_i, labels_i = [], []  # FALQON + ITE

    for i, A_val_str in enumerate(common_A):
        color = colors[i]

        # --- STANDARD (FALQON) ---
        if A_val_str in energies_A:
            (l,) = plt.plot(
                times_A[A_val_str],
                energies_A[A_val_str],
                label=f"A={A_val_str} (Standard)",
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=1,
            )
            handles_f.append(l)
            labels_f.append(f"A={A_val_str} (Standard)")

        # --- WITH ITE (FALQON + ITE) ---
        if A_val_str in energies_Ai:
            (l,) = plt.plot(
                times_Ai[A_val_str],
                energies_Ai[A_val_str],
                label=f"A={A_val_str} (w/ ITE)",
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=1,
            )
            handles_i.append(l)
            labels_i.append(f"A={A_val_str} (w/ ITE)")

    all_times = []
    if times_A:
        all_times.extend([t[-1] for t in times_A.values()])
    if times_Ai:
        all_times.extend([t[-1] for t in times_Ai.values()])
    if all_times:
        plt.xlim(0, max(all_times))

    plt.ylim(-3.2, 0.5)

    apply_paper_style(
        ax,
        "Comparison of FALQON Energy Evolution (FALQON vs FALQON ITE)",
        "Time (t)",
        "Energy",
    )

    final_handles = handles_f + handles_i
    final_labels = labels_f + labels_i

    plt.legend(
        final_handles,
        final_labels,
        loc="upper right",
        ncol=2,
        fontsize=16,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()

    output_file = "FALQON_Comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(
        output_file.replace(".png", ".eps"), format="eps", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Đã lưu biểu đồ: {output_file}")
