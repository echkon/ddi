import os
import csv
import itertools
import numpy as np

# =====================
# 1. PATH SETUP
# =====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
RESULT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "result"))

os.makedirs(RESULT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "ddi_covid.csv")
OUTPUT_FILE = os.path.join(RESULT_DIR, "mss_result.txt")


# =====================
# 2. LOAD DATA
# =====================

data = []
drugs = set()

with open(DATA_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d1 = row["drug_i"]
        d2 = row["drug_j"]
        kind = row["kind"]
        w = float(row["weight"])

        data.append((d1, d2, kind, w))

        drugs.add(d1)
        drugs.add(d2)

drug_list = sorted(list(drugs))
drug_index = {d: i for i, d in enumerate(drug_list)}

N = len(drug_list)

print("Drugs:", drug_list)
print("N =", N)


# =====================
# 3. MSS ENERGY FUNCTION
# =====================


def compute_energy(x, alpha):
    x = np.array(x)

    reward = -np.sum(x)

    penalty = 0

    for d1, d2, kind, w in data:
        if kind == "harm":
            i = drug_index[d1]
            j = drug_index[d2]
            penalty += w * x[i] * x[j]

    return reward + alpha * penalty


# =====================
# 4. BRUTE FORCE
# =====================

A_values = [2.5, 3.5, 4.5, 5.0]

for alpha in A_values:
    print(f"\nRunning brute force for alpha = {alpha}")

    best_energy = float("inf")
    best_config = None

    all_energies = []
    all_bitstrings = []

    for bits in itertools.product([0, 1], repeat=N):
        energy = compute_energy(bits, alpha)

        all_energies.append(energy)
        all_bitstrings.append("".join(map(str, bits)))

        if energy < best_energy:
            best_energy = energy

    energies = np.array(all_energies)
    bitstrings = np.array(all_bitstrings)

    A_str = str(alpha).replace(".", "_")

    # =====================
    # FILE PATHS
    # =====================

    txt_file = os.path.join(RESULT_DIR, f"bruteforce_mss_covid_A_{A_str}.txt")

    # =====================
    # SAVE TXT
    # =====================

    # decode config
    index_drug = {v: k for k, v in drug_index.items()}

    def decode(x):
        return [index_drug[i] for i in range(N) if x[i] == 1]

    sorted_idx = np.argsort(energies)
    top_k = 5

    ground_idx = np.where(energies == best_energy)[0]

    with open(txt_file, "w") as f:
        f.write("=== BRUTE FORCE RESULT ===\n\n")

        f.write(f"Alpha = {alpha}\n")
        f.write(f"N = {N}\n")
        f.write(f"Total configurations = {2**N}\n\n")

        f.write("All ground-state configurations:\n")
        for i in ground_idx:
            bs = bitstrings[i]
            config = tuple(map(int, bs))
            f.write(f"{bs} -> {decode(config)} | E = {energies[i]:.4f}\n")

        f.write(f"\nTop {top_k} solutions:\n")

        for i in sorted_idx[:top_k]:
            bs = bitstrings[i]
            en = energies[i]
            config = tuple(map(int, bs))
            f.write(f"{bs} -> {decode(config)} | E = {en:.4f}\n")

    # =====================
    # PRINT
    # =====================

    print(f"Saved TXT: {txt_file}")
