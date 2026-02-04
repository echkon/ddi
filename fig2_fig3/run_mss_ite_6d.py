import numpy as np
from qiskit.quantum_info import SparsePauliOp
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import os
import glob
import sys


# ==========================================================
# PATH SETUP & IMPORT ALGORITHMS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ALGO_DIR = os.path.join(BASE_DIR, "..", "algorithm")

if ALGO_DIR not in sys.path:
    sys.path.append(ALGO_DIR)

from falqon_compact import FalqonOptimizer


# ==========================================================
# 1. LOAD INPUT DATA
# ==========================================================

# Absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to shared data directory
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Input CSV file (drug–drug interaction data)
DATA_FILE = os.path.join(DATA_DIR, "ddi_a.csv")

# Load CSV (required columns: drug_i, drug_j, kind, weight)
df = pd.read_csv(DATA_FILE)

# ==========================================================
# 2. DRUG INDEXING AND SYSTEM SIZE
# ==========================================================

# Collect all unique drugs
drugs = sorted(set(df["drug_i"]).union(set(df["drug_j"])))

# Map each drug to a qubit index
drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}
idx_to_drug = {idx: drug for drug, idx in drug_to_idx.items()}

print("List of drugs:", drugs)
print("Drug-to-index mapping:", drug_to_idx)

# Number of qubits equals number of unique drugs
n = len(drugs)
print("Number of qubits:", n)

# ==========================================================
# 3. DRUG LABEL INTERPRETATION
# ==========================================================

label_map = {
    "A": "Ritonavir",
    "B": "Cabazitaxel",
    "C": "Metformin",
    "D": "Everolimus",
    "E": "Erlotinib",
    "F": "Topotecan",
}


def interpret_bitstring(bitstring, idx_to_drug, label_map):
    """
    Interpret a computational basis bitstring into selected drug names.

    Notes:
    - Qiskit uses little-endian convention
    - Rightmost bit corresponds to qubit 0
    """
    selected_drugs = []

    for qubit_idx, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            drug_label = idx_to_drug[qubit_idx]
            selected_drugs.append(label_map.get(drug_label, drug_label))

    return selected_drugs


# ==========================================================
# 4. HAMILTONIAN CONSTRUCTION
# ==========================================================


def build_hamiltonian(h, J, c, n):
    """
    Construct Ising Hamiltonian:
        H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + c I
    """
    paulis, coeffs = [], []

    # Linear Z terms
    for i in range(n):
        if h[i] != 0.0:
            z = ["I"] * n
            z[n - 1 - i] = "Z"
            paulis.append("".join(z))
            coeffs.append(h[i])

    # Quadratic ZZ interaction terms
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                z = ["I"] * n
                z[n - 1 - i] = "Z"
                z[n - 1 - j] = "Z"
                paulis.append("".join(z))
                coeffs.append(J[i, j])

    # Constant identity term
    if c != 0.0:
        paulis.append("I" * n)
        coeffs.append(c)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def driver_hamiltonian(n):
    """
    Transverse-field driver Hamiltonian:
        H_d = sum_i X_i
    """
    paulis, coeffs = [], []

    for i in range(n):
        x = ["I"] * n
        x[n - 1 - i] = "X"
        paulis.append("".join(x))
        coeffs.append(1.0)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


# ==========================================================
# 5. FIXED DATA ACROSS SIMULATIONS
# ==========================================================

# Extract harmful drug–drug interactions
harm_edges = [
    (drug_to_idx[di], drug_to_idx[dj], w)
    for di, dj, kind, w in zip(df["drug_i"], df["drug_j"], df["kind"], df["weight"])
    if kind == "harm"
]

sum_w_harm = sum(w for (_, _, w) in harm_edges)

H_d = driver_hamiltonian(n)
H_d_matrix = H_d.to_matrix()

psi_0 = np.ones((2**n, 1)) / np.sqrt(2**n)
times = np.linspace(0, 10, 1001)

# ==========================================================
# 6. CLEAN OLD OUTPUT FILES
# ==========================================================

print("\n--- Cleaning old simulation outputs ---")

files_to_remove = (
    glob.glob(os.path.join(BASE_DIR, "simulation_data_Ai_*.npz"))
    + glob.glob(os.path.join(BASE_DIR, "FALQON_Ai_*_bitstring_data.csv"))
    + glob.glob(os.path.join(BASE_DIR, "simulation_summary_all_Ai.csv"))
)

for f in files_to_remove:
    try:
        os.remove(f)
        print("Removed:", f)
    except OSError:
        pass

print("--- Cleanup completed ---\n")

# ==========================================================
# 7. MAIN SIMULATION LOOP
# ==========================================================

A_values = [2.5, 3.5, 4.5, 5.0]
summary_data = []

for A_val in A_values:
    print(f"\n--- Running simulation for A = {A_val:.1f} ---")

    # Build problem Hamiltonian
    h = np.zeros(n)
    J = np.zeros((n, n))
    c_mss = -n / 2 + (A_val / 4) * sum_w_harm

    for i in range(n):
        sum_h = sum(w for (u, v, w) in harm_edges if u == i or v == i)
        h[i] = 0.5 - (A_val / 4) * sum_h

    for i, j, w in harm_edges:
        J[i, j] = (A_val / 4) * w

    H_mss = build_hamiltonian(h, J, c_mss, n)

    # Ground-state energy
    H_matrix = csr_matrix(H_mss.to_matrix())
    eigvals, _ = eigsh(H_matrix, k=1, which="SA")
    ground_energy = eigvals[0]

    print(f"Ground-state energy: {ground_energy:.4f}")

    # Run FALQON
    optimizer = FalqonOptimizer(H_matrix, H_d_matrix)
    energies, states, betas = optimizer.run_default(
        psi_0, times, ite=True, store_states=True
    )

    # Final state analysis
    final_state = states[-1]
    probs = np.abs(final_state.flatten()) ** 2
    bitstrings = [format(i, f"0{n}b") for i in range(len(probs))]

    # Save raw data
    A_str = f"{A_val:.1f}".replace(".", "_")
    np.savez(
        f"simulation_data_Ai_{A_str}.npz",
        energies=energies,
        probs=probs,
        bitstrings=bitstrings,
        betas=betas,
        times=times,
        ground_state_energy=ground_energy,
    )

    # Save bitstring distribution
    df_bit = pd.DataFrame({"bitstring": bitstrings, "probability": probs}).sort_values(
        "probability", ascending=False
    )
    df_bit.to_csv(f"FALQON_Ai_{A_str}_bitstring_data.csv", index=False)

    top_k = 3

    # Indices of top-k probabilities (descending)
    top_indices = np.argsort(probs)[-top_k:][::-1]

    top_bitstrings = [bitstrings[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]

    # Interpret each bitstring
    top_selected_drugs = [
        interpret_bitstring(bs, idx_to_drug, label_map) for bs in top_bitstrings
    ]

    # Save summary
    summary_data.append(
        {
            "A_value": A_val,
            "Final_Energy": energies[-1],
            "Ground_State_Energy": ground_energy,
            "Bitstring_1": top_bitstrings[0],
            "Prob_1": top_probs[0],
            "Selected_Drugs_1": ", ".join(top_selected_drugs[0]),
            "Bitstring_2": top_bitstrings[1],
            "Prob_2": top_probs[1],
            "Selected_Drugs_2": ", ".join(top_selected_drugs[1]),
            "Bitstring_3": top_bitstrings[2],
            "Prob_3": top_probs[2],
            "Selected_Drugs_3": ", ".join(top_selected_drugs[2]),
        }
    )

# ==========================================================
# 9. SAVE FINAL SUMMARY
# ==========================================================

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("simulation_summary_all_Ai.csv", index=False)

print("\nTop 3 most probable solutions (last A value):")
for k in range(3):
    print(
        f"#{k+1}: {top_bitstrings[k]} "
        f"(p = {top_probs[k]:.4f}) -> {top_selected_drugs[k]}"
    )


print("\n--- ALL SIMULATIONS COMPLETED SUCCESSFULLY ---")
print("Summary saved to simulation_summary_all_Ai.csv")
