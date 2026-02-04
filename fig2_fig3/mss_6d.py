import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import os
import glob
import sys
import os

# ==========================================================
# PATH SETUP & IMPORT ALGORITHMS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(BASE_DIR, "..", "algorithm")

if ALGO_DIR not in sys.path:
    sys.path.append(ALGO_DIR)

from falqon import FalqonOptimizer


# ==========================================================
# 1. LOAD INPUT DATA
# ==========================================================

# Define base directory (location of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to shared data directory
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Load drug–drug interaction (DDI) data
df = pd.read_csv(os.path.join(DATA_DIR, "ddi_a.csv"))


# ==========================================================
# 2. BUILD DRUG INDEX AND SYSTEM SIZE
# ==========================================================

# Create a sorted list of unique drugs
drugs = set(df["drug_i"]).union(set(df["drug_j"]))
drugs = sorted(list(drugs))

# Map each drug to an integer index
drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}

print("List of drugs:", drugs)
print("Drug-to-index mapping:", drug_to_idx)

# Number of qubits equals the number of unique drugs
n = len(drugs)
print("Number of qubits required:", n)

# ==========================================================
# DRUG LABEL INTERPRETATION
# ==========================================================

idx_to_drug = {idx: drug for drug, idx in drug_to_idx.items()}

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
    Qiskit uses little-endian convention.
    Rightmost bit corresponds to qubit 0.
    """
    selected = []
    n = len(bitstring)

    for i, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            letter = idx_to_drug[i]  # qubit i
            real_name = label_map[letter]
            selected.append(real_name)

    return selected


# ==========================================================
# 3. CONSTRUCT PROBLEM HAMILTONIAN
# ==========================================================


def build_hamiltonian(h, J, c, n):

    paulis, coeffs = [], []

    # Linear Z terms
    for i in range(n):
        if abs(h[i]) != 0.0:
            z = ["I"] * n
            z[n - 1 - i] = "Z"
            paulis.append("".join(z))
            coeffs.append(h[i])

    # Quadratic ZZ interaction terms
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) != 0.0:
                z = ["I"] * n
                z[n - 1 - i] = "Z"
                z[n - 1 - j] = "Z"
                paulis.append("".join(z))
                coeffs.append(J[i, j])

    # Constant identity term
    if abs(c) != 0.0:
        paulis.append("I" * n)
        coeffs.append(c)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def driver_hamiltonian(n):

    paulis, coeffs = [], []

    for i in range(n):
        x = ["I"] * n
        x[n - 1 - i] = "X"
        paulis.append("".join(x))
        coeffs.append(1.0)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


# ==========================================================
# 4. FIXED DATA USED ACROSS ALL SIMULATIONS
# ==========================================================

# Extract harmful drug–drug interactions
harm_edges = [
    (drug_to_idx[di], drug_to_idx[dj], w)
    for di, dj, kind, w in zip(df["drug_i"], df["drug_j"], df["kind"], df["weight"])
    if kind == "harm"
]

# Sum of all harmful interaction weights
sum_w_harm = sum(w for (_, _, w) in harm_edges)

# Driver Hamiltonian and its matrix form
H_d = driver_hamiltonian(n)
H_d_matrix = H_d.to_matrix()

# Initial equal superposition state
psi_0 = np.ones((2**n, 1)) / np.sqrt(2**n)

# Time grid for FALQON evolution
times = np.linspace(0, 50, 3001)


# ==========================================================
# 5. CLEAN OLD OUTPUT FILES BEFORE RUNNING
# ==========================================================

print("\n--- Checking and removing old simulation outputs ---")

files_to_remove = (
    glob.glob("simulation_data_A_*.npz")
    + glob.glob("FALQON_A_*_bitstring_data.csv")
    + glob.glob("simulation_summary_all_A.csv")
)

if files_to_remove:
    for f in files_to_remove:
        try:
            os.remove(f)
            print(f"Removed old file: {f}")
        except OSError as e:
            print(f"Failed to remove {f}: {e}")
else:
    print("No old output files found.")

print("--- Cleanup completed. Starting new simulations ---\n")


# ==========================================================
# 6. MAIN SIMULATION LOOP
# ==========================================================

# Values of parameter A to be simulated
A_values = [2.5, 3.5, 4.5, 5.0]

# Store summary results for all A values
summary_data = []

for A_val in A_values:
    print(f"\n--- Running simulation for A = {A_val:.1f} ---")

    # ------------------------------------------------------
    # A. Construct problem Hamiltonian for given A
    # ------------------------------------------------------

    h = np.zeros(n)
    J = np.zeros((n, n))
    c_mss = -n / 2 + (A_val / 4) * sum_w_harm

    # Compute linear coefficients
    for i in range(n):
        sum_h = sum(w for (u, v, w) in harm_edges if u == i or v == i)
        h[i] = 0.5 - (A_val / 4) * sum_h

    # Compute quadratic coefficients
    for i, j, w in harm_edges:
        J[i, j] = (A_val / 4) * w

    # Encode Hamiltonian
    h_mss = build_hamiltonian(h, J, c_mss, n)

    # ------------------------------------------------------
    # B. Compute ground-state energy
    # ------------------------------------------------------

    H_mss_matrix = csr_matrix(h_mss.to_matrix())
    eigvals, _ = eigsh(H_mss_matrix, k=1, which="SA")
    ground_state_energy = eigvals[0]

    print(f"Ground-state energy (A={A_val:.1f}): {ground_state_energy:.4f}")

    # ------------------------------------------------------
    # C. Run FALQON optimization
    # ------------------------------------------------------

    optimizer = FalqonOptimizer(H_mss_matrix, H_d_matrix)
    energies, states, _ = optimizer.run(psi_0, times)

    # ------------------------------------------------------
    # D. Post-processing and data storage
    # ------------------------------------------------------

    A_str = f"{A_val:.1f}".replace(".", "_")
    base_name = f"FALQON_A_{A_str}"

    final_state = states[-1]
    probs = np.abs(final_state.flatten()) ** 2
    bitstrings = [format(i, f"0{n}b") for i in range(len(probs))]

    # Save numerical results
    np.savez(
        f"simulation_data_A_{A_str}.npz",
        energies=energies,
        ground_state_energy=ground_state_energy,
        probs=probs,
        bitstrings=bitstrings,
        times=times,
    )

    # Save bitstring probability distribution
    df_bitstring = pd.DataFrame(
        {"bitstring": bitstrings, "probability": probs}
    ).sort_values(by="probability", ascending=False)
    df_bitstring.to_csv(f"{base_name}_bitstring_data.csv", index=False)

    # ------------------------------------------------------
    # E. Collect summary statistics
    # ------------------------------------------------------

    top_k = 3

    # Indices of top-k probabilities (descending)
    top_indices = np.argsort(probs)[-top_k:][::-1]

    top_bitstrings = [bitstrings[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]

    # Interpret each bitstring
    top_selected_drugs = [
        interpret_bitstring(bs, idx_to_drug, label_map) for bs in top_bitstrings
    ]

    # Save summary (store top-3 explicitly)
    summary_data.append(
        {
            "A_value": A_val,
            "Final_Energy": energies[-1],
            "Ground_State_Energy": ground_state_energy,
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
# 7. SAVE FINAL SUMMARY FOR ALL A VALUES
# ==========================================================

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("simulation_summary_all_A.csv", index=False)

print("\nTop 3 most probable solutions (last A value):")
for k in range(3):
    print(
        f"#{k+1}: {top_bitstrings[k]} "
        f"(p = {top_probs[k]:.4f}) -> {top_selected_drugs[k]}"
    )

print("\n--- ALL SIMULATIONS COMPLETED SUCCESSFULLY ---")
print("Summary file saved: simulation_summary_all_A.csv")
