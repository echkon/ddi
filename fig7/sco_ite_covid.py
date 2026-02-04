import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import os
import glob
import sys

# ==========================================================
# PATH SETUP & ALGORITHM IMPORTS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(BASE_DIR, "..", "algorithm")

if ALGO_DIR not in sys.path:
    sys.path.append(ALGO_DIR)

from falqon_compact import FalqonOptimizer

# ==========================================================
# 1. LOAD INPUT DATA
# ==========================================================

DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "ddi_covid.csv")

df = pd.read_csv(DATA_FILE)

# ----------------------------------------------------------
# Calculate the number of nodes (drugs)
# ----------------------------------------------------------

drugs = set(df["drug_i"]).union(set(df["drug_j"]))
drugs = sorted(list(drugs))

drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}

print("List of drugs:", drugs)
print("Mapping:", drug_to_idx)

n = len(drugs)
print("Number of qubits is needed to illustrate drugs", n)


# ==========================================================
# BUILD PROBLEM HAMILTONIAN
# ==========================================================


def build_hamiltonian(h, J, c, n):
    """
    Construct the Ising Hamiltonian:
        H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + c I
    """
    paulis, coeffs = [], []

    # Linear terms
    for i in range(n):
        if abs(h[i]) != 0.0:
            z = ["I"] * n
            z[n - 1 - i] = "Z"
            paulis.append("".join(z))
            coeffs.append(h[i])

    # Quadratic interaction terms
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) != 0.0:
                z = ["I"] * n
                z[n - 1 - i] = "Z"
                z[n - 1 - j] = "Z"
                paulis.append("".join(z))
                coeffs.append(J[i, j])

    # Constant term
    if abs(c) != 0.0:
        paulis.append("I" * n)
        coeffs.append(c)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


# ==========================================================
# DRIVER HAMILTONIAN
# ==========================================================


def driver_hamiltonian(n):
    """
    Construct the transverse-field driver Hamiltonian:
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
# FIXED DATA USED ACROSS ALL SIMULATIONS
# ==========================================================

harm_edges = [
    (drug_to_idx[di], drug_to_idx[dj], w)
    for di, dj, kind, w in zip(df["drug_i"], df["drug_j"], df["kind"], df["weight"])
    if kind == "harm"
]

syn_edges = [
    (drug_to_idx[di], drug_to_idx[dj], w)
    for di, dj, kind, w in zip(df["drug_i"], df["drug_j"], df["kind"], df["weight"])
    if kind == "synergy"
]

sum_w_harm = sum(w for (_, _, w) in harm_edges)
sum_w_syn = sum(w for (_, _, w) in syn_edges)

# Driver Hamiltonian
H_d = driver_hamiltonian(n)
H_d_matrix = H_d.to_matrix()

# Initial equal superposition state
psi_0 = np.ones((2**n, 1)) / np.sqrt(2**n)

# Time grid
times = np.linspace(0, 15, 15001)

# SCO parameters
mu = 5.0
k = 3


# ==========================================================
# SIMULATION EXECUTION & DATA SAVING
# ==========================================================

print("\n--- Checking and removing data from previous runs ---")

files_to_remove = (
    glob.glob("simulation_data_Bc_*.npz")
    + glob.glob("FALQON_Bc_*_bitstring_data.csv")
    + glob.glob("simulation_summary_all_Bc.csv")
)

if files_to_remove:
    for f in files_to_remove:
        try:
            os.remove(f)
            print(f"Removed old file: {f}")
        except OSError as e:
            print(f"Error removing {f}: {e}")
else:
    print("No old files found to remove.")

print("--- Cleanup completed, starting new simulations ---\n")

# List of B values to be simulated
B_values = [2.5, 3.5, 4.5, 5.0]

summary_data = []


# ==========================================================
# MAIN SIMULATION LOOP
# ==========================================================

for B_val in B_values:
    print(f"\n--- Running simulation for B = {B_val:.1f} ---")

    # ------------------------------------------------------
    # A. Construct SCO Hamiltonian
    # ------------------------------------------------------

    h = np.zeros(n)
    J = np.zeros((n, n))

    c_sco = (
        -0.25 * sum_w_syn
        + 0.25 * B_val * sum_w_harm
        + mu * ((n / 2 - k) ** 2)
        + mu * n / 4
    )

    for i in range(n):
        sum_s = sum(w for (u, v, w) in syn_edges if u == i or v == i)
        sum_h = sum(w for (u, v, w) in harm_edges if u == i or v == i)
        h[i] = 0.25 * sum_s - 0.25 * B_val * sum_h - mu * (n / 2 - k)

    for i, j, w in syn_edges:
        J[i, j] += -0.25 * w

    for i, j, w in harm_edges:
        J[i, j] += 0.25 * B_val * w

    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] += mu / 2

    h_sco = build_hamiltonian(h, J, c_sco, n)

    # ------------------------------------------------------
    # B. Compute Ground State Energy
    # ------------------------------------------------------

    H_sco_matrix = csr_matrix(h_sco.to_matrix())
    eigvals, eigvecs = eigsh(H_sco_matrix, k=1, which="SA")
    ground_state_energy = eigvals[0]

    print(f"Ground state energy for B={B_val:.1f}: {ground_state_energy:.4f}")

    # ------------------------------------------------------
    # C. Run FALQON Optimization
    # ------------------------------------------------------

    optimizer = FalqonOptimizer(H_sco_matrix, H_d_matrix)
    energies, states, _ = optimizer.run_default(
        psi_0, times, ite=True, store_states=True, ite_dt=0.001
    )

    # ------------------------------------------------------
    # D. Post-processing and Saving Results
    # ------------------------------------------------------

    B_str = f"{B_val:.1f}".replace(".", "_")
    plot_base_name = f"FALQON_Bc_{B_str}"

    final_state = states[-1]
    probs = np.abs(final_state.flatten()) ** 2
    bitstrings = [format(i, f"0{n}b") for i in range(len(probs))]

    output_filename = f"simulation_data_Bc_{B_str}.npz"
    np.savez(
        output_filename,
        energies=energies,
        ground_state_energy=ground_state_energy,
        probs=probs,
        bitstrings=bitstrings,
        times=times,
    )

    print(f"Saved NPZ data to file: {output_filename}")

    df_bitstring = pd.DataFrame(
        {"bitstring": bitstrings, "probability": probs}
    ).sort_values(by="probability", ascending=False)

    bitstring_csv_filename = f"{plot_base_name}_bitstring_data.csv"
    df_bitstring.to_csv(bitstring_csv_filename, index=False)

    print(f"Saved bitstring CSV data to file: {bitstring_csv_filename}")

    # ------------------------------------------------------
    # E. Extract summary information
    # ------------------------------------------------------

    max_prob_idx = np.argmax(probs)

    summary_data.append(
        {
            "B_value": B_val,
            "Final_Energy": energies[-1],
            "Ground_State_Energy": ground_state_energy,
            "Final_Prob_Max": probs[max_prob_idx],
            "Most_Probable_Bitstring": bitstrings[max_prob_idx],
        }
    )


# ==========================================================
# SAVE SUMMARY RESULTS FOR ALL PENALTY VALUES
# ==========================================================

df_summary = pd.DataFrame(summary_data)
summary_filename = "simulation_summary_all_Bc.csv"
df_summary.to_csv(summary_filename, index=False)

print(f"\n--- ALL SIMULATIONS COMPLETED ---")
print(f"Summary results saved to CSV file: {summary_filename}")
