import numpy as np
import pandas as pd
import os
import sys
import glob

from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# ==========================================================
# 1. PATH SETUP & IMPORT ALGORITHMS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ALGO_DIR = os.path.join(BASE_DIR, "..", "algorithm")

if ALGO_DIR not in sys.path:
    sys.path.append(ALGO_DIR)

from falqon_compact import FalqonOptimizer


# ==========================================================
# 2. LOAD INPUT DATA
# ==========================================================

df = pd.read_csv(os.path.join(DATA_DIR, "ddi_a.csv"))


# ==========================================================
# 3. BUILD DRUG INDEX & SYSTEM SIZE
# ==========================================================

drugs = sorted(set(df["drug_i"]).union(df["drug_j"]))
drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}

print("List of drugs:", drugs)
print("Drug-to-index mapping:", drug_to_idx)

n = len(drugs)
print("Number of qubits required:", n)


# ==========================================================
# 4. HAMILTONIAN CONSTRUCTION FUNCTIONS
# ==========================================================


def build_hamiltonian(h, J, c, n):
    paulis, coeffs = [], []

    # Linear Z terms
    for i in range(n):
        if h[i] != 0.0:
            z = ["I"] * n
            z[n - 1 - i] = "Z"
            paulis.append("".join(z))
            coeffs.append(h[i])

    # Quadratic ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                z = ["I"] * n
                z[n - 1 - i] = "Z"
                z[n - 1 - j] = "Z"
                paulis.append("".join(z))
                coeffs.append(J[i, j])

    # Constant shift
    if c != 0.0:
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
# 5. FIXED DATA & INITIAL STATES
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

H_d = driver_hamiltonian(n)
H_d_matrix = H_d.to_matrix()

psi_0 = np.ones((2**n, 1)) / np.sqrt(2**n)
times = np.linspace(0, 10, 4001)

mu = 5.0
B = 2.5


# ==========================================================
# 6. CLEAN OLD OUTPUT FILES
# ==========================================================

print("\n--- Cleaning previous outputs ---")
files_to_remove = (
    glob.glob("simulation_data_ki_*.npz")
    + glob.glob("FALQON_k_*_bitstring_data.csv")
    + glob.glob("simulation_summary_all_ki.csv")
)

for f in files_to_remove:
    try:
        os.remove(f)
        print(f"Removed: {f}")
    except OSError:
        pass

print("--- Cleanup completed ---\n")


# ==========================================================
# 7. MAIN SIMULATION LOOP
# ==========================================================

K_values = [3, 4]
summary_data = []

for k_val in K_values:
    print(f"\n{'='*40}\nRunning SCO-FALQON for k = {k_val}\n{'='*40}")

    h = np.zeros(n)
    J = np.zeros((n, n))

    sum_w_syn = sum(w for (_, _, w) in syn_edges)
    sum_w_harm = sum(w for (_, _, w) in harm_edges)

    c_sco = (
        -0.25 * sum_w_syn
        + 0.25 * B * sum_w_harm
        + mu * ((n / 2 - k_val) ** 2)
        + mu * n / 4
    )

    for i in range(n):
        s_syn = sum(w for (u, v, w) in syn_edges if u == i or v == i)
        s_harm = sum(w for (u, v, w) in harm_edges if u == i or v == i)
        h[i] = 0.25 * s_syn - 0.25 * B * s_harm - mu * (n / 2 - k_val)

    for i, j, w in syn_edges:
        J[i, j] += -0.25 * w

    for i, j, w in harm_edges:
        J[i, j] += 0.25 * B * w

    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] += mu / 2

    H_sco = build_hamiltonian(h, J, c_sco, n)
    H_sco_matrix = csr_matrix(H_sco.to_matrix())

    eigvals, _ = eigsh(H_sco_matrix, k=1, which="SA")
    E_gs = eigvals[0]

    optimizer = FalqonOptimizer(H_sco_matrix, H_d_matrix)
    energies, states, _ = optimizer.run_default(
        psi_0, times, ite=True, ite_dt=0.01, store_states=True
    )

    final_state = states[-1]
    probs = np.abs(final_state.flatten()) ** 2
    bitstrings = [format(i, f"0{n}b") for i in range(len(probs))]

    k_str = str(k_val).replace(".", "_")

    np.savez(
        f"simulation_data_ki_{k_str}.npz",
        energies=energies,
        ground_state_energy=E_gs,
        probs=probs,
        bitstrings=bitstrings,
        times=times,
    )

    pd.DataFrame({"bitstring": bitstrings, "probability": probs}).sort_values(
        "probability", ascending=False
    ).to_csv(f"FALQON_k_{k_str}_bitstring_data.csv", index=False)

    summary_data.append(
        {
            "K_value": k_val,
            "Final_Energy": energies[-1],
            "Ground_State_Energy": E_gs,
            "Max_Probability": probs.max(),
            "Most_Probable_Bitstring": bitstrings[np.argmax(probs)],
        }
    )


# ==========================================================
# 8. SAVE FINAL SUMMARY
# ==========================================================

pd.DataFrame(summary_data).to_csv("simulation_summary_all_ki.csv", index=False)
print("\n--- ALL SCO-FALQON SIMULATIONS COMPLETED ---")
