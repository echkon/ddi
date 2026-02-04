import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm


class FalqonOptimizer:
    def __init__(self, h_cost=None, h_drive=None):
        self.h_cost = h_cost
        self.h_drive = h_drive

    def create_num_qubits(self):
        h_cost = self.create_h_cost()
        dim = len(h_cost)
        return int(np.log2(dim))

    def create_h_cost(self):
        if self.h_cost is None:
            self.h_cost = self.example_h_cost()
        return self.h_cost

    def create_h_drive(self):
        if self.h_drive is None:
            self.h_drive = self.example_h_drive()
        return self.h_drive

    def example_h_cost(self, N=4):
        pauli_op_list = []

        for j in range(N):
            s = ("I" * (N - j - 1)) + "Z" + ("I" * j)
            pauli_op_list.append((s, 1))

        for s in ["IIZZ", "IZZI", "ZZII", "ZIIZ"]:
            pauli_op_list.append((s, 1))

        h_op = SparsePauliOp.from_list(pauli_op_list)
        return 0.5 * h_op.to_matrix()

    def example_h_drive(self):
        N = self.create_num_qubits()
        pauli_op_list = []

        for j in range(N):
            s = "I" * (N - j - 1) + "X" + ("I" * j)
            pauli_op_list.append((s, 1))

        h_op = SparsePauliOp.from_list(pauli_op_list)
        return h_op.to_matrix()

    def beta(self, state):
        h_cost = self.create_h_cost()
        h_drive = self.create_h_drive()
        commutator = h_drive @ h_cost - h_cost @ h_drive
        return (-1j * state.conj().T @ commutator @ state)[0, 0]

    def quantum_state(self, t, state):
        beta_val = self.beta(state)
        h_opt = self.create_h_cost() + beta_val * self.create_h_drive()
        U = expm(-1j * t * h_opt)
        return U @ state

    def expectation_value(self, state):
        h_cost = self.create_h_cost()
        return (state.conj().T @ h_cost @ state)[0, 0]

    def run(self, psi, times):
        exp_val = []
        psi_val = []
        betas = []

        delta_t = times[1]

        for _ in times:
            current_beta = self.beta(psi)
            betas.append(current_beta)

            psi = self.quantum_state(delta_t, psi)
            psi_val.append(psi)

            exp = self.expectation_value(psi)
            exp_val.append(np.real(exp))

        return exp_val, psi_val, betas

    def plot(self, times, energies, filename="energy_vs_time.png"):
        plt.plot(times, energies)
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
