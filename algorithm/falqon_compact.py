# falqon_compact.py (patched)
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


class FalqonOptimizer:
    def __init__(self, h_cost, h_drive, dtype=np.complex64):
        self.dtype = np.dtype(dtype)
        # Force CSR sparse with consistent dtype
        self.h_cost = self._to_csr(h_cost)
        self.h_drive = self._to_csr(h_drive)

    def _to_csr(self, M):
        if sparse.issparse(M):
            return M.asformat("csr").astype(self.dtype, copy=False)
        else:
            return sparse.csr_matrix(M, dtype=self.dtype)

    def beta(self, state, lam=0.0):
        # β = -i <ψ| [Hd, Hc] |ψ>  (since [Hd, lam Hd]=0)
        Hc, Hd = self.h_cost, self.h_drive
        v1 = Hd.dot(Hc.dot(state))
        v2 = Hc.dot(Hd.dot(state))
        val = -1j * np.vdot(state, (v1 - v2))
        return float(val.real)

    def quantum_state(self, dt, state, beta_val, lam=0.0):
        # Build H = Hc + (lam+beta) Hd  (still CSR)
        H = self.h_cost + (lam + beta_val) * self.h_drive
        # Provide traceA for performance/stability: trace((-i dt) H)
        trH = float(H.diagonal().sum().real)  # real for Hermitian
        A = (-1j * dt) * H
        traceA = (-1j * dt) * trH
        return expm_multiply(A, state, traceA=traceA)

    def expectation_value(self, state, lam=0.0):
        H = self.h_cost + lam * self.h_drive
        v = H.dot(state)
        return float(np.vdot(state, v).real)

    def run_default(
        self,
        psi,
        times,
        store_states=False,
        ite=False,
        ite_step=2,
        ite_dt=0.1,
        lam=0.0,
    ):
        psi = np.asarray(psi, dtype=self.dtype)
        n = len(times)
        if n < 2:
            raise ValueError("times must have at least 2 points")
        dt = np.diff(times, prepend=times[0])

        energies, betas = [], []
        states = [] if store_states else None

        # Precompute for ITE: use Hc ψ directly (no building I-αH)
        for i in range(n):
            beta_val = self.beta(psi, lam) * 3.0
            if dt[i] != 0.0:
                psi = self.quantum_state(dt[i], psi, beta_val, lam)

            if ite and i % ite_step == 0:
                psi = psi - ite_dt * (self.h_cost.dot(psi))
                nrm = np.linalg.norm(psi)
                if not np.isfinite(nrm) or nrm == 0:
                    raise FloatingPointError("State norm invalid during ITE step.")
                psi /= nrm

            e = self.expectation_value(psi, lam)
            energies.append(e)
            betas.append(beta_val)
            if store_states:
                states.append(psi.copy())

        if store_states:
            return np.asarray(energies), states, np.asarray(betas)
        return np.asarray(energies), psi, np.asarray(betas)

    def run_kick(self, psi, times, beta_c=0.0, a_max=0.0, lam=0.0, store_states=False):
        psi = np.asarray(psi, dtype=self.dtype)
        n = len(times)
        dt = np.diff(times, prepend=times[0])
        T = float(times[-1]) if n else 0.0
        energies, betas = [], []
        states = [] if store_states else None

        for i in range(n):
            beta_val = self.beta(psi, lam)
            a_t = a_max * np.sin(np.pi * times[i] / (2 * T)) ** 2 if T > 0 else 0.0
            prob = abs(beta_c - beta_val) * a_t
            if np.random.rand() < prob:
                beta_val = beta_c
            if dt[i] != 0.0:
                psi = self.quantum_state(dt[i], psi, beta_val, lam)
            energies.append(self.expectation_value(psi, lam))
            betas.append(beta_val)
            if store_states:
                states.append(psi.copy())

        if store_states:
            return np.asarray(energies), states, np.asarray(betas)
        return np.asarray(energies), psi, np.asarray(betas)

    def run_perturbation(self, psi, times, p_max=0.0, store_states=False):
        psi = np.asarray(psi, dtype=self.dtype)
        n = len(times)
        dt = np.diff(times, prepend=times[0])
        T = float(times[-1]) if n else 0.0
        energies, betas = [], []
        states = [] if store_states else None

        for i in range(n):
            lam = (
                p_max * np.sin(np.pi * times[i] / (2 * T) - np.pi / 2) ** 2
                if T > 0
                else 0.0
            )
            beta_val = self.beta(psi, lam)
            if dt[i] != 0.0:
                psi = self.quantum_state(dt[i], psi, beta_val, lam)
            energies.append(self.expectation_value(psi, lam))
            betas.append(beta_val)
            if store_states:
                states.append(psi.copy())

        if store_states:
            return np.asarray(energies), states, np.asarray(betas)
        return np.asarray(energies), psi, np.asarray(betas)
