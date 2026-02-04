import numpy as np
import tqix as tq
import itertools
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
import itertools
import csv


class FalqonOptimizer:
    def __init__(self, h_cost=None, h_drive=None):
        self.h_cost = h_cost
        self.h_drive = h_drive

    def create_num_qubits(self):
        h_cost = self.create_h_cost()
        dim = len(h_cost)
        numq = int(np.log2(dim))
        return numq

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

        s = "IIZZ"
        pauli_op_list.append((s, 1))
        s = "IZZI"
        pauli_op_list.append((s, 1))
        s = "ZZII"
        pauli_op_list.append((s, 1))
        s = "ZIIZ"
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
        # print(h_cost)
        h_drive = self.create_h_drive()
        # print(h_drive)

        com = h_drive @ h_cost - h_cost @ h_drive
        # print(com)
        res = -1j * state.conj().T @ com @ state
        # print(res)
        return res[0, 0]

    def quantum_state(self, t, state):
        beta_val = self.beta(state)
        h_opt = self.create_h_cost() + beta_val * self.create_h_drive()
        U = expm(-1j * t * h_opt)
        state_evl = U @ state
        return state_evl

    def expectation_value(self, state):
        h_cost = self.create_h_cost()
        exp_val = state.conj().T @ h_cost @ state
        return exp_val

    # def run(self, psi, times):
    #     exp_val = []
    #     psi_val = []
    #     delta_t = times[1]
    #     for t in times:
    #         psi = self.quantum_state(delta_t, psi)
    #         psi_val.append(psi)
    #         exp = self.expectation_value(psi)
    #         exp_val.append(np.real(exp[0, 0]))
    #         # print(t, np.real(exp[0,0]))

    #     return exp_val, psi_val

    def run(self, psi, times):
        exp_val = []
        psi_val = []
        betas = []  # [MỚI] 1. Khởi tạo danh sách chứa các giá trị beta

        delta_t = times[1]  # Giả sử bước thời gian là đều

        for t in times:
            # [MỚI] 2. Tính toán beta hiện tại để lưu lại
            # (Lưu ý: Việc này tính beta thêm 1 lần nữa, nhưng đảm bảo an toàn logic)
            current_beta = self.beta(psi)

            # Nếu beta là số phức (dù phần ảo = 0), bạn có thể muốn lấy phần thực:
            # current_beta = np.real(self.beta(psi))

            betas.append(current_beta)

            # Cập nhật trạng thái
            psi = self.quantum_state(delta_t, psi)
            psi_val.append(psi)

            # Tính năng lượng kỳ vọng
            exp = self.expectation_value(psi)
            exp_val.append(np.real(exp[0, 0]))
            # print(t, np.real(exp[0,0]))

        # [MỚI] 3. Trả về thêm biến betas
        return exp_val, psi_val, betas

    def plot(self, data, fn="enesVstimes.png"):
        # plot energy vs time

        plt.plot(data[0], data[1], label="enes vs times")
        plt.savefig(fn)

    def plot_top_probabilities(self, P, p, num_top=10, filename=None):
        """
        Hàm này nhận vào danh sách xác suất và bitstring, sau đó vẽ biểu đồ
        cho num_top giá trị có xác suất cao nhất.
        """
        print(
            f"\nPlotting histogram for top {num_top} states with highest probibility..."
        )

        # 1. Kết hợp xác suất và bitstring
        combined_data = list(zip(P, p))

        # 2. Sắp xếp theo xác suất giảm dần
        sorted_data = sorted(combined_data, key=lambda item: item[0], reverse=True)

        # 3. Lấy `num_top` phần tử đầu tiên (hoặc ít hơn nếu không đủ)
        actual_num_top = min(num_top, len(sorted_data))
        top_data = sorted_data[:actual_num_top]

        # 4. Tách lại thành 2 danh sách để vẽ (xử lý trường hợp rỗng)
        if not top_data:
            top_P, top_p = [], []
        else:
            top_P, top_p = zip(*top_data)

        # --- Bắt đầu vẽ biểu đồ ---
        plt.figure(figsize=(12, 7))
        bars = plt.bar(top_p, top_P, color="seagreen", width=0.6)

        plt.title(f"Top {actual_num_top} Highest Probabilities", fontsize=18, pad=15)
        plt.xlabel("Bitstrings", fontsize=14, labelpad=8)
        plt.ylabel("Probability", fontsize=14, labelpad=8)
        plt.xticks(rotation=45, ha="right", fontsize=11, fontfamily="monospace")
        plt.yticks(fontsize=12)

        # Hiển thị giá trị trên mỗi cột
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.01,
                f"{yval:.3f}",
                ha="center",
                va="bottom",
            )

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        # THAY ĐỔI Ở ĐÂY
        if filename:
            plt.savefig(filename, dpi=400)  # Lưu file với chất lượng cao
            plt.close()  # Đóng figure để giải phóng bộ nhớ
        else:
            plt.show()  # Nếu không có filename thì mới hiển thị

    def visualization(
        self, state, filename_all=None, filename_top=None, filename_data=None
    ):
        up = tq.bx(2, 0)
        dn = tq.bx(2, 1)
        dim = len(state)
        numq = int(np.log2(dim))

        # Tạo danh sách bitstrings
        p = ["".join(bits) for bits in itertools.product("01", repeat=numq)]
        s = [tq.tensorx(*(up if bit == "0" else dn for bit in state)) for state in p]
        # print(f"List of bistrings: {p}")

        P = [np.real((tq.daggx(state) @ i @ tq.daggx(i) @ state)[0][0]) for i in s]

        # Ghi dữ liệu bistring và probabilities vào file csv
        if filename_data:
            try:
                with open(filename_data, "w", newline="", encoding="utf-8") as f:
                    # Tạo một đối tượng writer
                    writer = csv.writer(f)

                    # Ghi dòng tiêu đề (header)
                    writer.writerow(["Bitstring", "Probability"])

                    # Ghi từng cặp dữ liệu (p[i], P[i]) vào file
                    for i in range(len(p)):
                        writer.writerow([p[i], P[i]])

                print(f"Successfully exported data file: {filename_data}")
            except Exception as e:
                print(f"Error: {e}")

        plt.figure(figsize=(max(4, numq * 4), 6))
        bars = plt.bar(p, P, color="steelblue", edgecolor="none", width=0.6, alpha=0.6)

        # Làm cho đồ thị “paper-style”
        plt.title("State Probabilities", fontsize=18, pad=15)
        plt.xlabel("Bitstrings", fontsize=14, labelpad=8)
        plt.ylabel("Probability", fontsize=14, labelpad=8)

        # Nhãn trục X nghiêng nhẹ (giống hình)
        plt.xticks(rotation=45, ha="right", fontsize=10, fontfamily="monospace")
        plt.yticks(fontsize=12)

        # Tắt khung trên và phải (giống figure “clean”)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # Điều chỉnh khoảng cách lề
        plt.margins(x=0.01)

        # Tăng khoảng trắng phía dưới
        plt.subplots_adjust(bottom=0.25)

        # Lưu ảnh với chất lượng cao
        plt.tight_layout()

        # THAY ĐỔI Ở ĐÂY
        if filename_all:
            plt.savefig(filename_all, dpi=400)
            plt.close()
        else:
            plt.show()

        # --- Gọi hàm vẽ top 10 và truyền filename vào ---
        self.plot_top_probabilities(P, p, num_top=10, filename=filename_top)


class FalqonOptimizerTensor:
    def __init__(self, h_cost=None, h_drive=None):
        self.h_cost = h_cost
        self.h_drive = h_drive

    def create_h_cost(self):
        if self.h_cost is None:
            linear, quadratic, offset = self.example_h_cost()
            self.h_cost = [linear, quadratic, offset]
        return self.h_cost

    def create_h_drive(self):
        if self.h_drive is None:
            self.h_drive = self.example_h_drive()
        return self.h_drive

    def example_h_cost(self, N=4):
        linear = {"[0]": 0.5, "[1]": 0.5, "[2]": 0.5, "[3]": 0.5}
        quadratic = {
            ("[2]", "[3]"): 0.5,
            ("[1]", "[2]"): 0.5,
            ("[0]", "[1]"): 0.5,
            ("[0]", "[3]"): 0.5,
        }
        offset = 0

        return linear, quadratic, offset

    def example_h_drive(self, N=4):
        h = {}
        for i in range(N):
            h[f"[{i}]"] = 1
        return h

    def hc_psi(self, psi):
        h = self.create_h_cost()
        linear, quadratic, offset = h[0], h[1], h[2]

        for key, value in linear.items():
            m = int(key[1:-1])
            psi[m] = tq.sigmaz() @ psi[m] * value

        for key, value in quadratic.items():
            m = int(key[0][1:-1])
            n = int(key[1][1:-1])

            psi[m] = tq.sigmaz() @ psi[m] * value
            psi[n] = tq.sigmaz() @ psi[n]

        return psi

    def hd_psi(self, psi):
        h = self.create_h_drive()

        for key, value in h.items():
            m = int(key[1:-1])
            psi[m] = tq.sigmax() @ psi[m] * value

        return psi

    def beta(self, state):
        offset = self.create_h_cost()[2]
        hc_psi = self.hc_psi(state)
        hd_psi = self.hd_psi(state)
        hc_psi_dag = []
        for i in hc_psi:
            hc_psi_dag.append(np.conjugate(i).T)

        hd_psi_dag = []
        for i in hd_psi:
            hd_psi_dag.append(np.conjugate(i).T)

        beta1 = 1
        beta2 = 1
        for i in range(len(hd_psi)):
            beta1 *= hd_psi_dag[i] @ hc_psi[i]
            beta2 *= hc_psi_dag[i] @ hd_psi[i]

        beta = beta1 - beta2

        # print(beta)

        return -1j * beta[0, 0]

    def quantum_state(self, t, state):
        beta_val = self.beta(state)
        Hd = self.create_h_drive()

        Ud = []
        for i in range(len(Hd)):
            Ud.append(expm(-1j * t * tq.sigmax() * beta_val))

        Hc = self.create_h_cost()
        linear, quadratic, offset = Hc[0], Hc[1], Hc[2]

        Uc = [np.identity(2)] * len(Hd)

        for key, value in linear.items():
            m = int(key[1:-1])
            Uc[m] = expm(-1j * value * tq.sigmaz())

        for key, value in quadratic.items():
            m = int(key[0][1:-1])
            n = int(key[1][1:-1])

            Uc[m] = expm(-1j * value * tq.sigmaz())
            Uc[n] = expm(-1j * tq.sigmaz())

        # Uc_offset= expm(-1j*offset)

        for i in range(len(Uc)):
            state[i] = Uc[i] @ state[i]

        for i in range(len(Ud)):
            state[i] = Ud[i] @ state[i]

        print("state=", state)

        return state

    def expectation_value(self, state):
        hc_psi = self.hc_psi(state)
        hc_psi_dag = []

        for i in hc_psi:
            hc_psi_dag.append(np.conjugate(i).T)

        exp_val = 1
        for i in range(len(hc_psi_dag)):
            exp_val *= np.array(hc_psi_dag)[i] @ np.array(state)[i]

        return exp_val[0, 0]

    def run(self, psi, times):
        exp_val = []
        psi_val = []
        for t in times:
            psi = self.quantum_state(t, psi)
            psi_val.append(psi)
            exp_val.append(np.real(self.expectation_value(psi)))
            # print(t, np.real(exp[0,0]))

        return exp_val, psi_val

    def plot(self, data, fn="enesVstimes.png"):
        # plot energy vs time

        plt.plot(data[0], data[1], label="enes vs times")
        plt.savefig(fn)

    def visualization(self, state, fn="histogram.png"):
        up = tq.bx(2, 0)
        dn = tq.bx(2, 1)

        state_full = state[0]
        for i in range(1, len(state)):
            state_full = tq.tensorx(state_full, state[i])

        dim = len(state_full)
        numq = int(np.log2(dim))

        # Generate the list p of binary strings for N qubits
        p = ["".join(bits) for bits in itertools.product("01", repeat=numq)]
        s = [
            tq.tensorx(*(up if bit == "0" else dn for bit in state_full))
            for state_full in p
        ]
        P = []

        for i in s:
            P.append(
                np.real((tq.daggx(state_full) @ i @ tq.daggx(i) @ state_full)[0][0])
            )

        # plot histogram
        plt.figure(figsize=(100, 100))
        plt.bar(p, P, color="skyblue", edgecolor="black")

        # Add title and labels
        plt.title("Histogram")
        plt.xlabel("Classical bits")
        plt.ylabel("Frequencies")

        # Rotate x labels if needed
        plt.xticks(rotation=90)
        plt.savefig(fn)
