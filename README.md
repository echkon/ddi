# ddi

## Data and Scripts for:  
**Feedback-Based Quantum Control for Safe and Synergistic Drug Combination Design**

**Authors:**  
Mai Nguyen Phuong Nhi¹, Lan Nguyen Tran², Le Bin Ho² ³  

¹ University of Science, Vietnam National University, Ho Chi Minh City 700000, Vietnam  
² Vietnam National University, Ho Chi Minh City 700000, Vietnam  
³ Frontier Research Institute for Interdisciplinary Sciences and  
   Department of Applied Physics, Graduate School of Engineering,  
   Tohoku University, Sendai 980-8578, Japan  

---

## Overview

This repository contains the numerical data and plotting scripts associated with the paper:

**Feedback-Based Quantum Control for Safe and Synergistic Drug Combination Design**  
Mai Nguyen Phuong Nhi, Lan Nguyen Tran, and Le Bin Ho  
arXiv:2601.18082 (2026)

Drug combination therapies are widely used in modern medicine, but identifying combinations that are both effective and safe remains a challenging optimization problem due to complex drug–drug interactions (DDIs). This work proposes a quantum-inspired optimization framework in which known synergistic and harmful interactions are encoded into a cost function. A feedback-based quantum algorithm is employed to efficiently search for drug subsets that maximize therapeutic synergy while avoiding adverse interactions. The numerical results demonstrate that the proposed approach can robustly identify high-quality drug combinations using realistic interaction data.

---

## File Structure

### fig1/
- Numerical data and plotting scripts for Fig. 1

### fig2_fig3/
- Numerical data and plotting scripts for Figs. 2 and 3

### fig4/
- Numerical data and plotting scripts for Fig. 4

### fig5/
- Numerical data and plotting scripts for Fig. 5

### fig6/
- Numerical data and plotting scripts for Fig. 6

### fig7/
- Numerical data and plotting scripts for Fig. 7

Each directory contains the raw data files and the corresponding Python scripts used to reproduce the figures in the paper.

---

## Data Description

**Drug–Drug Interaction Encoding:**  
Synergistic and harmful drug interactions are encoded as weighted terms in an Ising-type cost function, allowing the drug selection problem to be formulated as a combinatorial optimization task.

**Feedback-Based Optimization:**  
A feedback-based quantum algorithm is used to iteratively update control parameters based on measurement outcomes, enabling efficient convergence without explicit gradient evaluation.

**Numerical Evaluation:**  
Simulation results based on realistic DDI datasets demonstrate that the method successfully balances safety and synergy, outperforming naive or unstructured selection strategies.

---

## Requirements

To reproduce the numerical results and figures, the following software is required:

- Python ≥ 3.9  
- NumPy  
- Matplotlib  
- Pandas  

(Optional)  
- Seaborn for enhanced visualization  

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/echkon/ddi.git
   cd ddi
