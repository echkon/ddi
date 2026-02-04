# ddi

## Data and Scripts for:  
**Feedback-Based Quantum Control for Safe and Synergistic Drug Combination Design**

**Authors:**  
Mai Nguyen Phuong Nhi¹², Lan Nguyen Tran¹², Le Bin Ho³⁴  

¹ University of Science, Vietnam National University, Ho Chi Minh City 700000, Vietnam  
² Vietnam National University, Ho Chi Minh City 700000, Vietnam  
³ Frontier Research Institute for Interdisciplinary Sciences, Tohoku University, Sendai 980-8578, Japan  
⁴ Department of Applied Physics, Graduate School of Engineering, Tohoku University, Sendai 980-8579, Japan

---

## Overview

This repository contains the numerical data and plotting scripts associated with the paper:

**Feedback-Based Quantum Control for Safe and Synergistic Drug Combination Design**  
Mai Nguyen Phuong Nhi, Lan Nguyen Tran, and Le Bin Ho  
arXiv:2601.18082 (2026)

Drug combination therapies are widely used in modern medicine, but identifying combinations that are both effective and safe remains a challenging optimization problem due to complex drug–drug interactions (DDIs). This work proposes a quantum-inspired optimization framework in which known synergistic and harmful interactions are encoded into a cost function. A feedback-based quantum algorithm is employed to efficiently search for drug subsets that maximize therapeutic synergy while avoiding adverse interactions. The numerical results demonstrate that the proposed approach can robustly identify high-quality drug combinations using realistic interaction data.

---

## File Structure

### algorithm/

Core algorithm implementations.

- `falqon.py`  
  Standard FALQON optimizer.

- `falqon_compact.py`  
  Compact version of FALQON incorporating Imaginary Time Evolution (ITE).


### data/

Input datasets for drug–drug interaction (DDI) analysis.

- `ddi_a.csv`  
  Main DDI dataset used in 6-drug simulations.

- `ddi_r.csv`  
  Implicit DDI dataset for 6-drug network plotting.

- `ddi_covid.csv`  
  COVID-related DDI dataset.

- `node_mapping_6_drugs.csv`  
  Mapping between abstract node labels (A–F) of ddi_r.csv to real drug names.


### fig1/

Scripts for generating Figure 1.

- `graph_6_drugs.py`  
  Visualization of the 6-drug interaction network.


### fig2_fig3/

Numerical simulations and plots for Figures 2 and 3 (6-drug MSS problem).

- `mss_6d.py`  
  Run FALQON for the 6-drug MSS problem.

- `mss_ite_6d.py`  
  MSS problem using FALQON + ITE.

- `plot_mss_6d.py`  
  Plot results of standard FALQON.

- `plot_mss_ite_6d.py`  
  Plot results of FALQON + ITE.

- `plot_combined.py`  
  Comparison plot for standard FALQON and FALQON + ITE.


### fig4/

Numerical simulations and plotting scripts for Figure 4 (6-drug SCO problem). 

- `sco_k_ite.py`  
  SCO problem using FALQON + ITE for different size k.

- `plot_sco_k_ite.py`  
  Visualization of Fig. 4 results.


### fig5/

Scripts for generating Figure 5.

- `graph_covid.py`  
  Network visualization for the COVID-related DDI dataset.


### fig6/

Numerical simulations and plots for Figure 6 (COVID MSS problem).

- `mss_ite_covid.py`  
  MSS simulation on COVID dataset using FALQON + ITE.

- `plot_mss_ite_covid.py`  
  Plotting results for Fig. 6.


### fig7/

Numerical simulations and plots for Figure 7 (COVID SCO problem).

- `sco_ite_covid.py`  
  SCO problem on COVID dataset using FALQON + ITE.

- `plot_sco_ite_covid.py`  
  Visualization of Fig. 7 results.


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
