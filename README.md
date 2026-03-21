# Active Learning Framework for Surrogate Modelling of Centrifugal Pump Performance

This repository presents a hybrid framework that integrates Computational Fluid Dynamics (CFD), Machine Learning (ML), and Active Learning (AL) strategies to efficiently construct surrogate models for centrifugal pump performance prediction. The objective is to reduce the computational cost of CFD simulations while maintaining high predictive accuracy through structured sampling strategies.

{author} Daniel Morantes-Morales
{mail} d.morantes@uniandes.edu.co
---

## Methodology Overview

The framework combines:

### 1. Synthetic Data Generation
- Fluid properties (Newtonian and non-Newtonian) generated from literature-based ranges  
- Rheological models including Power Law and Cross models  

### 2. CFD-Based Dataset Construction
- High-fidelity CFD simulations used to generate pump performance data  
- External simulation pipeline (not fully distributed)

### 3. Surrogate Modeling
- XGBoost  
- Gaussian Process Regression (GPR)  

### 4. Active Learning Strategies

Eight ML–AL configurations were evaluated:

#### Baseline (Random Sampling via LHS)
- XGBoost + Random  
- GPR + Random  

#### Greedy Sampling (Input Space)
- XGBoost + Greedy Inputs  
- GPR + Greedy Inputs  

#### Greedy Sampling (Output Space)
- XGBoost + Greedy Outputs  

#### Hybrid Sampling (Inputs + Outputs)
- XGBoost + Greedy Inputs/Outputs  
- GPR + Greedy Inputs/Outputs  

#### Uncertainty-Driven Sampling (Best Performing)
- GPR + Optimization over the predictive uncertainty field  

---

## Notebooks

### Initial Workflow

- **1stDataGen.ipynb**  
  Synthetic fluid generation, preprocessing, BEP normalization, and baseline model training (XGBoost and GPR).  
  Includes reconstruction of models to extract predictions and extended evaluation metrics.

---

### Active Learning Workflows

- **2ndDataGen_BaselineLHS.ipynb**  
  Baseline data augmentation using Latin Hypercube Sampling (random selection).

- **2ndDataGen_Gsi.ipynb**  
  Greedy sampling based on input-space diversity.

- **2ndDataGen_Gso.ipynb**  
  Greedy sampling based on output-space criteria (XGBoost only).

- **2ndDataGen_iGsXgBoost.ipynb**  
  Hybrid sampling combining input and output criteria using XGBoost.

- **2ndDataGen_iGsGPR.ipynb**  
  Hybrid sampling using GPR, including dynamic alpha scheduling.

- **2ndDataGen_GPvariational.ipynb**
  Uncertainty-driven active learning using GPR with optimization over the predictive uncertainty field.  
  **This notebook corresponds to the best-performing strategy in the study.**

---

### Final Analysis

- **FinalResults.ipynb**  
  Consolidates and visualizes results across all ML–AL configurations.  
  This notebook loads precomputed artifacts and performs comparative analysis.
  Includes:
  - performance comparison across all paths  
  - MSE evolution vs number of fluids  
  - uncertainty evolution analysis  
  - identification of the best-performing strategy  
  - generation of final manuscript figures  

---

## Evaluation Metrics

The following metrics are used:

- Mean Squared Error (MSE): tracked at every iterations of the ML+AL paths
- Coefficient of Determination (R²): tracked at every iterations of the ML+AL paths
- Mean Absolute Error (MAE): capture for the initial and final model (snapshots) for every combination
- Explained Variance (EV): capture for the initial and final model (snapshots) for every combination
- Kling–Gupta Efficiency (KGE): computed for the initial models and the final best performed models 
- Percent Bias (PBIAS): computed for the initial models and the final best performed models   

---

## Repository Structure

.
├── notebooks/
├── src/
│   ├── Functions_PumpAI.py
│   └── train_best_gpr_variational.py
├── data/
├── artifacts/
├── artifacts_bestPath/
├── requirements.txt
├── Makefile
└── README.md

---

## Stored Artifacts and Model Assets

### `artifacts/`
Contains intermediate objects required across notebooks:
- scalers  
- intermediate datasets  
- trained models from different paths  
- auxiliary results  

These are used to maintain workflow continuity between notebooks.

---

### `artifacts_bestPath/` (Final Model Assets)

This directory contains the complete set of assets corresponding to the **best-performing model** (GPR with uncertainty-driven active learning).

It serves as the **model asset repository**, enabling direct reuse and reproduction without retraining.

Included objects:

- trained GPR model  
- feature and target scalers  
- evaluation metrics  
- predictive uncertainty outputs  
- train/test predictions  
- feature and target column definitions  

These assets allow:

- direct inference using the trained surrogate model  
- reproducibility of reported results  
- reuse of the model in external workflows or applications  

> This folder represents the final deliverable of the framework and can be treated as a deployable model asset package.

---

## Best-Path Verification Script

A standalone Python script is included:

src/train_best_gpr_variational.py

This script is intended to:

- verify that the repository environment is correctly configured  
- reproduce the final GPR training logic outside the notebook interface  
- confirm that the best-performing model can be retrained  

Notes:
- This script does **not replace the notebooks**  
- Final artifacts are already generated and stored  
- Local file paths may need to be updated before execution  

---

## Makefile Commands

The repository includes a `Makefile` to simplify common tasks:

```bash
make install
make notebooks
make best-model
make clean
make check-artifacts

	•	make install → install dependencies
	•	make notebooks → launch Jupyter
	•	make best-model → run verification script
	•	make clean → remove cache files
	•	make check-artifacts → verify repo structure

⸻

Installation

pip install -r requirements.txt

⸻

Usage

jupyter notebook

⸻
Reproducibility_Notes
	•	Python version: 3.11–3.12
	•	XGBoost version: 1.7.6
	•	GPR implemented using scikit-learn
	•	Hyperparameter tuning via RandomizedSearchCV
	•	Final models reconstructed using fixed hyperparameters

Some components are not fully reproducible due to:
	•	external CFD simulations
	•	non-distributed full datasets

However, the repository provides:
	•	full methodology
	•	partial datasets
	•	complete implementation
	•	final model reconstruction

⸻

Key Contributions and Findings
	•	Active Learning significantly reduces CFD simulation requirements
	•	Structured sampling improves surrogate model generalization
	•	Hybrid strategies improve early-stage exploration
	•	Uncertainty-driven GPR provides the best performance
	•	Predictive uncertainty is an effective sampling criterion for complex fluid systems

⸻

Data Availability

The complete CFD dataset is not publicly distributed.
A subset of data and full implementation are provided for partial reproducibility.

⸻

Contact

For questions or collaboration inquiries, please contact the corresponding author.
{author}: Daniel Morantes-Morales
{mail}: d.morantes@uniandes.edu.co

---