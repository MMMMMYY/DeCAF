# DeCAF

**DeCAF** is a client-side defense framework designed to detect and mitigate fairness attacks in federated learning (FL).  
It monitors model output probabilities across sensitive attributes and communication rounds to compute **fairness impact scores**, which are then verified and used for adaptive aggregation.

---

## Files

- **`adult_script.py`** – Example script for running DeCAF on the **Adult** dataset.  
  Includes dataset preprocessing, FL setup, attack injection, and mitigation evaluation.

- **`attacks.py`** – Implements baseline attacks and related functions. 

- **`concealed_attack.py`** – Defines the **Concealed Fairness Attack (CFA)**, including data poisoning based and model poisoning based.

- **`datasets_utils.py`** – Provides dataset loading and preprocessing utilities for tabular datasets such as **Adult**, **UTKFace**, and **Drug Consumption**.  
  Supports sensitive attribute partitioning (e.g., gender, race) and label encoding.

- **`evaluation.py`** – Computes performance and fairness metrics:
  - Accuracy and F1-score 
  - Fairness metrics such as **Statistical Parity Difference (SPD)** and **Equalized Odds Difference (EOD)**  

- **`federated_framework.py`** – Implements the core federated learning loop:
  - Local training  
  - Model aggregation  
  - Attack simulation and mitigation integration  

- **`mitigation.py`** – Core implementation of the **DeCAF** defense:
  - Computes fairness impact scores based on inter-attribute and inter-round probability shifts  
  - Performs clustering-based verification  
  - Adjusts client aggregation weights adaptively  

- **`utils.py`** – General utility functions for model initialization, logging, and visualization.

---

## Usage

See example run in the **`adult_script.py`**.
