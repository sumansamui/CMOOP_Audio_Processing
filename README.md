# Surrogate-Assisted Memetic Constrained Multi-Objective Optimization for TinyML Audio Classification

This repository contains the code, experimental setup, and supplementary materials for the paper:

**"Memetic Constrained Evolutionary Optimization of Deep Learning Models for Audio Classification on Resource-Constrained Devices"**

Submitted to the **IEEE Transactions on Evolutionary Computation** (Special Issue on Recent Trends in Evolutionary Constrained Optimization).

# Abstract
Deploying deep learning models for audio classification tasks such as keyword spotting and bird call identification on resource-constrained embedded devices requires balancing model accuracy, memory usage, and false positive rate. This work proposes a hybrid evolutionary optimization framework combining global search with surrogate-guided local refinement to efficiently address this multi-objective problem under constraints. By leveraging predictive surrogate models, adaptive penalty mechanisms, and selective local search, the proposed method substantially reduces the computational burden. Experimental results show that the approach consistently discovers high-quality models satisfying multiple deployment objectives while being significantly more time-efficient than standard evolutionary baselines.

## 📝 Overview

The goal of this project is to optimize deep neural network architectures for audio classification tasks such as **Keyword Spotting (KWS)** and **Bird Call Identification** on **TinyML devices**, under multiple conflicting objectives and constraints, including:

- **Accuracy** (maximize)
- **Model size** (minimize)
- **False Positive Rate (FPR)** (minimize)
- **Hardware feasibility** constraints (e.g., memory)

We propose a **Surrogate-Assisted NSGA-II (SA-NSGA-II)** framework with:

- Gaussian Process (Kriging) based surrogate modeling
- Constraint-aware scoring and filtering
- Lamarckian surrogate-guided local search
- Variance-based uncertainty sampling
- Smart population initialization via multi-stage CMOO

## 📊 Datasets

We used the following publicly available datasets:

- **[Google Speech Commands v2 (GSC)](https://www.tensorflow.org/datasets/catalog/speech_commands)**  
  For training and evaluating keyword spotting models on TinyML hardware.

- **[BirdCLEF 2021](https://www.kaggle.com/competitions/birdclef-2021)**  
  For bird species classification from real-world environmental recordings.



## 🚀 Key Features of This Repository

- ✅ **Constraint-Aware Surrogate Scoring**  
  Penalty-based scalarization of predicted objective and constraint violations.

- 🧠 **Surrogate-Guided Lamarckian Local Search**  
  Local refinement using surrogate landscapes to improve exploitation.

- 📈 **Variance-Based Exploration**  
  High-uncertainty samples are selectively evaluated to improve surrogate fidelity.

- ♟️ **Sub-Problem Decomposition-Based Initialization**  
  High-quality initial population generation by solving bi-objective CMOO sub-problems.

- 📐 **Comprehensive Evaluation Metrics**  
  Uses **Hypervolume (HV)**, **Inverted Generational Distance (IGD)**, and **Spread** to assess solution quality and diversity.

 ## Environment Setup Instructions

# 1. Create a new Conda environment (Python 3.12 recommended):
     conda create --name cmoo_audio python=3.12
     conda activate cmoo_audio

# 2. Install dependencies using pip:

     pip install -r requirements.txt


## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.