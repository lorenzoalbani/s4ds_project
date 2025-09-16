# Uncertainty as a Fairness Measure - Replication Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R Version](https://img.shields.io/badge/R-%3E%3D%204.0-blue)](https://www.r-project.org/)

**R-based replication** of the experiments from the paper "Uncertainty as a Fairness Measure", exploring how uncertainty quantification can provide deeper insights into fairness and bias detection in machine learning models beyond traditional point-based metrics.

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Types of Uncertainty](#types-of-uncertainty)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

## 🎯 Overview

This project is a **replication study** of the original paper "Uncertainty as a Fairness Measure", implemented in R. The original work demonstrated that traditional fairness evaluation in machine learning relies heavily on point-based metrics, which may overlook important sources of bias hidden in model uncertainty.

### Original Paper's Contributions

- Introduction of uncertainty-based fairness metrics (epistemic, aleatoric, predictive)
- Demonstration that models can appear fair using conventional metrics while exhibiting unfairness in uncertainty measures
- Implementation using Bayesian Neural Networks with Monte Carlo sampling
- Evaluation on synthetic and real-world datasets

### Our Replication

We replicated the experiments on the **three synthetic datasets** described in the original paper, implementing the methodology in R using simpler BNN architectures. Our results show **different patterns** from the original findings, highlighting implementation sensitivity and methodological challenges in uncertainty quantification.

## 💡 Motivation

Machine learning fairness faces several critical challenges:

1. **Fairness Gerrymandering**: Models may appear fair overall but discriminate against specific subgroups
2. **Imbalanced Distributions**: Minority groups are often underrepresented in training data
3. **Real-world Complexities**: Noisy data, missing values, and labeling errors introduce additional bias sources

Traditional point-based metrics alone are insufficient to capture these nuanced fairness issues.

## 🔬 Types of Uncertainty

Our framework considers three complementary types of uncertainty:

### 🎲 Aleatoric Uncertainty (Ua)
- **Definition**: Reflects irreducible noise inherent in the data
- **Measurement**: Computed over different class predictions
- **Source**: Fundamental randomness that cannot be reduced with more data

### 🧠 Epistemic Uncertainty (Ue)
- **Definition**: Represents model's lack of knowledge about the input
- **Measurement**: Computed across different model instances
- **Source**: Insufficient training data or model complexity limitations

### 📊 Predictive Uncertainty (Up)
- **Definition**: Combined measure of both aleatoric and epistemic uncertainty
- **Measurement**: Total uncertainty in model predictions
- **Formula**: Up = Ua + Ue

## 🛠 Methodology

### Uncertainty Quantification

Each model produces a probability P_m of predicting the positive class, where uncertainty is quantified through:

- **Monte Carlo Sampling**: Multiple forward passes with different weight samples
- **Bayesian Neural Networks**: Treat weights as distributions rather than point estimates
- **Bootstrap Sampling**: Generate diverse training datasets for aleatoric uncertainty estimation

## 📖 Original Paper Reference

**"Uncertainty as a Fairness Measure"** - This replication study is based on the original paper which introduced the concept of using predictive, aleatoric, and epistemic uncertainty to reveal hidden biases in machine learning models beyond traditional point-based fairness metrics.

## 👥 Replication Team

- **Albani Lorenzo**
- **Ascione Luigi**
- **Del Rosso Filippo**

*Statistics for Data Science - A.A. 2024/2025*  
*Replication Study Project*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original paper authors for the theoretical framework
- R community for the `bnns` package and Stan integration

## 📊 Replication Results

### Comparison: Our R Implementation vs Original Paper

| Dataset | Metric | Group | Our Results | Original Paper | Match |
|---------|--------|--------|-------------|----------------|--------|
| **SD1** | Macc | G0/G1 | 0.74/0.93 | 0.95/0.95 | ❌ |
| | Fepis | Ratio | 1.90 | 1.01 | ❌ |
| | Falea | Ratio | 0.94 | 4.68 | ❌ |
| **SD2** | Macc | G0/G1 | 0.90/1.00 | 0.95/0.95 | ❌ |
| | Fepis | Ratio | 4.25 | 2.75 | ⚠️ |
| | Falea | Ratio | 0.88 | 0.87 | ✅ |
| **SD3** | Macc | G0/G1 | 0.80/0.90 | 0.74/0.93 | ⚠️ |

### Replication Challenges Identified

1. **Implementation Gap**: R `bnns` package vs original Python specialized libraries
2. **Architecture Sensitivity**: Simple BNNs vs complex custom architectures in original
3. **Dataset Coupling**: Synthetic datasets were tightly designed for specific model behaviors
4. **Library Maturity**: Python ecosystem more mature for advanced BNN implementations

### Key Findings from Our Replication

- **Partial Success**: We captured some uncertainty patterns but not the exact magnitudes
- **Method Validation**: The general approach works, but specific results depend heavily on implementation details  
- **Alternative Methods**: Random Forest ensembles achieved high accuracy but poor uncertainty discrimination
- **Future Work**: More sophisticated BNN architectures needed to fully replicate original findings
