# Improving Load Forecasting with XAI and Expert-Guided Prompting

Implementation and experimental resources for improving electricity load forecasting using **Explainable AI (XAI), feature selection, domain knowledge, and LLM-based prompting**.

## Overview

Electricity load forecasting depends on many temporal, environmental, and contextual variables.

Using all available features does not necessarily improve forecasting performance, and identifying meaningful variables often requires both data-driven analysis and domain expertise.

This study combines:

- XAI-based feature importance analysis
- expert knowledge
- LLM-assisted reasoning
- forecasting model evaluation

to identify informative features for electricity load forecasting.

## Approach

The overall workflow is:

1. Train a baseline load forecasting model
2. Analyze feature contributions using XAI methods
3. Identify candidate important variables
4. Combine model-derived evidence with expert knowledge
5. Use prompting to support feature selection
6. Retrain and evaluate forecasting models using the selected features

XAI methods such as **LIME and SHAP** are used to analyze feature contributions.

## Repository Contents

This repository contains code for:

- Load forecasting experiments
- Data preprocessing
- Feature importance analysis
- XAI-based feature selection
- Model training and evaluation
- Experimental comparison of feature subsets

## Environment

Main dependencies:

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- SHAP
- LIME

## Publication

**Improving Load Forecasting with Feature Selection via XAI and Expert-Guided Prompting**

*IEEE Access*, 2025.

## Citation

If you use this code or methodology, please cite the corresponding paper.
