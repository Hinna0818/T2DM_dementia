# T2DM-Associated Dementia Risk Analysis

This repository contains the analysis code and supplementary materials for the study:  
**"Polygenic and Polysocial Risk Scores in Post-Type 2 Diabetes Dementia: Risk Stratification and Predictive Modeling in the UK Biobank Cohort"**  


## Overview

This study investigates how genetic susceptibility (measured by **Polygenic Risk Scores, PRS**) and social disadvantage (measured by **Polysocial Risk Scores, PsRS**) independently and jointly influence dementia risk among individuals with T2DM.

## Main Analyses

- **Cox Proportional Hazards Models**  
  Evaluated the associations of PsRS and PRS with all-cause dementia (ACD), Alzheimer’s disease (AD), and vascular dementia (VD).

- **Stratified and Interaction Analyses**  
  Explored how PsRS impacts dementia risk across different genetic risk groups.

- **Machine Learning Prediction Models**  
  Applied multiple algorithms (e.g., XGBoost, CatBoost) to predict ACD risk in medium-to-high PRS populations.  
  Model interpretability was enhanced with **SHAP** and **ALE** analyses.


## Tools & Packages

- **R (v4.4.2)** — Cox models, statistical analysis
- **Python (v3.9)** — Machine learning, model interpretation
- Key libraries: `rms`, `survival`, `xgboost`, `shap`, `aplot`, `lightgbm`, `catboost`

## Contact

For questions or collaboration inquiries, please contact:  
**Hinna0818** (GitHub)  
or reach out via email (nanh302311@gmail.com).


