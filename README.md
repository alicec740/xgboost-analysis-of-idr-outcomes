# XGBoost Analysis of No Surprises Act IDR Outcomes
An XGBoost model identified unexpected predictors of No Surprises Act arbitration outcomes. Model interpretation revealed that Practice/Facility Size was strongly associated with default decisions, motivating a refined analysis that excluded defaults. The refined model revealed Provider Email Domain as a dominant predictor, which further investigation suggested may proxy for organizational identity or dispute-management structure.

This project uses XGBoost to examine patterns associated with arbitration outcomes in the federal Independent Dispute Resolution (IDR) process established under the No Surprises Act. Rather than treating model performance as the endpoint, the analysis uses feature importance, SHAP values, descriptive analysis, and cross-validation to investigate unexpected predictive signals and their substantive meaning.

The analysis proceeds in two stages. The initial model identifies Practice/Facility Size as the strongest feature. Further investigation shows that missing facility-size information is strongly associated with default decisions, motivating a refined analysis that excludes default disputes. In the refined model, Provider Email Domain emerges as the strongest feature. Additional analysis shows substantial differences in outcomes across provider email domains and suggests that the feature may act as a proxy for organizational identity or dispute-management structure.


### Key findings

- Practice/Facility Size was the highest-gain feature in the initial model.
- 99.32% of disputes with unknown Practice/Facility Size were default decisions.
- After excluding default decisions, Practice/Facility Size fell substantially in feature importance.
- Provider Email Domain became the highest-gain feature in the refined model.
- Domain-level SHAP effects corresponded to substantial differences in observed arbitration outcomes.
- These domain-level effects remained directionally consistent across four validation folds.
- The results suggest that provider email domains may encode organizational or dispute-management characteristics not directly represented by other model features.

## Project Overview

## Analytical Approach

## Key Findings

## Data & Methodology

## Limitations




