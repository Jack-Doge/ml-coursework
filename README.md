# 📊 Taiwan Stock Prediction Using Ridge Regression & XGBoost

This repository contains the source code and result materials for a two-part final project focused on predicting Taiwan stock returns using both linear and non-linear models. The project involves portfolio construction, model evaluation, and feature importance analysis based on fundamental and market factors.

---

## 📁 Project Structure

- `part1/`: Code and materials for **Part I** – model construction, tuning, and baseline portfolio performance.
- `part2/`: Code and materials for **Part II** – result analysis, model interpretation, and time-evolving feature importance visualization.
- `report/`: PDF reports for both parts (not containing code).

---

## 📌 Part I Summary

- **Data**: Monthly data from **2009/01 to 2025/02** for the top 500 Taiwan-listed companies.
- **Features**: Includes firm-specific and market-level predictors, such as turnover, P/B, dividend yield, market risk premium, value, momentum, profitability, etc.
- **Models**: Ridge Regression and XGBoost, tuned with hyperparameters using walk-forward validation.
- **Output**:
  - L/S portfolio returns.
  - Mean and mean-volatility adjusted performance metrics.
  - Top-3 features and partial dependence plots.

---

## 📌 Part II Summary

- **Data & Methodology**: Continuation of Part I with expanded analysis.
- **Key Insights**:
  - Comparison to benchmarks and explanations for out/underperformance.
  - Interpretation of top features during strong vs. weak periods.
  - Visualization of **feature importance evolution over time** using SHAP/permutation-based heatmaps.

---

## 🔧 Technologies Used

- Python 3.11
- scikit-learn
- XGBoost
- SHAP
- matplotlib / seaborn
- pandas / NumPy

---

## Contact

Author: **Jack Huang**  
Email: `113352025@g.nccu.edu.tw`  
Graduate Student, National Chengchi University, Department of Money & Banking

---

