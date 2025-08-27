# 📈 Taiwan Stock Return Prediction

This project focuses on predicting Taiwan stock returns using various financial factors and machine learning models. It includes data preparation, model training (Ridge Regression, XGBoost), and result analysis.

---

## 📁 File Structure

- `config.py`: Defines global configurations, such as feature names and target variables.
- `data_preparation.py`: Scripts for initial data cleaning, merging, and transformation from raw Excel files.
- `data_processing.py`: Processes the raw data into a clean, usable format for modeling.
- `ridge.py`: Implements a Ridge Regression model with a walk-forward validation approach to predict stock returns.
- `xg.py`: Implements an XGBoost model using a similar walk-forward validation approach.
- `summary.py`: Analyzes and visualizes the results from the Ridge and XGBoost models, comparing their performance.
- `HW[1-3].py`: Scripts for specific homework assignments, exploring concepts like influential observations, Principal Component Regression (PCR), and Arbitrage Pricing Theory (APT).
- `data/`: Contains all data, including raw, processed, and result files.
- `summary/`: Contains saved plots from the analysis in `summary.py`.

---

## 🚀 How to Run

1.  **Data Preparation**: The initial data preparation from raw `.xlsx` files is done in `data_preparation.py`. The main modeling scripts consume data that is already processed.

2.  **Run Models**: Execute the regression and machine learning models. These scripts will generate result files in the `data/` directory.
    ```bash
    python ridge.py
    python xg.py
    ```

3.  **Analyze Results**: Run the summary script to generate analysis plots and comparisons.
    ```bash
    python summary.py
    ```

4.  **Homework Scripts**: The `HW` scripts can be run individually to see the results of specific analyses.
    ```bash
    python HW1.py
    python HW2.py
    python HW3.py
    ```

---

## 🔧 Technologies Used

- Python 3.9+
- pandas
- numpy
- scikit-learn
- XGBoost
- statsmodels
- matplotlib

---

## 👤 Author

- **Jack Huang**
- `113352025@g.nccu.edu.tw`