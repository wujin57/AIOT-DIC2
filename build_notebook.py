import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True)
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip().splitlines(keepends=True)
    })

# Title
add_md("""# 50 Startups Profit Prediction — Machine Learning Solution
## End-to-End CRISP-DM Workflow using scikit-learn
**Author:** Antigravity Data Science Assistant  
**Dataset:** 50 Startups (`R&D Spend`, `Administration`, `Marketing Spend`, `State`, `Profit`)  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)
""")

# 1. Business Understanding
add_md("""---
# 1. Business Understanding

### 1.1 Business Problem & Context
Venture capital investors, angel networks, and corporate incubators face critical resource-allocation challenges when backing early-to-growth-stage startups. Startups operate with limited cash runways. Misallocating capital across Research & Development (R&D), General & Administrative (G&A) overhead, or commercial marketing can result in insolvency before achieving market viability. Stakeholders need data-driven intelligence to forecast potential profitability and benchmark expenditure structures.

### 1.2 Project Objectives
Develop and validate a supervised regression model capable of predicting a startup's **Profit ($)** based on its operational expenditures and state of incorporation.

### 1.3 Machine Learning Task Formulation
- **Task Type:** Supervised Regression (continuous quantitative target).
- **Target Variable ($y$):** `Profit` (USD)
- **Predictor Variables ($X$):**
  - `R&D Spend` (USD, continuous)
  - `Administration` (USD, continuous)
  - `Marketing Spend` (USD, continuous)
  - `State` (Nominal categorical: *New York*, *California*, *Florida*)

### 1.4 Appropriateness of Multiple Linear Regression
1. **Direct Economic Interpretability:** Coefficients provide transparent marginal returns ($\Delta \text{Profit}$ per $\$1.00$ spend).
2. **Low Sample Complexity:** With only $N=50$ observations, complex non-parametric models (e.g., deep neural networks or gradient boosting) risk extreme sample variance and overfitting. Ordinary Least Squares (OLS) offers high sample efficiency.
3. **No Multicollinearity Pitfalls with Regular Encoding:** Categorical states can be cleanly integrated via one-hot encoding with dummy-variable trap handling.
""")

# Setup
add_code("""# Required Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Aesthetics & Config
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
""")

# 2. Data Understanding
add_md("""---
# 2. Data Understanding

In this phase, we load the raw dataset directly from the remote repository, inspect schema dimensions, verify data types, check for missing values, analyze target distribution, and visualize feature correlations.
""")

add_code("""# Load dataset directly from URL
DATASET_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/refs/heads/master/datasets/50_Startups.csv"
df = pd.read_csv(DATASET_URL)

print("Dataset Shape:", df.shape)
print("\\nFirst 5 Rows:")
display(df.head())
""")

add_code("""# Dataset Schema and Summary Statistics
print("--- Data Types & Non-Null Values ---")
print(df.info())

print("\\n--- Missing Values Check ---")
print(df.isnull().sum())

print("\\n--- Unique Categorical Values for 'State' ---")
print(df['State'].value_counts())

print("\\n--- Summary Statistics ---")
display(df.describe().round(2))
""")

add_code("""# Target Variable (Profit) Statistical Distribution
profit = df['Profit']
print(f"Mean Profit:     ${profit.mean():,.2f}")
print(f"Median Profit:   ${profit.median():,.2f}")
print(f"Std Deviation:   ${profit.std():,.2f}")
print(f"Min Profit:      ${profit.min():,.2f}")
print(f"Max Profit:      ${profit.max():,.2f}")
print(f"Skewness:        {profit.skew():.4f}")

# Visualizing Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.histplot(profit, kde=True, ax=axes[0], color="#1f77b4", bins=10)
axes[0].set_title("Distribution of Profit (Histogram + KDE)", fontweight="bold")
axes[0].set_xlabel("Profit ($)")

sns.boxplot(x=profit, ax=axes[1], color="#aec7e8", flierprops={"markerfacecolor":"#d62728", "markersize":8})
axes[1].set_title("Profit Boxplot (Outlier Check)", fontweight="bold")
axes[1].set_xlabel("Profit ($)")
plt.tight_layout()
plt.savefig("figures/eda_profit_distribution.png", dpi=300)
plt.show()
""")

add_code("""# Scatter Plots: Numerical Features vs Profit
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
num_features = ["R&D Spend", "Administration", "Marketing Spend"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, col in enumerate(num_features):
    sns.regplot(
        data=df, x=col, y="Profit", ax=axes[i],
        scatter_kws={"alpha": 0.75, "color": colors[i], "s": 45},
        line_kws={"color": "#d62728", "linewidth": 1.8}
    )
    axes[i].set_title(f"{col} vs Profit", fontweight="bold")
    axes[i].set_xlabel(f"{col} ($)")
    axes[i].set_ylabel("Profit ($)")

plt.tight_layout()
plt.savefig("figures/eda_scatter_plots.png", dpi=300)
plt.show()
""")

add_code("""# Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues", square=True, linewidths=0.8, vmin=-0.1, vmax=1.0)
plt.title("Correlation Matrix of Numerical Features", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("figures/eda_correlation_heatmap.png", dpi=300)
plt.show()
""")

add_md("""### Key Findings from Data Understanding:
1. **Target Behavior:** Profit is approximately normally distributed (mean $\\approx \\$112,012$, median $\\approx \\$107,978$) with minor skewness ($0.023$) and one lower outlier ($index=49$, Profit $\\approx \\$14,681$).
2. **Strongest Predictor:** `R&D Spend` shows an almost perfect linear relationship with `Profit` ($r = 0.973$).
3. **Secondary Predictor:** `Marketing Spend` has a moderately strong positive relationship ($r = 0.748$), though it also shares collinearity with `R&D Spend` ($r = 0.724$).
4. **Weak Predictor:** `Administration` displays very weak correlation with `Profit` ($r = 0.201$), suggesting overhead expenses do not scale linearly with commercial success.
5. **No Missing Data:** The dataset is 100% complete across all 50 records.
""")

# 3. Data Preparation
add_md("""---
# 3. Data Preparation

Data Preparation establishes a reproducible, leakage-free pipeline:
1. **Feature-Target Separation:** Isolate $X$ (predictors) and $y$ (target).
2. **Train-Test Split:** Perform an 80/20 partition using `random_state=42`.
3. **Encoding Categorical Features:** Encode `State` using `OneHotEncoder(drop='first')` to prevent the **dummy-variable trap** (perfect multicollinearity in linear models).
4. **ColumnTransformer & Pipeline:** Package transformation and modeling steps together so that encoders are fitted strictly on the training partition.
""")

add_code("""# 3.1 Feature Matrix & Target Vector
X = df.drop(columns=["Profit"])
y = df["Profit"]

# 3.2 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Total instances: {len(df)}")
print(f"X_train shape:   {X_train.shape} (80%)")
print(f"X_test shape:    {X_test.shape}  (20%)")
""")

add_code("""# 3.3 ColumnTransformer Specification
numerical_cols = ["R&D Spend", "Administration", "Marketing Spend"]
categorical_cols = ["State"]

# drop='first' drops the reference state ('California') to avoid the dummy variable trap
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
    ]
)

print("Preprocessor configured successfully:")
print(preprocessor)
""")

# 4. Modeling
add_md("""---
# 4. Modeling

We assemble a complete `Pipeline`:
$$\\text{Raw Data} \\longrightarrow \\text{ColumnTransformer} \\longrightarrow \\text{One-Hot Encoding} \\longrightarrow \\text{LinearRegression}$$
""")

add_code("""# 4.1 Assemble and Fit the Pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train the model on the training set
model_pipeline.fit(X_train, y_train)
print("Pipeline fitted successfully on X_train!")
""")

# 5. Model Evaluation
add_md("""---
# 5. Model Evaluation

We evaluate the trained model on the unseen test dataset ($N=10$) using standard regression metrics:
- **MAE (Mean Absolute Error)**: Average absolute magnitude of error.
- **MSE (Mean Squared Error)**: Average of squared errors (heavily penalizes large deviations).
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals, returned to dollar units.
- **$R^2$ (Coefficient of Determination)**: Proportion of variance in `Profit` explained by the regression model.
""")

add_code("""# 5.1 Test Set Predictions and Metrics
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("==================================================")
print("        TEST SET PERFORMANCE METRICS (N=10)       ")
print("==================================================")
print(f"Mean Absolute Error (MAE):       ${mae:,.2f}")
print(f"Mean Squared Error (MSE):        ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE):  ${rmse:,.2f}")
print(f"R² (Variance Explained):         {r2:.4f} ({r2*100:.2f}%)")
print("==================================================")
""")

# 6. Cross-Validation
add_md("""---
# 6. Cross-Validation

Because the dataset contains only 50 observations, a single test split ($N=10$) carries variance. We perform **5-Fold Cross-Validation** through the Pipeline to validate generalization stability.

> **Crucial Data Insight:** The raw dataset is monotonically ordered by `Profit` descending. Therefore, standard sequential $K$-fold splits without shuffling yield non-overlapping profit ranges, distorting fold $R^2$. We utilize `KFold(n_splits=5, shuffle=True, random_state=42)` to ensure balanced distribution across all folds.
""")

add_code("""# 6.1 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    model_pipeline,
    X,
    y,
    cv=kf,
    scoring=["r2", "neg_root_mean_squared_error"],
    return_train_score=False
)

cv_r2 = cv_results["test_r2"]
cv_rmse = -cv_results["test_neg_root_mean_squared_error"]

cv_summary = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)],
    "R² Score": cv_r2,
    "RMSE ($)": cv_rmse
})

display(cv_summary.round(4))
print("-" * 50)
print(f"Mean R²:   {cv_r2.mean():.4f} (± {cv_r2.std():.4f})")
print(f"Mean RMSE: ${cv_rmse.mean():,.2f} (± ${cv_rmse.std():,.2f})")
""")

# 7. Actual vs Predicted Analysis
add_md("""---
# 7. Actual vs Predicted Analysis

We analyze the test set sample-by-sample and generate an **Actual vs Predicted Profit** scatter plot with an ideal $y=x$ reference line.
""")

add_code("""# 7.1 Prediction DataFrame
analysis_df = pd.DataFrame({
    "Actual Profit ($)": y_test.values,
    "Predicted Profit ($)": y_pred,
    "Residual / Error ($)": y_test.values - y_pred,
    "Absolute Error ($)": np.abs(y_test.values - y_pred),
    "Error Rate (%)": (np.abs(y_test.values - y_pred) / y_test.values) * 100
}, index=y_test.index)

display(analysis_df.round(2))
""")

add_code("""# 7.2 Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 7))
plt.scatter(y_test, y_pred, color="#1f77b4", s=80, edgecolors="black", alpha=0.85, label="Test Startups")

# Reference line y = x
line_min = min(y_test.min(), y_pred.min()) - 5000
line_max = max(y_test.max(), y_pred.max()) + 5000
plt.plot([line_min, line_max], [line_min, line_max], color="#d62728", linestyle="--", linewidth=2, label="Ideal Line (y = x)")

plt.title("Actual vs. Predicted Profit (Test Set)", fontweight="bold", pad=12)
plt.xlabel("Actual Profit ($)", fontweight="bold")
plt.ylabel("Predicted Profit ($)", fontweight="bold")
plt.xlim(line_min, line_max)
plt.ylim(line_min, line_max)
plt.legend(frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()
plt.savefig("figures/model_actual_vs_predicted.png", dpi=300)
plt.show()
""")

# 8. Model Coefficients
add_md("""---
# 8. Model Coefficients & Statistical Interpretation

We extract the learned weights from the trained OLS regressor and interpret each parameter.
""")

add_code("""# 8.1 Extract Feature Names and Parameters
feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
coefficients = model_pipeline.named_steps["regressor"].coef_
intercept = model_pipeline.named_steps["regressor"].intercept_

coef_table = pd.DataFrame({
    "Feature Name": feature_names,
    "Coefficient": coefficients
})

print(f"Model Intercept (β0): ${intercept:,.4f}\\n")
display(coef_table.round(4))
""")

add_md("""### Coefficient Interpretation:
1. **Intercept ($\\beta_0 = \\$54,028.04$):** Baseline expected profit for a startup incorporated in California (omitted reference state) with zero spend across all categories.
2. **`R&D Spend` ($+0.8056$):** Each additional $\\$1.00$ invested in R&D is associated with an expected increase of $\\approx \\$0.81$ in Profit, holding all other variables constant.
3. **`Administration` ($-0.0688$):** Each additional $\\$1.00$ in administration overhead is associated with an expected decrease of $\\approx \\$0.07$ in Profit.
4. **`Marketing Spend` ($+0.0299$):** Each additional $\\$1.00$ in marketing is associated with an expected increase of $\\approx \\$0.03$ in Profit.
5. **`State_Florida` ($+\\$938.79$):** Florida startups are predicted to achieve $\\$938.79$ higher profit than California startups, holding spending identical.
6. **`State_New York` ($+\\$6.99$):** New York startups are predicted to achieve $\\$6.99$ higher profit than California startups, holding spending identical.

> **Caution — Association vs. Causation:**  
> These coefficients represent statistical conditional associations within the fitted model. They **do not prove causal mechanisms**. For example, startups that spend heavily on R&D may do so because they possess superior founding teams, higher venture funding, and stronger product validation (omitted confounding variables).
""")

# 9. Deployment
add_md("""---
# 9. Deployment

In the Deployment phase, we demonstrate how the pipeline can be packaged and used in operational environments to score new startup investment opportunities.
""")

add_code("""# 9.1 Inference Function for Operational Deployment
def predict_new_startup(pipeline_model, rd_spend: float, admin_spend: float, marketing_spend: float, state: str) -> float:
    \"\"\"
    Predicts startup profit given departmental budgets and state.
    \"\"\"
    input_data = pd.DataFrame([{
        "R&D Spend": float(rd_spend),
        "Administration": float(admin_spend),
        "Marketing Spend": float(marketing_spend),
        "State": str(state)
    }])
    predicted_profit = pipeline_model.predict(input_data)[0]
    return predicted_profit

# 9.2 Demonstration with Prospective Startups
prospective_deals = [
    {"Name": "Alpha AI", "State": "New York", "R&D": 130000.0, "Admin": 120000.0, "Mkt": 310000.0},
    {"Name": "Beta Biotech", "State": "California", "R&D": 95000.0, "Admin": 105000.0, "Mkt": 220000.0},
    {"Name": "Gamma CleanTech", "State": "Florida", "R&D": 160000.0, "Admin": 140000.0, "Mkt": 400000.0},
]

print("=== LIVE MODEL INFERENCE ON NEW OPPORTUNITIES ===")
for deal in prospective_deals:
    val = predict_new_startup(model_pipeline, deal["R&D"], deal["Admin"], deal["Mkt"], deal["State"])
    print(f"Startup '{deal['Name']}' ({deal['State']}) -> Forecasted Profit: ${val:,.2f}")
""")

# Write notebook file
notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13.1"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("fifty_startups_ml.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook 'fifty_startups_ml.ipynb' generated successfully!")
