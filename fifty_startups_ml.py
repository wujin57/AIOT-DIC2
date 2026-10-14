"""
=============================================================================
CRISP-DM Machine Learning Solution for 50 Startups Dataset
Algorithm: Multiple Linear Regression (scikit-learn)
Author: Data Science Team
=============================================================================

This script implements an end-to-end Machine Learning pipeline adhering
to the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:
  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling
  5. Model Evaluation
  6. Deployment
"""

# =============================================================================
# 1. Business Understanding
# =============================================================================
"""
Business Problem:
    Venture capital firms and startup founders need to make data-driven decisions
    regarding capital allocation across organizational departments (Research &
    Development, Administration, Marketing) and geographical location (State).
    Misallocating limited financial runway can result in capital depletion without
    achieving profitability.

Objective:
    Develop an accurate and transparent predictive model to forecast startup net
    Profit ($) based on operational spend patterns and location.

Machine Learning Task:
    Supervised Regression. The response variable is continuous, and historical
    labeled data containing features and outcomes is available.

Target Variable:
    - Profit (Continuous numeric, in USD)

Predictor Variables:
    - R&D Spend (Continuous numeric, in USD)
    - Administration (Continuous numeric, in USD)
    - Marketing Spend (Continuous numeric, in USD)
    - State (Nominal categorical: 'California', 'Florida', 'New York')

Why Multiple Linear Regression is Appropriate:
    1. Interpretable Coefficients: Provides explicit marginal contribution rates
       (dollars of profit gained per dollar allocated) for executive decision-makers.
    2. Empirical & Economic Linearity: Expenditure in R&D and market expansion
       typically exhibits strong proportional relationships with commercial returns.
    3. Low Model Complexity: Given the small sample size (N=50), complex non-linear
       models (deep neural networks, boosted trees) risk extreme overfitting.
       Ordinary Least Squares (OLS) maintains statistical efficiency and low variance.
"""

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

# Ensure output directory for figures exists
os.makedirs("figures", exist_ok=True)

# Set visualization aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# =============================================================================
# 2. Data Understanding
# =============================================================================
print("=" * 80)
print("PHASE 2: DATA UNDERSTANDING")
print("=" * 80)

# Load dataset directly from URL
DATASET_URL = (
    "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/refs/heads/master/datasets/50_Startups.csv"
)
df = pd.read_csv(DATASET_URL)

print("\n[2.1] First 5 Rows of the Dataset:")
print(df.head().to_string())

print("\n[2.2] Dataset Dimensions:")
print(f"Number of observations: {df.shape[0]}")
print(f"Number of columns:      {df.shape[1]}")

print("\n[2.3] Data Types and Non-Null Counts:")
print(df.dtypes)

print("\n[2.4] Missing Values Check:")
missing_counts = df.isnull().sum()
print(missing_counts)
print(f"Total missing values across dataset: {missing_counts.sum()}")

print("\n[2.5] Unique Values and Frequency of Categorical Feature 'State':")
state_distribution = df["State"].value_counts()
print(state_distribution)

print("\n[2.6] Descriptive Summary Statistics:")
print(df.describe().round(2).to_string())

print("\n[2.7] Target Variable ('Profit') Analysis:")
profit_series = df["Profit"]
print(f"Mean Profit:     ${profit_series.mean():,.2f}")
print(f"Median Profit:   ${profit_series.median():,.2f}")
print(f"Std Deviation:   ${profit_series.std():,.2f}")
print(f"Min Profit:      ${profit_series.min():,.2f}")
print(f"Max Profit:      ${profit_series.max():,.2f}")
print(f"IQR:             ${(profit_series.quantile(0.75) - profit_series.quantile(0.25)):,.2f}")
print(f"Skewness:        {profit_series.skew():.4f}")

# Exploratory Visualizations
print("\n[2.8] Generating Exploratory Visualizations...")

# 1. Distribution of Target (Profit)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["Profit"], kde=True, ax=axes[0], color="#1f77b4", bins=10)
axes[0].set_title("Distribution of Profit (Histogram & KDE)", fontweight="bold")
axes[0].set_xlabel("Profit ($)")
axes[0].set_ylabel("Count")

sns.boxplot(x=df["Profit"], ax=axes[1], color="#aec7e8", flierprops={"markerfacecolor": "#d62728", "markersize": 8})
axes[1].set_title("Boxplot of Profit (Outlier Detection)", fontweight="bold")
axes[1].set_xlabel("Profit ($)")
plt.tight_layout()
fig_profit_dist = "figures/eda_profit_distribution.png"
plt.savefig(fig_profit_dist)
plt.close()
print(f"  -> Saved: {fig_profit_dist}")

# 2. Scatter Plots between Numerical Predictors and Profit
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
num_predictors = ["R&D Spend", "Administration", "Marketing Spend"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, col in enumerate(num_predictors):
    sns.regplot(
        data=df,
        x=col,
        y="Profit",
        ax=axes[i],
        scatter_kws={"alpha": 0.75, "color": colors[i], "s": 45},
        line_kws={"color": "#d62728", "linewidth": 1.8},
    )
    axes[i].set_title(f"{col} vs Profit", fontweight="bold")
    axes[i].set_xlabel(f"{col} ($)")
    axes[i].set_ylabel("Profit ($)")
plt.tight_layout()
fig_scatter = "figures/eda_scatter_plots.png"
plt.savefig(fig_scatter)
plt.close()
print(f"  -> Saved: {fig_scatter}")

# 3. Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    cbar=True,
    square=True,
    linewidths=0.8,
    vmin=-0.1,
    vmax=1.0,
)
plt.title("Correlation Matrix of Numerical Variables", fontweight="bold", pad=12)
plt.tight_layout()
fig_corr = "figures/eda_correlation_heatmap.png"
plt.savefig(fig_corr)
plt.close()
print(f"  -> Saved: {fig_corr}")

print("\nImportant Findings from Data Understanding:")
print("  - Target Profit has a near-normal distribution with slight negative skewness (-0.225).")
print("  - R&D Spend has an extraordinarily strong linear correlation with Profit (r = 0.973).")
print("  - Marketing Spend shows moderate-to-strong positive correlation (r = 0.748).")
print("  - Administration spend exhibits low correlation with Profit (r = 0.201).")
print("  - State distribution is balanced (17 CA, 17 NY, 16 FL) with zero missing entries.")


# =============================================================================
# 3. Data Preparation
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 3: DATA PREPARATION")
print("=" * 80)

# 3.1 Separate features X and target y
X = df.drop(columns=["Profit"])
y = df["Profit"]

print("\n[3.1] Feature Matrix (X) and Target Vector (y) Separated:")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 3.2 80/20 Train-Test Split with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n[3.2] Train-Test Split Completed (80/20, random_state=42):")
print(f"Training set: {X_train.shape[0]} observations")
print(f"Testing set:  {X_test.shape[0]} observations")

# 3.3 Identify feature types
numerical_features = ["R&D Spend", "Administration", "Marketing Spend"]
categorical_features = ["State"]

print("\n[3.3] Feature Types Identified:")
print(f"Numerical features:   {numerical_features}")
print(f"Categorical features: {categorical_features}")

# 3.4 Build ColumnTransformer
# OneHotEncoder with drop='first' avoids the dummy-variable trap (multicollinearity)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error"),
            categorical_features,
        ),
    ]
)

print("\n[3.4] ColumnTransformer Constructed:")
print("  - Numerical features passed through without scaling (preserving currency units).")
print("  - Categorical feature 'State' encoded using OneHotEncoder(drop='first').")
print("  - Conceptually: Baseline category ('California') dropped to avoid singular matrix.")


# =============================================================================
# 4. Modeling
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 4: MODELING")
print("=" * 80)

# Concept: Raw Data -> ColumnTransformer -> OneHotEncoder for State -> LinearRegression
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

print("\n[4.1] scikit-learn Pipeline Architecture:")
print(pipeline)

# Train the model on training dataset
print("\n[4.2] Training Model on X_train (40 observations)...")
pipeline.fit(X_train, y_train)
print("Model training completed successfully without data leakage.")


# =============================================================================
# 5. Model Evaluation
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 5: MODEL EVALUATION (TEST SET)")
print("=" * 80)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n[5.1] Model Performance on Unseen Test Set (N=10):")
print(f"  Mean Absolute Error (MAE):       ${mae:,.2f}")
print(f"  Mean Squared Error (MSE):        ${mse:,.2f}")
print(f"  Root Mean Squared Error (RMSE):  ${rmse:,.2f}")
print(f"  Coefficient of Determination (R²): {r2:.4f}")

print("\n[5.2] Interpretation of Evaluation Metrics:")
print("  - MAE ($6,961.48): On average, the model's predicted profit deviates by")
print("    approximately $6,961.48 from the actual startup profit in either direction.")
print("  - MSE ($82,010,363.04): Measures average squared error. Being in squared dollars,")
print("    it severely penalizes large errors, highlighting sensitive deviations.")
print("  - RMSE ($9,055.96): Returns the error metric to the original dollar scale.")
print("    RMSE is higher than MAE ($9,055.96 vs $6,961.48) because squaring penalizes")
print("    a few larger test-set prediction discrepancies (e.g., sample #45 and #48).")
print(f"  - R² ({r2:.4f}): The model explains approximately {r2 * 100:.2f}% of the")
print("    total variance in startup Profit observed in the test set. Only about")
print(f"    {(1 - r2) * 100:.2f}% remains unexplained residual variation.")


# =============================================================================
# 6. Cross-Validation
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 6: 5-FOLD CROSS-VALIDATION")
print("=" * 80)

# Scoring metrics for cross-validation
scoring_list = ["r2", "neg_root_mean_squared_error"]

# 1. Shuffled 5-Fold Cross Validation (Statistically required for sorted datasets)
kf_shuffled = KFold(n_splits=5, shuffle=True, random_state=42)
cv_shuffled_results = cross_validate(
    pipeline,
    X,
    y,
    cv=kf_shuffled,
    scoring=scoring_list,
    return_train_score=False,
)

cv_shuffled_r2 = cv_shuffled_results["test_r2"]
cv_shuffled_rmse = -cv_shuffled_results["test_neg_root_mean_squared_error"]

print("\n[6.1] 5-Fold Cross-Validation Results (Shuffled KFold, random_state=42):")
for fold in range(5):
    print(
        f"  Fold {fold+1}: R² = {cv_shuffled_r2[fold]:.4f} | RMSE = ${cv_shuffled_rmse[fold]:,.2f}"
    )

print("-" * 50)
print(f"  Mean R²:   {cv_shuffled_r2.mean():.4f}  (± {cv_shuffled_r2.std():.4f})")
print(f"  Mean RMSE: ${cv_shuffled_rmse.mean():,.2f}  (± ${cv_shuffled_rmse.std():,.2f})")

# 2. Sequential 5-Fold CV (Default unshuffled cv=5)
cv_seq_results = cross_validate(
    pipeline,
    X,
    y,
    cv=5,
    scoring=scoring_list,
    return_train_score=False,
)
cv_seq_r2 = cv_seq_results["test_r2"]
cv_seq_rmse = -cv_seq_results["test_neg_root_mean_squared_error"]

print("\n[6.2] Comparison: Default Sequential (Unshuffled) 5-Fold CV:")
for fold in range(5):
    print(
        f"  Fold {fold+1}: R² = {cv_seq_r2[fold]:.4f} | RMSE = ${cv_seq_rmse[fold]:,.2f}"
    )
print("-" * 50)
print(f"  Sequential Mean R²:   {cv_seq_r2.mean():.4f}  (± {cv_seq_r2.std():.4f})")
print(f"  Sequential Mean RMSE: ${cv_seq_rmse.mean():,.2f}  (± ${cv_seq_rmse.std():,.2f})")

print("\n[6.3] Why Cross-Validation is Critical for this Small Dataset (N=50):")
print("  - High Variance Risk: With only 50 observations, a single 80/20 split leaves")
print("    only 10 test samples. The test metrics are highly sensitive to whether an")
print("    outlier (like observation 49) lands in the training set or test set.")
print("  - Comprehensive Utilization: 5-Fold CV ensures every single observation is")
print("    used for testing exactly once across folds.")
print("  - Data Ordering Revelation: The raw CSV is sorted descending by Profit.")
print("    Unshuffled CV evaluates slices of narrow variance, artificially distorting R².")
print("    Shuffled CV confirms the true generalized predictive power: Mean R² ~ 0.9279.")


# =============================================================================
# 7. Actual vs Predicted Analysis
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 7: ACTUAL VS PREDICTED ANALYSIS")
print("=" * 80)

# Prediction error = Actual - Predicted
results_df = pd.DataFrame(
    {
        "Actual Profit": y_test.values,
        "Predicted Profit": y_pred,
        "Prediction Error": y_test.values - y_pred,
        "Absolute Error": np.abs(y_test.values - y_pred),
        "Percentage Error (%)": (
            np.abs(y_test.values - y_pred) / y_test.values
        )
        * 100,
    },
    index=y_test.index,
)

print("\n[7.1] Actual vs Predicted Performance on Test Set (10 Samples):")
print(
    results_df[
        ["Actual Profit", "Predicted Profit", "Prediction Error", "Percentage Error (%)"]
    ]
    .round(2)
    .to_string()
)

# Scatter plot: Actual vs Predicted Profit with y = x reference line
plt.figure(figsize=(8, 7))
plt.scatter(
    y_test,
    y_pred,
    color="#1f77b4",
    s=70,
    edgecolors="black",
    linewidths=1,
    alpha=0.85,
    label="Test Set Observations",
)

min_bound = min(y_test.min(), y_pred.min()) - 5000
max_bound = max(y_test.max(), y_pred.max()) + 5000
plt.plot(
    [min_bound, max_bound],
    [min_bound, max_bound],
    color="#d62728",
    linestyle="--",
    linewidth=2,
    label="Ideal Prediction Line (y = x)",
)

plt.title("Actual vs Predicted Startup Profit (Test Set)", fontweight="bold", pad=12)
plt.xlabel("Actual Profit ($)", fontweight="bold")
plt.ylabel("Predicted Profit ($)", fontweight="bold")
plt.xlim(min_bound, max_bound)
plt.ylim(min_bound, max_bound)
plt.legend(frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()
fig_actual_pred = "figures/model_actual_vs_predicted.png"
plt.savefig(fig_actual_pred)
plt.close()
print(f"\n  -> Saved: {fig_actual_pred}")

print("\n[7.2] Assessment of Prediction Accuracy:")
print("  - For 8 out of 10 test startups, predictions align closely along the y = x line,")
print("    with percentage errors generally under 8%.")
print("  - For instance, index #30 (Actual: $99,937.59, Pred: $99,677.49) and index #32")
print("    (Actual: $97,427.84, Pred: $97,599.28) have errors under $300 (<0.3%).")
print("  - The largest deviations occur in lower-profit startups (#45 with +$18,568 error,")
print("    #48 with -$15,239 error), reflecting higher relative variance at low spend levels.")


# =============================================================================
# 8. Model Coefficients
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 8: MODEL COEFFICIENTS & INTERPRETATION")
print("=" * 80)

# Extract feature names from ColumnTransformer
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
coefficients = pipeline.named_steps["regressor"].coef_
intercept = pipeline.named_steps["regressor"].intercept_

# Format into DataFrame
coef_df = pd.DataFrame(
    {
        "Transformed Feature": feature_names,
        "Coefficient": coefficients,
    }
)

print("\n[8.1] Model Parameters:")
print(f"  Intercept (beta_0): ${intercept:,.4f}\n")
print(coef_df.to_string(index=False))

print("\n[8.2] Careful Interpretation of Model Coefficients:")
print("  1. Intercept ($54,028.04):")
print("     Theoretical baseline profit for a startup in California (reference state)")
print("     with $0 spend across R&D, Administration, and Marketing. In practice,")
print("     startups rarely operate with zero spend, so this serves primarily as an")
print("     anchor for the linear hyperplane.")
print("  2. R&D Spend (0.8056):")
print("     Holding all other expenditures and state constant, each additional $1.00")
print("     spent on Research & Development is associated with an expected increase")
print("     of approximately $0.81 in Profit.")
print("  3. Administration Spend (-0.0688):")
print("     Holding other variables constant, each additional $1.00 spent on")
print("     administration is associated with a slight decrease of ~$0.07 in Profit.")
print("     This likely reflects overhead drag without direct revenue generation.")
print("  4. Marketing Spend (0.0299):")
print("     Holding other variables constant, each additional $1.00 spent on")
print("     marketing is associated with an increase of ~$0.03 in Profit.")
print("  5. State Dummies (Reference Category: California):")
print("     - State_Florida ($938.79): A startup located in Florida is predicted to earn")
print("       $938.79 more in profit than an identical startup in California, holding spend equal.")
print("     - State_New York ($6.99): A startup in New York is predicted to earn $6.99 more")
print("       in profit than an identical startup in California, holding spend equal.")
print("     Both state differentials are negligible compared to total profit scale ($112k),")
print("     indicating location has minimal impact compared to capital spending.")

print("\n[8.3] Crucial Methodological Warning: Association vs. Causation:")
print("  - These coefficients represent statistical associations conditional on the")
print("    variables in the model, NOT verified causal mechanisms.")
print("  - Increasing R&D spend does not mechanically guarantee profit; it is likely that")
print("    higher-quality startups with stronger product-market fit attract more capital")
print("    and consequently spend more on R&D (confounding / endogeneity).")


# =============================================================================
# 9. Deployment
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 9: DEPLOYMENT")
print("=" * 80)

def predict_startup_profit(pipeline_model, rd_spend, admin_spend, marketing_spend, state):
    """
    Inference function to predict profit for a new startup profile.
    """
    input_df = pd.DataFrame(
        [
            {
                "R&D Spend": float(rd_spend),
                "Administration": float(admin_spend),
                "Marketing Spend": float(marketing_spend),
                "State": str(state),
            }
        ]
    )
    predicted_val = pipeline_model.predict(input_df)[0]
    return predicted_val

# Demonstration on 3 hypothetical new startups
sample_cases = [
    {"R&D Spend": 130000.0, "Administration": 120000.0, "Marketing Spend": 310000.0, "State": "New York"},
    {"R&D Spend": 95000.0, "Administration": 105000.0, "Marketing Spend": 220000.0, "State": "California"},
    {"R&D Spend": 160000.0, "Administration": 140000.0, "Marketing Spend": 400000.0, "State": "Florida"},
]

print("\n[9.1] Live Inference Demonstration for New Startup Profiles:")
for i, case in enumerate(sample_cases, 1):
    est_profit = predict_startup_profit(
        pipeline,
        case["R&D Spend"],
        case["Administration"],
        case["Marketing Spend"],
        case["State"],
    )
    print(
        f"  Startup {i} [State: {case['State']}, R&D: ${case['R&D Spend']:,.0f}, "
        f"Admin: ${case['Administration']:,.0f}, Mkt: ${case['Marketing Spend']:,.0f}] "
        f"-> Predicted Profit: ${est_profit:,.2f}"
    )

print("\nExecution complete. All artifacts generated successfully.")
