# Credit Card Default Prediction Using Machine Learning

A machine-learning project that explores credit-card customer data,
prepares it for classification, compares five models, tunes Random
Forest, and reviews model performance and feature importance.

## Project Overview

This notebook builds classification models to predict whether a
credit-card customer will default on payment in the next month. The
target column is `default.payment.next.month`, where `1` indicates
default and `0` indicates no default.

The notebook is organised into four main tasks:

1.  Exploratory Data Analysis (EDA)
2.  Data Preprocessing
3.  Model Training
4.  Model Evaluation, Visualisation and Hyperparameter Tuning

The work is an academic analysis and should not be used directly for
real lending or credit decisions without further validation.

## Dataset

The notebook loads a local CSV file named `Credit_Card.csv`, using a
semicolon (`;`) as the separator:

``` python
df = pd.read_csv("Credit_Card.csv", sep=";")
```

The dataset loaded by the notebook contains **34,788 rows and 30
columns**.

Important fields used or explored include:

  -----------------------------------------------------------------------
  Field                               Description in the project
  ----------------------------------- -----------------------------------
  `default.payment.next.month`        Target variable: next-month default
                                      status

  `LIMIT_BAL`                         Customer credit limit

  `LIMIT_BAL_LOG`                     Log-transformed credit limit
                                      feature

  `BILL_AMT_SUM`                      Aggregated billing amount

  `PAY_0`, `PAY_2`--`PAY_6`           Repayment-status history fields

  `PAY_AMT1`, `PAY_AMT2`, etc.        Repayment amount fields

  `SEX`, `EDUCATION`, `MARRIAGE`,     Categorical/demographic fields
  `CITY`                              

  `RISK_RATING`                       Risk-related field examined in EDA

  `risk_leak`                         Investigated and removed before
                                      model training due to possible
                                      target leakage
  -----------------------------------------------------------------------

The target is imbalanced in the source data: approximately **80.87%** of
records are non-defaults and **19.13%** are defaults.

> **Dataset requirement:** The notebook expects `Credit_Card.csv` to be
> available in the runtime's working directory. The notebook does not
> itself download the file.

## Notebook Workflow

### 1. Exploratory Data Analysis

The notebook:

-   Checks dataset dimensions, column names, data types and descriptive
    statistics.
-   Examines missing values and duplicate records.
-   Reviews numerical features such as `LIMIT_BAL`, `AGE`,
    `BILL_AMT_SUM` and `LIMIT_BAL_LOG` using summary statistics,
    histograms and boxplots.
-   Examines categorical fields and calculates default rates across
    categories.
-   Investigates repayment-history variables and their relationship with
    the target.
-   Checks the relationship between `risk_leak` and default status.

The EDA identifies class imbalance and outliers, especially in
credit-limit and billing variables. The notebook's written analysis says
these outliers are retained because they may represent genuine
customers; scaling is applied later.

### 2. Data Preprocessing

The notebook creates a working copy named `df_processed` and performs
the following preparation:

1.  Removes duplicate rows if any are present.
2.  Fills missing numerical values in `LIMIT_BAL`, `AGE`, `PAY_AMT1` and
    `PAY_AMT2` with each column's median.
3.  Fills missing values in `SEX`, `EDUCATION` and `MARRIAGE` with each
    column's mode.
4.  Recalculates `LIMIT_BAL_LOG` using `numpy.log1p(LIMIT_BAL)`.
5.  Investigates `risk_leak` and removes it from the processed dataset
    because it may reveal information about the target.
6.  Encodes `CITY` using scikit-learn's `LabelEncoder`.
7.  Separates the target (`y`) from the input features (`X`).
8.  Splits the data into training and testing sets with an 80/20 split,
    `random_state=42`, and stratification on the target.
9.  Standardises the features using `StandardScaler`, fitting the scaler
    on the training data and applying it to the test data.
10. Applies SMOTE to the scaled training data to balance the target
    classes.

After the split, the notebook reports 25,544 training rows and 6,386
test rows, with 28 input features. SMOTE balances the training classes
to 20,235 observations in each class. The test set is not resampled.

### 3. Model Training

Five classifiers are initialised and trained on the SMOTE-resampled
training data:

  -----------------------------------------------------------------------
  Model                               Configuration shown in the notebook
  ----------------------------------- -----------------------------------
  Logistic Regression                 `max_iter=500`, `random_state=42`

  Random Forest                       `n_estimators=200`,
                                      `random_state=42`

  XGBoost                             `random_state=42`,
                                      `eval_metric="logloss"`

  AdaBoost                            `random_state=42`

  Multi-Layer Perceptron (MLP)        `hidden_layer_sizes=(100,)`,
                                      `max_iter=500`, `random_state=42`
  -----------------------------------------------------------------------

The trained models generate class predictions and default-class
probabilities for the held-out test data.

### 4. Model Evaluation

The notebook compares models using:

-   **Accuracy** --- overall proportion of correct predictions.
-   **Precision** --- among predicted defaults, the proportion that are
    actual defaults.
-   **Recall** --- among actual defaults, the proportion correctly
    identified.
-   **F1-score** --- harmonic mean of precision and recall.
-   **ROC-AUC** --- discrimination across classification thresholds.

It also produces classification reports, confusion matrices and a
combined ROC-curve plot.

#### Baseline results recorded in the notebook

The following values are from the notebook's displayed performance
table, evaluated on the test split:

  Model                   Accuracy   Precision   Recall   F1-score   ROC-AUC
  --------------------- ---------- ----------- -------- ---------- ---------
  Random Forest             0.8276      0.6239   0.4288     0.5083    0.8112
  XGBoost                   0.8227      0.6242   0.3693     0.4640    0.7624
  AdaBoost                  0.7805      0.4740   0.5147     0.4935    0.7479
  Logistic Regression       0.7700      0.4565   0.5622     0.5039    0.7400
  MLP                       0.7350      0.3953   0.5192     0.4489    0.7223

These are the notebook's recorded results, not guarantees of performance
on new data. In particular, default-class recall varies across the
models, so accuracy alone does not describe how well defaults are
detected.

### 5. Random Forest Hyperparameter Tuning

The notebook tunes Random Forest with two approaches, using ROC-AUC as
the search scoring metric and five-fold cross-validation.

**GridSearchCV** searches combinations of:

-   `n_estimators`: 100 or 200
-   `max_depth`: 10 or `None`
-   `min_samples_split`: 2 or 5
-   `min_samples_leaf`: 1 or 2

The best parameters recorded are:

``` python
{
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 200
}
```

The best cross-validation ROC-AUC printed by GridSearchCV is
approximately `0.9463`.

**RandomizedSearchCV** tests 20 sampled parameter combinations from a
larger set, also using five-fold cross-validation. The best parameters
recorded are:

``` python
{
    "n_estimators": 300,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_depth": None
}
```

The best cross-validation ROC-AUC printed by RandomizedSearchCV is
approximately `0.9469`.

The notebook then evaluates both tuned estimators on the same held-out
test set and compares their test ROC-AUC values with the original Random
Forest:

  Random Forest version      Test ROC-AUC
  ------------------------ --------------
  Original Random Forest         0.811203
  Grid Search RF                 0.811203
  Random Search RF               0.812144

The recorded test ROC-AUC values show only a small difference between
the original and tuned Random Forest models.

### 6. Feature Importance

The notebook extracts `feature_importances_` from the best GridSearchCV
Random Forest model, sorts the features by importance and plots the top
15.

This provides a model-specific view of which input features contributed
most to the fitted forest. Feature importance indicates patterns learned
by this model; it does not establish causation or prove that a feature
is appropriate for operational credit decisions.

## How to Run the Notebook

### Option A: Google Colab

1.  Upload the `.ipynb` notebook to Google Colab or open it from your
    repository.
2.  Upload `Credit_Card.csv` to the Colab session, or mount storage and
    update the CSV path in the dataset-loading cell.
3.  Run the notebook cells in order from the beginning.
4.  Review the printed evaluation tables and generated plots.

### Option B: Run Locally

Use a Python environment with the notebook's required libraries
installed. The imports used in the notebook include:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

Then start Jupyter Notebook or JupyterLab, open the `.ipynb` file,
ensure `Credit_Card.csv` is in the expected working directory, and run
the cells in order.

The notebook does not include a pinned dependency file, so the command
above installs current compatible package versions rather than
reproducing a specific environment exactly.

## Reproducibility and Limitations

-   The train/test split uses `random_state=42` and stratifies by the
    target.
-   The scaler is fitted on the training split and then applied to the
    test split.
-   SMOTE is applied to the training data before the cross-validation
    searches. Because resampling happens before cross-validation,
    synthetic samples may influence cross-validation folds; a more
    robust evaluation would put SMOTE inside an imbalanced-learn
    pipeline so it is applied separately within each fold.
-   The notebook uses `LabelEncoder` for `CITY`. This assigns integer
    codes to categories; an encoding approach designed for nominal input
    features may be worth comparing.
-   The dataset includes demographic fields. Any real-world use would
    require careful fairness, privacy, explainability and regulatory
    review.
-   The model scores are based on this dataset and split. Temporal
    validation, independent test data and post-deployment monitoring are
    not shown in the notebook.

## Repository Contents

A typical repository structure for this project is:

``` text
predictive-modelling-using-ML/
├── README.md
├── Predictive Modelling Ayushi(1).ipynb
└── Credit_Card.csv       # Include only if permitted by the dataset's terms
```

If the dataset cannot be redistributed, leave it out of GitHub and add
instructions explaining how to obtain it and where to place it before
running the notebook.

## Project Summary

This project demonstrates an end-to-end classification workflow for
next-month credit-card default prediction: data exploration, cleaning,
feature preparation, class balancing, model comparison, evaluation,
Random Forest hyperparameter search and feature-importance
visualisation.

The notebook is an educational modelling exercise. Its evaluation
results should be interpreted with the dataset, preprocessing decisions
and validation limitations in mind.
