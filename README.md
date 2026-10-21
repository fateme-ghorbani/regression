# House Price Analysis and Prediction Using Linear Regression

## Overview

This project explores a housing dataset and implements a **Linear Regression** model to predict property prices based on housing characteristics.

The notebook covers exploratory data analysis (EDA), data cleaning, missing-value imputation, feature encoding, outlier analysis, and model evaluation. It also highlights the importance of preventing data leakage when building machine learning models.

## Dataset

The project uses `HousePrice1.csv`, which initially contains **3,479 records and 8 columns**.

The dataset includes the following features:

| Feature | Description |
|---|---|
| `Area` | Property area |
| `Room` | Number of rooms |
| `Parking` | Parking availability |
| `Warehouse` | Warehouse availability |
| `Elevator` | Elevator availability |
| `Address` | Property location |
| `Price` | Property price |
| `Price(USD)` | Property price expressed in US dollars |

**Target variable:** `Price`

## Exploratory Data Analysis

The notebook explores the dataset using statistical summaries and visualizations, including:

- Histograms and boxplots to examine the distributions of property area and price.
- Count plots and pie charts to explore property characteristics.
- Regression plots to examine relationships between price and individual features.
- Analysis of potential outliers and differences across property attributes.

## Data Preprocessing

The notebook includes the following preprocessing steps:

1. Removes records with missing addresses.
2. Encodes the categorical `Address` feature using `LabelEncoder`.
3. Uses `KNNImputer` to handle remaining missing values.
4. Identifies potential outliers using Z-scores.
5. Applies feature standardization using `StandardScaler`.

Some preprocessing outputs, including the outlier-filtered dataset and standardized dataset, are created but are not used in the subsequent model training in the current implementation.

## Model Development

A Linear Regression model is trained using Scikit-learn.

- **Algorithm:** Linear Regression
- **Train-test split:** 80% training and 20% testing
- **Random state:** 42
- **Evaluation metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Squared Logarithmic Error (MSLE), and R-squared (\(R^2\))

## Initial Evaluation Results

The current notebook reports the following metrics:

| Metric | Reported Value |
|---|---:|
| MSE | 6,299.04 |
| RMSE | 79.37 |
| MAE | 64.41 |
| MSLE | Approximately 0 |
| R-squared | Approximately 1.00 |

**Important:** These results should not be interpreted as reliable predictive performance. The current feature set includes `Price(USD)`, which is directly derived from the target variable `Price`. This introduces **target leakage** and explains the unrealistically strong results.

A valid evaluation requires removing `Price(USD)` from the predictors and retraining the model before interpreting its performance.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Linear Regression
- Exploratory Data Analysis

## Repository Structure

```text
regression/
├── regression.ipynb
├── HousePrice1.csv
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/fateme-ghorbani/regression.git
cd regression
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook regression.ipynb
```

Run the notebook cells in order to explore the dataset, perform preprocessing, train the regression model, and review the evaluation results.

## Limitations and Future Improvements

- Remove `Price(USD)` from the input features to prevent target leakage.
- Split the data into training and test sets before fitting preprocessing transformations.
- Fit the imputer, encoder, and scaler using training data only, then apply them to the test data.
- Ensure the outlier-filtered dataset is actually used if outlier removal is part of the final workflow.
- Re-evaluate the model after correcting the preprocessing and feature selection.
- Compare Linear Regression with alternative regression models.
- Inspect residuals to assess model assumptions and prediction errors.

## Learning Outcomes

This project demonstrates an end-to-end introductory regression workflow, from exploratory data analysis and preprocessing to model training and evaluation. It also illustrates why careful feature selection and leakage-free evaluation are essential for trustworthy machine learning results.
