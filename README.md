# FIES Household Income Prediction

Machine learning models that predict the **monthly household income** and the
**income bracket** of Filipino families from demographic composition, asset
ownership, occupation and expenditure patterns recorded in the Philippine
Statistics Authority's Family Income and Expenditure Survey (FIES).

## Overview

The project covers two supervised learning tasks on 41,544 household records:

- **Regression**: predict monthly household income in Philippine pesos (PHP).
  Models: NuSVR, Random Forest, Gradient Boosting, Histogram-based Gradient
  Boosting, and a Stacking Regressor (NuSVR and HGB with a Ridge meta-learner).
- **Classification**: assign each household to one of seven income brackets
  (Poor through Rich, following the PIDS income class thresholds). Models:
  Random Forest, Gradient Boosting, Histogram-based Gradient Boosting, a hard
  Voting Classifier, and a Stacking Classifier.

Key feature engineering steps:

- Mapping several hundred occupation titles to the ten ISCO major groups with
  an ordinal skill level.
- Converting the household head's highest grade completed to years of schooling.
- Housing features such as floor area per capita and people per bedroom.
- Log transformation of skewed expenditure columns and of the regression target.

## Repository Structure

```
.
├── household_income_prediction.ipynb   Analysis, training and evaluation
├── fies_income/
│   ├── occupations.py                  ISCO mapping and skill levels
│   ├── features.py                     Target and feature engineering
│   └── modeling.py                     Preprocessor, metrics, plot helpers
├── tests/                              pytest suite for fies_income
├── data/                               Dataset location (not tracked)
├── requirements.txt                    Runtime dependencies
├── requirements-dev.txt                Notebook, testing and linting tools
└── pyproject.toml                      Ruff and pytest configuration
```

## Requirements

- Python 3.11 or newer (developed on Python 3.12)
- About 4 GB of free memory
- The pinned packages in `requirements.txt`: numpy, pandas 3, scikit-learn 1.8,
  matplotlib and seaborn

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/alumbayar/fies-household-income-prediction.git
   cd fies-household-income-prediction
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

2. Download the
   [Family Income and Expenditure dataset](https://www.kaggle.com/datasets/grosvenpaul/family-income-and-expenditure)
   from Kaggle and extract it so the CSV is at:

   ```
   data/Family Income and Expenditure.csv
   ```

   With `curl`:

   ```bash
   curl -L -o data/archive.zip \
     https://www.kaggle.com/api/v1/datasets/download/grosvenpaul/family-income-and-expenditure
   unzip data/archive.zip -d data && rm data/archive.zip
   ```

## Usage

Open the notebook from the repository root so that `fies_income` and the
`data/` path resolve:

```bash
jupyter lab household_income_prediction.ipynb
```

To execute it end to end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 household_income_prediction.ipynb
```

A full run trains ten models, including several NuSVR and Gradient Boosting
fits on roughly 33,000 rows, and takes about 1 hour 45 minutes on a 4-core
machine. All randomness is seeded with `RANDOM_STATE = 42`, so repeated runs
produce identical results.

## Testing and Linting

```bash
pytest --cov=fies_income --cov-report=term-missing
ruff check .
ruff format --check .
```

The test suite covers every line of the `fies_income` package.

## Results

Evaluated on a stratified 20% hold-out set (8,309 households).

### Regression: Monthly Income

| Model                  | MAE (PHP) | MAPE (%) | R2    | RMSE (PHP) |
|------------------------|----------:|---------:|------:|-----------:|
| Stacking Regressor     |     3,778 |     16.3 | 0.695 |     13,554 |
| Hist Gradient Boosting |     3,780 |     16.4 | 0.697 |     13,505 |
| Gradient Boosting      |     3,864 |     17.2 | 0.741 |     12,497 |
| Random Forest          |     4,027 |     18.0 | 0.714 |     13,137 |
| NuSVR                  |     4,433 |     18.8 | 0.613 |     15,277 |

The Stacking Regressor has the lowest percentage error. Gradient Boosting has
the highest R2 and lowest RMSE, which suggests it makes fewer large errors on
high-income households.

### Classification: Income Bracket

| Model                        | Accuracy | F1 Macro |
|------------------------------|---------:|---------:|
| Gradient Boosting Classifier |    0.792 |    0.604 |
| Hist GBC Classifier          |    0.769 |    0.537 |
| Voting Classifier            |    0.783 |    0.521 |
| Stacking Classifier          |    0.772 |    0.517 |
| Random Forest Classifier     |    0.770 |    0.477 |

The Gradient Boosting Classifier recalls 90% of Poor households and 78% of Low
Income households. Recall is much lower for the Upper Income and Rich brackets,
which together make up less than 1% of the data.

The notebook also contains distribution plots, a correlation heatmap, an
actual-versus-predicted scatter plot for the best regressor and a normalized
confusion matrix for the best classifier.

## Data Source

Philippine Statistics Authority, Family Income and Expenditure Survey, as
published on Kaggle by `grosvenpaul`.
