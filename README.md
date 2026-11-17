# Carfornia_House_prediction_model

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive machine-learning application that estimates the **median home value of a California census block** using linear regression and scikit-learn's California Housing dataset.

> This project does not predict the exact sale price of an individual property. The dataset is based on 1990 census information and the application is intended for education and demonstration.

## Features

- Interactive controls for all eight model inputs
- Live prediction displayed in U.S. dollars
- Actual-versus-predicted test-set visualization
- MSE, RMSE, and R² evaluation metrics
- Model coefficient table
- Dataset explorer with CSV download
- Persisted model bundle containing the estimator and feature order
- Automated tests and GitHub Actions continuous integration
- Streamlit Community Cloud and Docker deployment options

## Model summary

| Item | Value |
|---|---|
| Algorithm | Ordinary least-squares linear regression |
| Dataset | California Housing dataset from scikit-learn |
| Target | `MedHouseVal`, measured in units of $100,000 |
| Features | 8 numerical census-block attributes |
| Split | 60% training / 40% testing |
| Random state | 42 |
| Test R² | Approximately 0.594 |
| Test RMSE | Approximately $73,729 |

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/forva2025/Carfornia_House_Prediction_model.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The first run needs internet access so scikit-learn can download the dataset. The trained model is saved to `models/house_price_pipeline.joblib`.

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
│   └── config.toml
├── docs/
│   ├── DEPLOYMENT.md
│   └── MODEL_CARD.md
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── modeling.py
├── tests/
│   └── test_modeling.py
├── .dockerignore
├── .gitignore
├── app.py
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Deployment

The simplest deployment target is Streamlit Community Cloud:

1. Push this folder to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create an app and select the repository.
4. Set the entry point to `app.py`.
5. Deploy.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Streamlit Cloud, Docker, and Render instructions.

## Limitations

- The underlying data describes California census blocks rather than individual houses.
- The observations are derived from the 1990 U.S. census and do not reflect current prices.
- Linear regression cannot capture all nonlinear relationships in the housing market.
- Predictions outside the training distribution can be unreliable.
- The target variable is capped in the source dataset.

## Possible improvements

- Add cross-validation and hyperparameter comparison
- Compare Ridge, Lasso, Random Forest, and gradient-boosting models
- Add feature engineering and transformation for skewed variables
- Report prediction intervals or uncertainty estimates
- Add automated data-drift and model-quality monitoring

## License

Released under the [MIT License](LICENSE).
