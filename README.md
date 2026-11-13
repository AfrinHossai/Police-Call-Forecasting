# Police Call Forecasting

### Next-Hour Police Call Volume Prediction Using Machine Learning

**CS 171 Machine Learning Project | San José State University**

## Project Overview

This project develops and evaluates machine learning models for predicting the number of police calls for service expected in San José during the next hour.

Using publicly available police call records from the City of San José, the project transforms incident-level data into hourly observations, engineers historical and temporal features, and compares traditional machine learning and neural network models.

The forecasting pipeline uses chronological training, validation, and test splits to evaluate predictive performance while reducing the risk of temporal data leakage.

The project emphasizes predictive modeling, time-series feature engineering, model comparison, and reproducible evaluation.

## Project Highlights

- **Prediction task:** Next-hour police call volume forecasting
- **Problem type:** Supervised regression
- **Data source:** City of San José Police Calls for Service
- **Data coverage:** 2024, 2025, and partial 2026
- **Hourly observations:** 21,236
- **Numerical predictors:** 48
- **Models evaluated:** Eight modeling approaches, including baselines
- **Evaluation metrics:** MAE, RMSE, and R²
- **Best reported model:** Tuned Random Forest

### Best Reported Test Performance

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 5.616 calls |
| Root Mean Squared Error (RMSE) | 8.142 calls |
| R² Score | 0.711 |

These results are from the project's documented held-out test evaluation. They may vary slightly if the data, code, saved models, or software environment changes.

## Technologies Used

**Programming Language**
- Python

**Data Processing and Analysis**
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- Linear Regression
- Elastic Net
- Decision Tree
- Random Forest

**Deep Learning**
- TensorFlow / Keras
- Multilayer Perceptron (MLP)
- Long Short-Term Memory (LSTM)

**Visualization**
- Matplotlib
- Seaborn
- Great Tables

**Development Environment**
- Google Colab
- Google Drive
- Jupyter Notebooks
- Git and GitHub

## Machine Learning Models

The project compares eight forecasting approaches.

| Model | Purpose |
|---|---|
| Mean Baseline | Predicts using the training-set mean |
| Persistence Baseline | Uses recent observed call volume |
| Linear Regression | Establishes a linear predictive model |
| Elastic Net | Applies regularization to linear regression |
| Decision Tree | Learns nonlinear decision rules |
| Random Forest | Combines multiple decision trees |
| MLP | Models nonlinear feature relationships |
| LSTM | Learns from sequences of historical observations |

Hyperparameters and LSTM lookback windows are selected using validation data.

The held-out test set is used for final evaluation rather than hyperparameter selection.

## Dataset

### Data Source

The project uses publicly available **Police Calls for Service** records from the City of San José Open Data Portal.

**Source:** [City of San José Police Calls for Service](https://data.sanjoseca.gov/dataset/police-calls-for-service)

### Data Coverage

The source data includes:

- 2024 police call records
- 2025 police call records
- Partial 2026 police call records

The source records contain fields such as:

- `OFFENSE_DATE`
- `OFFENSE_TIME`
- `PRIORITY`
- `CALL_TYPE`

### Data Preparation

The project:

1. Loads and combines yearly incident records.
2. Removes exact duplicate records.
3. Constructs timestamps from date and time information.
4. Sorts records chronologically.
5. Aggregates incidents into hourly call counts.
6. Creates the next-hour forecasting target.
7. Engineers calendar, lag, rolling-average, priority, and call-type features.
8. Assigns chronological training, validation, and test periods.
9. Removes observations without complete required historical features.

### Prediction Target

`NEXT_HOUR_CALLS`

This variable represents the number of police calls expected during the following hour.

The final model-ready dataset contains 21,236 hourly observations and 48 numerical predictors.

**Note:** Raw source CSV files are not included in the GitHub repository because of their size.

## Repository Structure

```text
Police-Call-Forecasting/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_and_split.ipynb
│   ├── 04_baseline_and_tree_models.ipynb
│   ├── 05_neural_networks.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── README.md
│
├── models/
│   ├── LinearRegression/
│   ├── ElasticNet (Regularization)/
│   ├── DecisionTree/
│   ├── RandomForest/
│   ├── MLP/
│   └── LSTM/
│
├── figures/
│   └── README.md
│
├── results/
│   ├── Baseline + Simple Model Results/
│   ├── split_assignments.csv
│   ├── split_procedure.json
│   ├── split_summary.csv
│   ├── target_distribution_by_split.csv
│   └── README.md
│
├── Results_Graphs/
│   ├── master_model_comparison.csv
│   ├── master_model_forecast_comparison.png
│   ├── rf_feature_importance.png
│   └── lstm_lookback_comparison.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

The repository also includes trained model artifacts, diagnostic figures, prediction files, and additional evaluation outputs within the corresponding directories.

## Project Workflow

The machine learning workflow is organized into six notebooks.

### Notebook 01 — Data Preparation

**File:** `notebooks/01_data_preparation.ipynb`

This notebook:

- Loads and combines yearly police call records.
- Removes duplicate records.
- Creates timestamps.
- Aggregates incidents into hourly observations.
- Constructs the next-hour prediction target.
- Engineers historical and temporal features.
- Assigns chronological dataset splits.

Important generated files include:

```text
model_ready_hourly.csv
candidate_features.json
final_split_audit.csv
```

### Notebook 02 — Exploratory Data Analysis

**File:** `notebooks/02_eda.ipynb`

Explores:

- Distribution of hourly police call counts
- Hour-of-day patterns
- Day-of-week patterns
- Monthly and yearly patterns
- Priority distributions
- Call-type distributions
- Feature correlations
- High-volume and extreme-demand periods
- Distribution changes across dataset splits

The exploratory visualizations are saved in the `figures/` directory.

### Notebook 03 — Preprocessing and Split Export

**File:** `notebooks/03_preprocessing_and_split.ipynb`

This notebook:

- Uses the chronological dataset splits.
- Separates predictors from the target.
- Fits median imputation using the training set.
- Fits standardization using the training set.
- Applies the fitted preprocessing pipeline to validation and test data.
- Saves scaled and unscaled dataset versions.
- Preserves the ordered feature list.

Important generated files include:

```text
train_scaled.csv
validation_scaled.csv
test_scaled.csv

train_unscaled.csv
validation_unscaled.csv
test_unscaled.csv

numerical_preprocessor.joblib
preprocessing_metadata.json
candidate_features.json
```

### Notebook 04 — Baselines and Traditional Machine Learning

**File:** `notebooks/04_baseline_and_tree_models.ipynb`

Trains and evaluates:

- Mean Baseline
- Persistence Baseline
- Linear Regression
- Elastic Net
- Decision Tree
- Random Forest

Hyperparameters are selected using validation performance.

Saved models include:

```text
models/LinearRegression/linear_regression.joblib
models/ElasticNet (Regularization)/elastic_net.joblib
models/DecisionTree/decision_tree_.joblib
models/RandomForest/random_forest.joblib
```

The test set should not be used for model tuning.

### Notebook 05 — Neural Network Models

**File:** `notebooks/05_neural_networks.ipynb`

This notebook trains:

- MLP with 100 epochs
- MLP with early stopping
- LSTM models with lookback windows of 2, 4, 6, 12, 24, and 48 hours

The LSTM lookback window is selected using validation performance.

The documented selected value is:

```python
best_ts = 6
```

Saved neural network models include:

```text
models/MLP/mlp_model_epoch100.keras
models/MLP/mlp_model_es.keras

models/LSTM/lstm_model_timestep_2.keras
models/LSTM/lstm_model_timestep_4.keras
models/LSTM/lstm_model_timestep_6.keras
models/LSTM/lstm_model_timestep_12.keras
models/LSTM/lstm_model_timestep_24.keras
models/LSTM/lstm_model_timestep_48.keras
```

Training histories are also saved as CSV files.

### Notebook 06 — Final Model Evaluation

**File:** `notebooks/06_model_evaluation.ipynb`

This notebook:

- Loads saved traditional and neural network models.
- Preserves the feature order used during training.
- Aligns predictions to comparable test timestamps.
- Accounts for the selected LSTM lookback.
- Evaluates each model using held-out test data.
- Computes MAE, MSE, RMSE, and R².
- Creates model comparison and diagnostic visualizations.

The documented selected LSTM lookback is:

```python
BEST_TIME_STEP = 6
```

The evaluation includes both baseline models for comparison.

Final results are stored in `Results_Graphs/`.

## Training, Validation, and Test Splits

The project uses chronological data splits to preserve temporal order.

| Split | Observations | Period |
|---|---:|---|
| Training | 14,830 | Jan. 8, 2024 – Sep. 30, 2025 |
| Validation | 3,624 | Oct. 1, 2025 – Feb. 28, 2026 |
| Testing | 2,782 | Mar. 1, 2026 – Jul. 8, 2026 |
| **Total** | **21,236** | |

### Evaluation Procedure

- **Training set:** Fit model parameters.
- **Validation set:** Select hyperparameters and LSTM lookback.
- **Test set:** Evaluate final model performance.

The test set should not be used to select model configurations.

This separation helps reduce optimistic performance estimates caused by test-set leakage.

## Model Evaluation Results

The project evaluates predictive accuracy using:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predicted and actual call counts.

**Root Mean Squared Error (RMSE)**

Measures prediction error while giving greater weight to larger errors.

**R² Score**

Measures the proportion of variability in call volume explained by the model relative to a mean-prediction reference.

### Best Reported Model: Tuned Random Forest

| Metric | Held-Out Test Result |
|---|---:|
| MAE | 5.616 |
| RMSE | 8.142 |
| R² | 0.711 |

The tuned random forest achieved the strongest reported performance in the project's model comparison.

These are the documented results of this experiment, not a guarantee of performance on future operational data.

### Model Comparison Files

The repository includes:

- [Master model comparison data](Results_Graphs/master_model_comparison.csv)
- [Forecast comparison visualization](Results_Graphs/master_model_forecast_comparison.png)
- [Random Forest feature importance](Results_Graphs/rf_feature_importance.png)
- [LSTM lookback comparison](Results_Graphs/lstm_lookback_comparison.png)

## Selected Visualizations

### Model Forecast Comparison

The following figure compares model predictions with observed police call volume.

![Master Model Forecast Comparison](Results_Graphs/master_model_forecast_comparison.png)

### Random Forest Feature Importance

This visualization shows the relative importance of features in the fitted random forest model.

![Random Forest Feature Importance](Results_Graphs/rf_feature_importance.png)

Feature importance describes the model's predictive behavior and should not be interpreted as causal evidence.

### LSTM Lookback Comparison

This figure summarizes results for the evaluated LSTM lookback windows.

![LSTM Lookback Comparison](Results_Graphs/lstm_lookback_comparison.png)

### Hourly Police Call Patterns

This visualization illustrates average police call volume by hour of day.

![Average Calls by Hour](figures/average_calls_by_hour.png)

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/AfrinHossai/Police-Call-Forecasting.git
```

### 2. Navigate to the Repository

```bash
cd Police-Call-Forecasting
```

### 3. Install Required Packages

For a compatible Python environment, install the required packages using:

```bash
python3 -m pip install -r requirements.txt
```

The notebooks were developed in Google Colab. The environment may require additional configuration depending on package availability and runtime versions.

### 4. Download Raw Data

Download the Police Calls for Service files from:

[City of San José Open Data Portal](https://data.sanjoseca.gov/dataset/police-calls-for-service)

Rename the files:

```text
police_calls_2024.csv
police_calls_2025.csv
police_calls_2026.csv
```

The files must retain the source columns expected by the notebooks.

### 5. Prepare Google Drive

The notebooks use the following Google Drive project directory:

```text
MyDrive/CS171_Police_Call_Project/
```

Create this structure:

```text
CS171_Police_Call_Project/
│
├── data_raw/
│   ├── police_calls_2024.csv
│   ├── police_calls_2025.csv
│   └── police_calls_2026.csv
│
├── data_processed/
├── figures/
├── results/
├── Results_Graphs/
└── models/
    ├── LinearRegression/
    ├── ElasticNet (Regularization)/
    ├── DecisionTree/
    ├── RandomForest/
    ├── MLP/
    └── LSTM/
```

These Google Drive folders are part of the original notebook workflow. Their names do not need to match the GitHub repository name.

Notebook 04 also uses `MyDrive/Results` for some figures, so that directory may be required.

The following cell creates the required folders:

```python
from pathlib import Path

project_folder = Path("/content/drive/MyDrive/CS171_Police_Call_Project")

required_folders = [
    "data_raw",
    "data_processed",
    "figures",
    "results",
    "Results_Graphs",
    "models/LinearRegression",
    "models/ElasticNet (Regularization)",
    "models/DecisionTree",
    "models/RandomForest",
    "models/MLP",
    "models/LSTM",
]

for folder in required_folders:
    (project_folder / folder).mkdir(parents=True, exist_ok=True)

Path("/content/drive/MyDrive/Results").mkdir(parents=True, exist_ok=True)
```

### 6. Open the Notebooks in Google Colab

Open the notebook files from the repository's `notebooks/` directory.

Mount Google Drive when required:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Run the notebooks in the following order:

```text
01_data_preparation.ipynb
02_eda.ipynb
03_preprocessing_and_split.ipynb
04_baseline_and_tree_models.ipynb
05_neural_networks.ipynb
06_model_evaluation.ipynb
```

Later notebooks depend on processed data or saved models generated by earlier notebooks.

### 7. Configure the Neural Network Runtime

For Notebook 05, a GPU runtime is recommended.

In Google Colab:

```text
Runtime → Change runtime type → T4 GPU
```

A GPU can improve training speed, depending on runtime availability and workload characteristics.

### 8. Check LSTM Evaluation Settings

Before running Notebook 06, confirm that the selected LSTM lookback matches the validation-selected model.

For the documented experiment:

```python
BEST_TIME_STEP = 6
```

Also ensure the ordered feature list from `candidate_features.json` is used during evaluation.

**Important:** The project relies on saved files and Google Drive directories. Cloning the repository alone does not automatically reproduce the entire pipeline.

## Reproducibility Notes

The project uses a fixed chronological data split to support consistent model comparisons.

Numerical preprocessing is fitted using training data before being applied to validation and test sets.

The notebooks use random seed 42 and deterministic TensorFlow settings where possible.

However, exact results may vary because of:

- Package versions
- TensorFlow or Keras versions
- Hardware and runtime differences
- Source dataset updates
- Changes to preprocessing or model configuration
- Nondeterministic operations

The repository includes saved models, figures, and evaluation artifacts, but complete reproducibility has not been independently verified in a clean environment.

## Troubleshooting

### FileNotFoundError

Check that:

- Google Drive is mounted.
- The project directory is named `CS171_Police_Call_Project`.
- Raw CSV filenames match the expected names.
- Required model and output directories exist.
- Previous notebooks have produced the expected files.

### ModuleNotFoundError: great_tables

In Google Colab, run:

```python
!pip install -q great-tables
```

Restart the runtime if necessary.

### Feature-Name or Feature-Count Errors

Use the ordered feature list stored in `candidate_features.json`.

Avoid manually selecting an incompatible set or order of predictors during model evaluation.

### LSTM Input-Shape Errors

Confirm that:

- The selected lookback matches the saved model.
- Input sequences use the expected number of time steps.
- Predictions and target values are aligned to the same timestamps.

### Different Results After Rerunning

Results can change when package versions, datasets, saved models, or runtime configurations differ.

## Project Limitations

- The dataset covers a specific geographic area and historical period.
- Forecasting accuracy may decrease under changing demand patterns.
- High-volume or unusual incidents may be more difficult to predict accurately.
- Model performance may depend on feature construction and selected hyperparameters.
- Some historical features may require careful alignment for real-time forecasting.
- The project does not establish operational reliability for staffing or deployment decisions.
- The forecasting pipeline has not been independently validated for production use.

## Future Improvements

- Evaluate the models on additional future time periods.
- Investigate performance under seasonal and distribution shifts.
- Explore additional temporal and weather-related features.
- Compare against dedicated statistical time-series forecasting methods.
- Add automated testing for data processing and model evaluation.
- Implement rolling-origin or walk-forward validation.
- Improve reproducibility through pinned dependency versions.
- Develop a simple interactive forecasting dashboard.
- Investigate prediction intervals and uncertainty estimates.

## Team

- Afrin Hossain
- Anushka Chandrashekar
- Sampiya Bhusal

## Data Source and Acknowledgment

**City of San José — Police Calls for Service**

Data source: [City of San José Open Data Portal](https://data.sanjoseca.gov/dataset/police-calls-for-service)

This project was completed as part of CS 171 coursework at San José State University.

## Disclaimer

This project is intended for academic research and educational purposes.

The forecasts estimate short-term citywide police call demand and should not be treated as exact staffing, deployment, or public-safety decisions.

The predictive models are not intended for identifying individuals, making enforcement decisions, or replacing professional operational judgment.
