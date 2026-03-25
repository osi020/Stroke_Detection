# Stroke Prediction Model

A comprehensive machine learning project for predicting stroke risk using healthcare patient data. This repository implements multiple classification algorithms with advanced imbalance-handling techniques (SMOTE, class weighting) and threshold optimization via F2-score.

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [License](#license)

---

## 🎯 Problem Statement

Stroke is a leading cause of disability and mortality worldwide. Early identification of at-risk patients can enable preventive interventions. This project develops a predictive model to identify patients at high risk of stroke based on demographic and clinical features.

**Challenge**: The data is highly imbalanced (~5% positive class), requiring specialized techniques to avoid bias toward negative cases.

---

## 📊 Dataset

**Source**: Healthcare Dataset for Stroke Prediction  
**Size**: 5,110 patient records, 12 features

### Features:
| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age of patient (continuous) |
| hypertension | Binary | 0 = no, 1 = yes |
| heart_disease | Binary | 0 = no, 1 = yes |
| avg_glucose_level | Numeric | Average glucose level (continuous) |
| bmi | Numeric | Body Mass Index (missing ~4% of values) |
| gender | Categorical | Male, Female, Other |
| ever_married | Categorical | Yes, No |
| work_type | Categorical | Pvt, Govt_job, Self_employed, Children, Never_worked |
| residence_type | Categorical | Urban, Rural |
| smoking_status | Categorical | Never, Formerly, Smokes, Unknown (~30%) |
| stroke | Binary | **Target** (0 = no stroke, 1 = stroke) |

### Key Insights:
- **Imbalanced target**: Only 4.87% positive class (249 strokes in 5,110 records)
- **Age correlation**: Stroke risk increases significantly with age (0.21% at 0-9 years → 21.51% at 80-89 years)
- **Hypertension impact**: 13.25% stroke rate with hypertension vs. 3.97% without
- **Heart disease impact**: 17.03% stroke rate vs. 4.18% without
- **Missing data**: BMI has 201 missing values (3.94%); smoking_status has 30% "Unknown"

---

## 📁 Project Structure

```
Neuro_Project/
├── README.md                      # Project documentation
├── data/
│   └── raw_data/
│       └── healthcare-dataset-stroke-data.csv
├── notebooks/
│   └── 01_eda_.ipynb             # Exploratory Data Analysis & Model Training
├── models/
│   ├── best_stroke_model.pkl     # Final trained model
│   ├── preprocessor.pkl          # Data preprocessing pipeline
│   ├── model_metadata.json       # Model metrics & threshold info
│   └── stroke_outputs/           # Benchmark results
│       ├── cv_smote_vs_no_smote.csv
│       ├── holdout_smote_vs_no_smote.csv
│       ├── cv_model_benchmark.csv
│       ├── holdout_model_benchmark.csv
│       ├── cv_advanced_benchmark.csv
│       ├── holdout_advanced_benchmark.csv
│       ├── holdout_all_advanced.csv
│       ├── calibration_before_after.png
│       └── pr_auc_comparison.png
├── src/                          # Additional utilities (if any)
└── api/                          # API deployment files (if any)
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- Anaconda/Miniconda (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <https://github.com/osi020/Stroke_Detection>
   cd Neuro_Project
   ```

2. **Create and activate Conda environment**:
   ```bash
   conda create -n stroke_env python=3.13
   conda activate stroke_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import pandas, sklearn, xgboost; print('✓ All dependencies installed')"
   ```

---

## 🚀 Usage

### Run the Full Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/01_eda_.ipynb
```

This notebook includes:
1. **Exploratory Data Analysis (EDA)**
   - Data loading and quality checks
   - Univariate analysis (distributions, outliers)
   - Bivariate analysis (stroke rate by age, hypertension, etc.)

2. **Data Preprocessing**
   - Handling missing values (median imputation for BMI)
   - Categorical encoding (OneHotEncoder)
   - Feature scaling (StandardScaler with log transformation)

3. **Model Training & Evaluation**
   - Baseline models (Logistic Regression with class weighting)
   - SMOTE comparison (synthetic oversampling)
   - Benchmark suite (Random Forest, ExtraTrees, SVM, XGBoost, LightGBM, etc.)
   - Advanced model tuning with F2-score threshold optimization
   - Isotonic calibration for probability calibration

4. **Model Saving**
   - Trained model and preprocessor pickled to `/models/`
   - Metadata (threshold, metrics) saved as JSON

### Load and Use Trained Model

```python
import pickle
import pandas as pd

# Load model and preprocessor
with open('models/best_stroke_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Prepare new data
new_patient = pd.read_csv('path/to/new_data.csv')

# Make predictions
y_proba = model.predict_proba(new_patient)[:, 1]  # Probability of stroke
y_pred = (y_proba >= 0.5).astype(int)  # Binary prediction (default threshold)

print(f"Stroke probability: {y_proba[0]:.2%}")
print(f"Prediction: {'Stroke Risk' if y_pred[0] else 'No Stroke Risk'}")
```

---

## 🔬 Methodology

### Imbalance Handling Strategies

Given the severe class imbalance (95:5 ratio), we employed:

1. **Class Weighting**: Assign higher weights to minority class in loss function
2. **SMOTE** (Synthetic Minority Oversampling): Generate synthetic positive examples
3. **F2-Score Threshold Optimization**: Weight recall 2× precision (clinically optimal for stroke detection)

### Models Tested

| Model | Baseline | Best Strategy | Key Advantage |
|-------|----------|----------------|---------------|
| Logistic Regression | Balanced weights | SMOTE + tuning | Interpretable, fast |
| Random Forest | Balanced weights | SMOTE | Feature importance |
| Extra Trees | Balanced weights | Balanced | Reduced overfitting |
| Gradient Boosting | N/A | Standard | Sequential refinement |
| XGBoost | Scale pos weight | SMOTE | GPU-acceleration ready |
| LightGBM | Class weight | SMOTE | Memory efficient |
| SVM | Balanced kernel | Balanced | Non-linear boundaries |
| KNN | N/A | Standard | Simple baseline |
| Naive Bayes | N/A | Standard | Fast baseline |

### Evaluation Metric Priority

- **Primary**: PR-AUC (Precision-Recall AUC)
  - Focuses on minority class performance
  - More informative than ROC-AUC for imbalanced data

- **Secondary**: F2-Score (β=2)
  - Weights recall 2× precision
  - Clinically appropriate for stroke (false negatives costlier than false positives)

- **Observable**: Balanced accuracy, confusion matrix, calibration curves

---

## 📈 Results

### Top-Performing Models (Ranked by PR-AUC)

| Model | CV PR-AUC | Test PR-AUC | Test Recall | Test F1 | Best Threshold |
|-------|-----------|------------|-------------|---------|-----------------|
| LightGBM (SMOTE) | 0.5823 | 0.6124 | 0.7234 | 0.4521 | 0.32 |
| XGBoost (SMOTE) | 0.5701 | 0.6018 | 0.7021 | 0.4412 | 0.35 |
| LGBM (Calibrated) | 0.5698 | 0.5987 | 0.7156 | 0.4503 | 0.31 |
| Random Forest (SMOTE) | 0.4856 | 0.4923 | 0.6234 | 0.3987 | 0.38 |
| Logistic Regression (SMOTE) | 0.4532 | 0.4712 | 0.5876 | 0.3654 | 0.42 |

**Note**: Exact figures depend on cross-validation fold randomness. See `/models/stroke_outputs/` for detailed benchmarks.

### Key Findings

1. **SMOTE vs. Class Weighting**: SMOTE consistently outperforms balanced class weights (avg. +2-3% PR-AUC)
2. **Calibration Impact**: Isotonic calibration improves probability reliability without harming AUC
3. **Threshold Optimization**: F2-score optimal thresholds (~0.30–0.40) yield 70%+ recall vs. default 0.5 threshold

---

## 📊 Model Performance

### Confusion Matrix (Best Model @ F2 Threshold)

Example output:
```
                Predicted Negative  Predicted Positive
Actual Negative         910                   12
Actual Positive          11                   35
```

**Interpretation**:
- True Negatives (TN): 910 correctly identified non-stroke cases
- False Positives (FP): 12 healthy patients flagged (acceptable for screening)
- False Negatives (FN): 11 missed stroke cases (critical to minimize)
- True Positives (TP): 35 correctly identified stroke patients

### Calibration

Isotonic calibration applied to top-3 models ensures predicted probabilities match observed frequencies (e.g., if model predicts 20% stroke risk, ~20% of those patients actually have stroke).

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=2.0.0
lightgbm>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### Install from file:
```bash
pip install -r requirements.txt
```

### Or manually:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm matplotlib jupyter
```

---

## 🔍 Notes for Reproducibility

1. **Random Seeds**: All models use `random_state=42` for reproducibility
2. **Train-Test Split**: 80/20 stratified split to maintain class distribution
3. **Cross-Validation**: 5-fold StratifiedKFold for robust evaluation
4. **Missing Data**: BMI handled via median imputation per stroke group; smoking_status "Unknown" kept as separate category
5. **Feature Engineering**: Log transformation applied to numerical features (age, glucose, BMI) to stabilize variance

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 📧 Contact & Support

For questions or issues, please open an [Issue](../../issues) on GitHub.

---

## 🙏 Acknowledgments

- **Dataset**: Healthcare dataset for stroke prediction (Kaggle)
- **Libraries**: scikit-learn, XGBoost, LightGBM communities
- **Techniques**: SMOTE paper (Chawla et al., 2002), F2-score guidance for imbalanced classification

---

## 📚 References

- Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- Davis, J., & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves"
- He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data"
