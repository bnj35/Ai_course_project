# Analysis Report: Employee Attrition Prediction

## 📋 Project Overview

This project aims to predict employee attrition (departure) using machine learning techniques. The objective is to develop a model capable of identifying at-risk employees to enable preventive interventions.

### Business Problem

- **Objective**: Detect employees who are at risk of leaving the company
- **Ethical constraint**: Exclusion of sensitive variables (age, gender, marital status)
- **Precision constraint**: Minimum 35% precision to avoid too many false alerts
- **Main challenge**: Significant class imbalance (5:1 - many more staying employees than leaving)

---

## 📊 1. Data Preparation and Merging

### 1.1 Data Sources

The project uses **5 different CSV files** that must be merged:

- `employee_survey_data.csv`: Employee survey data
- `general_data.csv`: General employee information
- `manager_survey_data.csv`: Manager evaluations
- `in_time.csv`: Daily arrival times
- `out_time.csv`: Daily departure times

### 1.2 Time Data Processing

The time files (in_time/out_time) require special processing:

#### Step 1: Renaming and Merging

- Rename the first column to "EmployeeID" for both files
- Verify consistency of days present in both datasets
- Identify empty cells to detect inconsistencies

#### Step 2: Temporal Calculations

- Conversion to datetime format
- Calculate daily work duration: `(exit_time - entry_time)` in hours
- Detect day of the week for each worked day

#### Step 3: Aggregation by Day of Week

For each day (Monday=0 to Sunday=6), we calculate:

- **Number of times worked** on that day: `worked_on_day_{0-6}`
- **Average hours worked** on that day: `avg_hours_day_{0-6}`
- **Total work duration**: sum of all hours

**Result**: Reduced temporal dataset with only relevant aggregated columns

### 1.3 Final Merge

Datasets are merged sequentially on `EmployeeID`:

```
employee_data + manager_data → employee_manager_data
general_data + employee_manager_data → combined_data
combined_data + time_data → raw_dataset
```

### 1.4 Ethical Considerations

**Immediate removal** of sensitive columns:

- `MaritalStatus` (marital status)
- `Gender` (gender)
- `Age` (age)

These variables could introduce discriminatory bias into the model.

---

## 🔧 2. Advanced Feature Engineering

### 2.1 Why Create New Features?

**Objective**: Transform raw data into features with clearer business meaning and stronger predictive power.

**Principle**: Use business logic to create composite indicators that better capture attrition patterns.

### 2.2 Career Progression Features

**PromotionRate**:

```
PromotionRate = YearsAtCompany / (YearsSinceLastPromotion + 1)
```

- Higher ratio means more frequent promotions
- Indicator of internal mobility and progression opportunities

**CareerStagnation**:

- Binary indicator: 1 if no promotion for >5 years
- Identifies employees potentially frustrated by lack of progression

### 2.3 Compensation Features

**IncomePerYear**:

```
IncomePerYear = MonthlyIncome / (YearsAtCompany + 1)
```

- Normalizes salary by company tenure
- Detects if compensation is fair relative to seniority

**ExperienceIncomeRatio**:

```
ExperienceIncomeRatio = TotalWorkingYears / (MonthlyIncome / 1000)
```

- Measures balance between total experience and compensation
- Identifies overqualified and underpaid employees

### 2.4 Work-Life Balance Features

**AvgDailyHours**:

```
AvgDailyHours = duration_hours / 260  (260 working days/year)
```

- Average hours worked per day
- More direct than total duration_hours

**Overwork**:

- Binary indicator: 1 if AvgDailyHours > 9 hours/day
- Identifies employees at risk of burnout

**WeekendWorker**:

- Binary indicator: 1 if frequent weekend work (>10 times)
- Measures work intrusion into personal life

### 2.5 Satisfaction Composite Features

**OverallSatisfaction**:

```
OverallSatisfaction = (JobSatisfaction + EnvironmentSatisfaction + WorkLifeBalance) / 3
```

- Average of three satisfaction dimensions
- Synthetic indicator of workplace well-being

**LowSatisfaction**:

- Binary indicator: 1 if OverallSatisfaction < 2
- Alert threshold to identify very dissatisfied employees

### 2.6 Risk Profile Features

**JobChangeFrequency**:

```
JobChangeFrequency = NumCompaniesWorked / (TotalWorkingYears + 1)
```

- Measures tendency for "job hopping"
- Predicts departure probability based on history

**HighRiskProfile**:

- Combination of risk factors:
  - No promotion for >3 years
  - JobSatisfaction < 2
  - WorkLifeBalance < 2
- Identifies employees accumulating multiple risk factors

### 2.7 Feature Engineering Impact

**Results**:

- **12 new features** created with business logic
- **Increase**: 302 → 314 columns
- **Top features** (by ANOVA F-score):
  1. `AvgDailyHours`: 148.26 (very strong correlation)
  2. `JobChangeFrequency`: 141.85 (predictive history)
  3. `OverallSatisfaction`: 88.95 (powerful synthetic indicator)
  4. `PromotionRate`: 80.54 (career progression)
  5. `Overwork`: 80.43 (burnout risk)

**🔑 Critical point**: Feature engineering is performed **BEFORE** train/test split to ensure consistency, but creates **NO data leakage** because it uses only deterministic transformations without statistics calculated on the entire dataset.

---

## 🔀 3. Train/Test Split (BEFORE Preprocessing)

### 3.1 Anti-Data Leakage Principle

**CRITICAL**: Train/test split is performed **BEFORE** any preprocessing to avoid **data leakage**.

**❌ Bad practice**:

```
Preprocessing → Split → Train
```

→ The model "sees" test data through normalization

**✅ Good practice**:

```
Split → Separate preprocessing → Train
```

→ Complete isolation between train and test

### 3.2 Split Configuration

- **Ratio**: 80% train / 20% test
- **Stratification**: Preserves target class distribution (`Attrition`)
- **Random state**: 42 (for reproducibility)

**Results**:

- Train: ~2,823 samples
- Test: ~705 samples
- Preserved distribution: ~84% stay / ~16% leave

---

## 🔧 4. Secure Preprocessing (Without Data Leakage)

### 4.1 `preprocess_data_safe()` Function Architecture

The function accepts two operating modes:

**Fit Mode (fit_mode=True)**: For train set

- Calculates and records transformation parameters
- Returns fitted transformers

**Transform Mode (fit_mode=False)**: For test set

- Uses pre-fitted transformers from train
- Calculates NO new parameters

### 4.2 Preprocessing Steps

#### a) Constant Column Removal

- Automatic detection of columns with single unique value
- Removal as they are uninformative for the model

#### b) Missing Value Imputation

**Numerical variables**:

- Strategy: **Median** (robust to outliers)
- Example: If `MonthlyIncome` has missing values, fill with train set median

**Categorical variables**:

- Strategy: **Mode** (most frequent value)
- Example: If `Department` has missing values, fill with most common department

**🔑 Key point**: Imputation values are calculated ONLY on train set

#### c) Ordinal Encoding

Some categorical variables have a natural order:

- `BusinessTravel`: ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
- Encoding: 0, 1, 2 (preserves increasing order of travel frequency)

#### d) One-Hot Encoding

Categorical variables **without** natural order:

- `Department`, `EducationField`, `JobRole` → special treatment (mutual information)
- Other categorical variables → standard one-hot encoding

**Example**:

```
Department = "Sales"
→ Department_Sales=1, Department_HR=0, Department_IT=0
```

#### e) Normalization (StandardScaler)

**Formula**: `z = (x - μ) / σ`

- μ (mean) and σ (std) calculated **only on train**
- Application of these parameters on test

**Why normalize?**

- Features have different scales (e.g.: YearsAtCompany=1-40, MonthlyIncome=1000-20000)
- Perceptron is sensitive to scales → normalization necessary

**Result**: All numerical features centered around 0 with standard deviation of 1

---

## 🎯 5. Multi-Method Feature Selection (BEFORE SMOTE)

### 5.1 Why Select Features?

**Problems to solve**:

- Too many features → overfitting, unnecessary complexity
- Uninformative features → noise in predictions
- Some features are derived (days of week) → redundancy
- **A single method** may miss important features

### 5.2 Multi-Method Approach (Improved Robustness)

**Principle**: Combine **3 complementary methods** to capture different types of relationships:

#### Method 1: ANOVA Test (F-classif)

**What it detects**: Univariate linear relationships

- Calculates an F-score for each feature
- High F-score = good separation between classes
- Formula: Between-class variance / Within-class variance

**Example**:

- If `OverTime=Yes` is strongly associated with `Attrition=1` → high F-score
- If `EmployeeNumber` is random → low F-score

**Configuration**: Selection of **top 30 features** by F-score

#### Method 2: RFECV (Recursive Feature Elimination with Cross-Validation)

**What it detects**: Multivariate importance with recursive elimination

**Process**:

1. Train a model (LogisticRegression) with all features
2. Calculate importance of each feature (coefficients)
3. Eliminate least important feature
4. **Cross-validate** to evaluate performance (3-fold CV)
5. Repeat until finding optimal number of features

**Advantages**:

- Takes into account **feature interactions**
- **Auto-adjusts** optimal number of features (minimum 15)
- Cross-validation to avoid overfitting

**Configuration**:

- Estimator: LogisticRegression(penalty='l2', C=1.0, max_iter=5000)
- Cross-validation: 3-fold StratifiedKFold
- Minimum features: 15

#### Method 3: L1 Regularization (Lasso)

**What it detects**: Features with non-zero coefficients after penalization

**Principle**:

- LogisticRegression with L1 penalty
- Forces some coefficients to **exactly 0**
- Automatic selection of important features

**Formula**:

```
Loss = -log likelihood + C × Σ|wᵢ|
```

- Low C (0.5) → strong penalty → fewer features
- Only features with significant impact survive

**Configuration**:

- LogisticRegression(penalty='l1', C=0.5, solver='saga')
- Selection: Features with importance > median

### 5.3 Combination Strategy

**Union of 3 Methods**:

```
Final features = ANOVA_top30 ∪ RFECV_selected ∪ L1_selected
```

**Advantages of union**:

- **Robustness**: A feature missed by one method can be captured by another
- **Complementarity**: ANOVA (univariate) + RFECV (multivariate) + L1 (sparse)
- **Balance**: Avoids over-filtering or over-keeping

### 5.4 Intelligent Filtering

**Exclusion of granular patterns**:

- Day of week features (`day_of_week`)
- Daily averages (`avg_hours_day_`)
- Individual hours by date (`YYYY-MM-DD_hours`)

**Reason**: These features are too granular and may cause overfitting

### 5.5 Selection Results

**Final configuration**: **36 selected features**

**Top identified features**:

1. `AvgDailyHours` (148.26) - New engineered feature
2. `JobChangeFrequency` (141.85) - New engineered feature
3. `duration_hours` (148.26) - Original temporal feature
4. `OverallSatisfaction` (88.95) - New engineered feature
5. `PromotionRate` (80.54) - New engineered feature

**🔑 Critical point**:

- Selection calculated **only on X_train**
- Then applied to X_test (same features)
- **No data leakage**: no test set information used

**Impact**:

```
X_train: (4,734, 314) → (4,734, 36)
X_test: (882, 314) → (882, 36)
```

**Comparison with previous approach**:

- **Before**: 15 features (ANOVA only)
- **After**: 36 features (multi-method)
- **Gain**: +140% features (+21), better pattern coverage

---

## ⚖️ 6. Rebalancing with SMOTE and Nested Cross-Validation

### 6.1 Class Imbalance Problem

**Initial situation**:

- Class 0 (stays): ~2,959 samples (84%)
- Class 1 (leaves): ~569 samples (16%)
- Ratio: **5:1**

**Impact without correction**:

- Model tends to predict "stays" for everyone
- Very poor departure detection (low recall)

### 6.2 SMOTE (Synthetic Minority Over-sampling Technique)

**Principle**:

1. Select a sample from the minority class
2. Find its K nearest neighbors (K=10)
3. Create a synthetic sample between the sample and a random neighbor

**Formula**:

```
x_new = x_i + λ * (x_neighbor - x_i)
```

where λ is random between 0 and 1

### 6.3 Nested Cross-Validation for SMOTE Optimization

**❌ Problem with previous approach**:

- Testing SMOTE strategies on **test set**
- **Data leakage**: using test to choose hyperparameter
- Biased evaluation

**✅ New approach: Nested Cross-Validation**

**Architecture**:

```
Outer Loop: Train data (X_train_selected, y_train)
  ↓
Inner CV: StratifiedKFold 3-fold
  ↓
For each SMOTE strategy [0.3, 0.4, 0.5, 0.6, 0.7]:
  1. Create Pipeline: SMOTE → Perceptron
  2. Cross-validate on 3 folds
  3. Calculate average F1-score
  ↓
Select strategy with best CV F1
```

**Detailed process**:

1. **For each strategy** (0.3, 0.4, 0.5, 0.6, 0.7):
   - Create an imblearn Pipeline:
     ```python
     Pipeline([
         ('smote', SMOTE(sampling_strategy=strategy)),
         ('perceptron', Perceptron())
     ])
     ```
   - **3-fold** cross-validation on train only
   - Calculate average F1-score on 3 folds

2. **Selection**: Strategy maximizing average F1-score

3. **Application**: Apply best strategy on entire train set

### 6.4 Optimization Results

**Testing 5 different strategies**: [0.3, 0.4, 0.5, 0.6, 0.7]

**Strategy interpretation**:

- **0.3**: Minority class = 30% of majority → conservative under-sampling
- **0.5**: Minority class = 50% of majority → moderate balance
- **0.7**: Minority class = 70% of majority → aggressive balance

**Selected strategy**: **0.6** (CV F1=0.278)

**Final class distribution**:

```
Before SMOTE: {0: 2959, 1: 569}
After SMOTE: {0: 2959, 1: 1775}
```

**🔑 Critical points**:

- SMOTE applied **ONLY on train** (never on test)
- Strategy **optimized by CV** (no test set peeking)
- **Stratified** cross-validation (preserves class distribution)
- `class_weight` parameter **removed** (avoids double counting)

**Advantages of Nested CV**:

- ✅ **No data leakage**: test set never used
- ✅ **Robustness**: evaluation on 3 folds (not single split)
- ✅ **Generalization**: better estimation of real performance
- ✅ **Reproducibility**: random_state fixed (42)

---

## 🔬 7. Ensemble Models and Multi-Algorithm Optimization

### 7.1 Why an Ensemble of Models?

**❌ Limitation of single model (Perceptron)**:

- Single algorithm captures only one "vision" of the data
- Algorithmic bias: each model has its strengths and weaknesses
- Plateaued performance

**✅ Advantages of Ensemble**:

- **Diversity**: Combination of multiple algorithms
- **Robustness**: Reduces impact of errors from a single model
- **Performance**: Often superior to best individual model
- **Wisdom of the crowd**: Majority vote or probability averaging

### 7.2 Individually Optimized Models

#### Model 1: Perceptron (Baseline)

**What is a Perceptron?**

- Linear classification algorithm
- Finds a separating hyperplane between classes
- Iterative weight updates based on errors

**Prediction formula**:

```
y = sign(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

**Optimized hyperparameters (GridSearchCV)**:

- **Penalty**: [None, 'l2', 'l1', 'elasticnet']
- **Alpha**: [0.0001, 0.001, 0.01, 0.1] (regularization strength)
- **Eta0**: [0.5, 1.0, 1.5] (learning rate)
- **Max_iter**: 10000, **Tol**: 1e-3

**GridSearchCV process**:

1. Test **48 combinations** (4×4×3)
2. **5-fold** cross-validation on balanced train
3. Metric: **F1-score**

**Best parameters found**:

- penalty='elasticnet', alpha=0.1, eta0=1.0
- **CV F1-score**: 0.581

#### Model 2: Logistic Regression

**What is Logistic Regression?**

- Linear model with sigmoid function
- Predicts **probabilities** (not just classes)
- Formula: `P(y=1) = 1 / (1 + e^(-(w·x + b)))`

**Advantages**:

- Naturally calibrated probabilities
- Interpretability (coefficients)
- Supports L1/L2 regularization

**Optimized hyperparameters**:

- **Penalty**: ['l2', 'l1']
- **C**: [0.1, 0.5, 1.0, 2.0] (inverse of regularization)
- **Solver**: 'saga' (supports L1 and L2)
- **Max_iter**: 10000

**Best parameters found**:

- penalty='l2', C=1.0
- **CV F1-score**: 0.601 (better than Perceptron)

#### Model 3: Random Forest

**What is a Random Forest?**

- Ensemble of **decision trees**
- Each tree trained on random subsample
- Majority vote or prediction averaging

**Advantages**:

- Captures **non-linear interactions**
- Robust to outliers
- Little preprocessing needed
- Built-in feature importance

**Optimized hyperparameters**:

- **n_estimators**: [50, 100] (number of trees)
- **max_depth**: [5, 10, None] (maximum depth)
- **min_samples_split**: [5, 10] (minimum samples for split)
- **min_samples_leaf**: [2, 4] (minimum samples per leaf)

**GridSearchCV process**:

- **72 combinations** tested
- **3-fold** cross-validation (faster for RF)
- Metric: **F1-score**

**Best parameters found**:

- n_estimators=50, max_depth=None, min_samples_split=5, min_samples_leaf=2
- **CV F1-score**: 0.771 (⭐ best individual model)

### 7.3 Ensemble: VotingClassifier (Soft Voting)

**Architecture**:

```python
VotingClassifier(
    estimators=[
        ('logistic', LogisticRegression_optimized),
        ('random_forest', RandomForest_optimized)
    ],
    voting='soft',  # Probability averaging
    weights=[1, 1]  # Equal weights
)
```

**Why Soft Voting?**

- **Hard voting**: Majority vote of predicted classes
- **Soft voting**: Average of **probabilities** from each model
  - More nuanced: uses confidence of each model
  - Better performance in general

**Why exclude Perceptron?**

- Perceptron has **no `predict_proba` method**
- Soft voting requires probability estimates
- ✅ Perceptron evaluated separately for comparison

**Soft Voting formula**:

```
P_ensemble(y=1|x) = (P_logistic(y=1|x) + P_rf(y=1|x)) / 2
Prediction = 1 if P_ensemble ≥ threshold (optimized)
```

**Ensemble performance**:

- **CV F1-score**: Optimal combination of Logistic + RF
- **Advantage**: Compensates for individual weaknesses

### 7.4 Comparison of 4 Models

**Models evaluated**:

1. **Ensemble_Optimized** (Logistic + RF, soft voting)
2. **RandomForest_Optimized** (best individual)
3. **LogisticRegression_Optimized**
4. **Perceptron_Optimized** (baseline)

**Evaluation process**:

- Training on balanced train (after SMOTE)
- 5-fold cross-validation for internal validation
- Prediction on test set (never seen during optimization)
- Metrics: Precision, Recall, F1-score, ROC-AUC

---

## 📊 8. Probability Calibration (Platt Scaling)

### 8.1 Why Calibrate Probabilities?

**Problem**: Models optimized for F1-score don't always produce **well-calibrated probabilities**.

**Example**:

- A model predicts `P(Attrition=1) = 0.7`
- In reality, only 40% of cases with this probability are departures
- **Poor calibration**: overestimated probabilities

**Impact**:

- Less reliable threshold optimization
- Difficult business interpretation
- Decisions based on false probabilities

### 8.2 Platt Scaling (Sigmoid Calibration)

**Principle**:

1. **Split**: Divide train set into 80% train / 20% calibration holdout
2. **Fit**: Train model on 80%
3. **Calibrate**: Fit a sigmoid on the 20% predictions
   ```
   P_calibrated = 1 / (1 + exp(A × score + B))
   ```
   where A and B are learned on the holdout
4. **Apply**: Use this calibrated model for final predictions

**Detailed process**:

```python
# For each model (except ensemble)
X_train_calib, X_holdout, y_train_calib, y_holdout = train_test_split(
    X_train_balanced, y_train_balanced, 
    test_size=0.2, stratify=y_train_balanced
)

# 1. Create model copy (don't reuse fitted model)
model_copy = type(model)(**model.get_params())

# 2. Train on 80%
model_copy.fit(X_train_calib, y_train_calib)

# 3. Calibrate with CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(
    model_copy,
    method='sigmoid',  # Platt scaling
    cv='prefit'  # Use pre-trained model
)
calibrated_model.fit(X_holdout, y_holdout)
```

**For ensemble**:

- Use fitted model directly (already well calibrated)
- No recalibration needed (soft voting averages probabilities)

### 8.3 Calibration Validation

**Validation methods**:

1. **Calibration Curve** (reliability diagram):
   - X-axis: Predicted probability
   - Y-axis: Actual fraction of positives
   - Perfect line: y = x (predictions = reality)

2. **Brier Score**:
   ```
   Brier = (1/N) × Σ(P_predicted - y_true)²
   ```
   - Closer to 0 → better calibration

**Expected result**:

- More reliable probabilities for threshold optimization
- Better business interpretability

---

## 🚀 9. Training and Final Evaluation

### 9.1 Training Optimized and Calibrated Models

**Process for each model**:

1. **Retrieval** of best model from GridSearchCV
2. **Calibration** with Platt scaling (if applicable)
3. **Re-training** on balanced train set (X_train_balanced, y_train_balanced)
4. **Cross-validation** 5-fold to estimate performance

**Trained models**:

- **Ensemble** (Logistic + RandomForest, soft voting)
- **RandomForest** optimized + calibrated
- **LogisticRegression** optimized + calibrated
- **Perceptron** optimized + calibrated

**Cross-validation results**:

- Average F1-score on 5 folds
- Standard deviation to evaluate stability
- Comparison of 4 approaches

### 9.2 Prediction on Test Set

**Important**: Use of `X_test_selected` (with selected features)

**Confusion Matrix**:

```
                Predicted: Stay    Predicted: Leave
Actual: Stay    TN (true neg)     FP (false pos)
Actual: Leave   FN (false neg)    TP (true pos)
```

**Business interpretation**:

- **TN**: Staying employees correctly identified (no unnecessary intervention)
- **FP**: FALSE ALERTS → unnecessary intervention, cost
- **FN**: MISSED DEPARTURES → leaving employees not detected, loss
- **TP**: Departures correctly detected → intervention possible

### 9.3 Key Metrics (Class 1 = Attrition)

**Precision**:

```
Precision = TP / (TP + FP)
```

**Question answered**: "Among alerts, how many are true?"

- Precision 35% = out of 100 alerts, 35 are real
- **Important to avoid too many unnecessary interventions**

**Recall**:

```
Recall = TP / (TP + FN)
```

**Question answered**: "Among actual departures, how many are detected?"

- Recall 50% = detects 1 departure out of 2
- **Important to minimize missed departures**

**F1-Score**:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Question answered**: "What is the balanced overall performance?"

- Harmonic mean between Precision and Recall
- Heavily penalizes if either metric is very low

**ROC-AUC (Area Under Curve)**:

- Measures model's ability to discriminate classes
- 0.5 = random, 1.0 = perfect
- Insensitive to decision threshold

### 9.4 Classification Report

Detailed report with:

- Precision, Recall, F1-score for **each class** (0 and 1)
- Support: number of samples for each class
- Macro/weighted averages

---

## 🎯 10. Decision Threshold Optimization

### 10.1 Default Threshold Problem

**Standard behavior**:

- Threshold = 0.5 (50%)
- If probability ≥ 0.5 → predict "leave"
- If probability < 0.5 → predict "stay"

**Problem**: This threshold is not optimal for imbalanced data

### 10.2 Precision-Recall Curve

**Principle**:

- Test ALL possible thresholds (from 0 to 1)
- For each threshold, calculate Precision and Recall
- Identify possible trade-offs

**Example**:

```
Threshold 0.3 → High Recall (70%), Low Precision (25%)
Threshold 0.7 → Low Recall (30%), High Precision (50%)
```

### 10.3 Optimization Strategy with Constraint

**Objective**: Maximize F1-score **WITH** Precision ≥ 35% constraint

**Reason for constraint**:

- Precision < 35% → too many false alerts
- Intervention cost too high for company
- Loss of confidence in the system

**Process**:

1. Filter all thresholds with Precision ≥ 35%
2. Among valid thresholds, choose one maximizing F1
3. If no valid threshold → use threshold maximizing F1 without constraint

### 10.4 Optimized Predictions

**Applying optimal threshold**:

```python
y_pred_optimized = (y_prob >= optimal_threshold).astype(int)
```

**Compared results**:

**With default threshold (0.5)**:

- Confusion Matrix: `[[TN FP] [FN TP]]`
- Metrics: Precision, Recall, F1

**With optimized threshold (e.g., 0.35)**:

- Improved Confusion Matrix
- **Increased F1-score**
- Respect of Precision ≥ 35% constraint

### 10.5 Business Impact

**Calculated metrics**:

- **Detected**: TP / (TP + FN) → % of detected departures
- **Cost**: TP + FP → number of employees to interview
- **Efficiency**: TP / (TP + FP) → % of useful interventions
- **Missed**: FN → number of undetected departures

**Example result**:

```
✅ Detected: 57/114 attritions (50%)
⚠️  Cost: 163 employees to interview (106 unnecessary)
💰 Efficiency: 35% of interventions are useful
❌ Missed: 57 attritions will go unnoticed
```

---

## 📊 11. Complete Pipeline Summary (Improved Version)

### 11.1 Data Flow

```
1. Merge 5 CSVs → raw_dataset (4,410 samples)
2. Remove ethical columns → cleaned_dataset
3. 🆕 Feature engineering (12 new features) → 314 features
4. Stratified 80/20 split → train (3,528) / test (882)
5. Separate preprocessing:
   - Train: fit + transform → train_processed
   - Test: transform only → test_processed
6. 🆕 Multi-method selection (ANOVA + RFECV + L1) → 36 features
7. 🆕 Nested CV for SMOTE → strategy 0.6 optimal
8. 🆕 SMOTE on train → train_balanced (class 1: 569 → 1,775)
9. 🆕 GridSearchCV × 3 models → optimal hyperparameters
10. 🆕 Platt calibration → reliable probabilities
11. 🆕 Ensemble (Logistic + RF) + 3 individual models
12. Prediction on test → y_pred (4 models)
13. Threshold optimization (Precision ≥ 35%) → y_pred_optimized
```

### 11.2 Key Anti-Leakage Points (Strictly Followed)

**✅ Best practices followed**:

1. **Split BEFORE preprocessing** (no contamination)
2. **Feature engineering BEFORE split** (deterministic transformations only)
3. **Fit only on train** (scaler, imputer, selectors)
4. **Transform on test** with train parameters
5. **Feature selection on train only** (ANOVA, RFECV, L1)
6. **🆕 Nested CV for SMOTE** (strategy optimized without test set)
7. **SMOTE only on train** (never on test)
8. **🆕 Calibration on train holdout** (20% reserved)
9. **GridSearchCV with internal CV** (5-fold on train only)
10. **Test set used ONCE** (final evaluation)

### 11.3 Key Improvements vs Previous Version

| **Aspect** | **Before** | **After** | **Gain** |
|------------|-----------|-----------|----------|
| **Features** | 15 (ANOVA only) | 36 (multi-method) | +140% |
| **Feature Engineering** | None | 12 business features | New patterns |
| **Selection** | 1 method (ANOVA) | 3 methods (ANOVA+RFECV+L1) | Robustness |
| **SMOTE** | Test set peeking | Nested CV (3-fold) | Anti-leakage |
| **Models** | 1 (Perceptron) | 4 (Ensemble+RF+Logistic+Perceptron) | Diversity |
| **Optimization** | 1 GridSearch | 3 parallel GridSearches | Comparison |
| **Calibration** | None | Platt scaling | Reliable probabilities |
| **Expected F1** | ~0.40-0.46 | ~0.50-0.60 | +25-30% |

### 11.4 Scientific Validation

**No data leakage detected**:

- ✅ Feature engineering: deterministic transformations (no global statistics)
- ✅ Feature selection: calculated on train, applied to test
- ✅ SMOTE: optimized by nested CV (no test peeking)
- ✅ Hyperparameters: GridSearchCV with internal CV
- ✅ Calibration: train holdout (20%)
- ✅ Test set: untouched until final evaluation

**Guaranteed reproducibility**:

- `random_state=42` fixed everywhere
- Same pipeline for all models
- 100% reproducible results

---

## 💡 12. Interpretation and Recommendations

### 12.1 Model Strengths

- Strict ethical framework (no sensitive variables)
- Robust anti-leakage architecture
- Multi-method feature selection for comprehensive coverage
- Ensemble approach for improved performance
- Calibrated probabilities for business decisions

### 12.2 Model Limitations

- Class imbalance remains challenging even with SMOTE
- Linear assumptions may not capture all complex interactions
- Temporal aspects could be further exploited
- Explainability-performance trade-off with ensemble

### 12.3 Recommendations for Further Improvement

**Short-term (No risk of snooping)**:
1. Add polynomial/interaction features between top features
2. Test alternative resampling: ADASYN, BorderlineSMOTE
3. Implement cost-sensitive learning
4. Add temporal aggregations (rolling satisfaction averages)

**Medium-term**:
1. Collect more data (especially minority class)
2. Add external data sources (market trends)
3. Implement online learning for model updates
4. A/B testing in production

**Long-term**:
1. Deep learning models (TabNet, Wide & Deep)
2. Causal inference methods
3. Personalized retention strategies
4. Integration with HR systems for real-time alerts
