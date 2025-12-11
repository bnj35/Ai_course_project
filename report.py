import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tarfile
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

warnings.filterwarnings('ignore')


folder_path = 'data/'
employee_file_name = 'employee_survey_data.csv'
general_file_name = 'general_data.csv'
manager_file_name = 'manager_survey_data.csv'
#for the in out time 
time_folder_path = 'in_out_time/'
in_time_file_name = 'in_time.csv'
out_time_file_name = 'out_time.csv'

# Load each dataset
employee_data = pd.read_csv(os.path.join(folder_path, employee_file_name))
general_data = pd.read_csv(os.path.join(folder_path, general_file_name))
manager_data = pd.read_csv(os.path.join(folder_path, manager_file_name))
in_time_data = pd.read_csv(os.path.join(folder_path, time_folder_path, in_time_file_name))
out_time_data = pd.read_csv(os.path.join(folder_path, time_folder_path, out_time_file_name))


##############################
# TIME
##############################

# merge in_time and out_time data on the first column (Unknown that is actually EmployeeID)
# rename the first column to EmployeeID for both datasets because it is unnamed
in_time_data.rename(columns={in_time_data.columns[0]: 'EmployeeID'}, inplace=True)
out_time_data.rename(columns={out_time_data.columns[0]: 'EmployeeID'}, inplace=True)

#check if days are present in both datasets
in_time_days = set(in_time_data.columns[1:])
out_time_days = set(out_time_data.columns[1:])
missing_in_out = in_time_days.difference(out_time_days)
# display the missing days
print(f"Days missing in either in_time or out_time data: {missing_in_out}")

# go through each column to check empty cells present only in one of the datasets
for day in in_time_days.intersection(out_time_days):
    in_time_empty = set(in_time_data.index[in_time_data[day].isnull()])
    out_time_empty = set(out_time_data.index[out_time_data[day].isnull()])
    missing_in_out_rows = in_time_empty.symmetric_difference(out_time_empty)
    if missing_in_out_rows:
        print(f"Day {day} has missing entries in either in_time or out_time data at rows: {missing_in_out_rows}")

# convert all columns except the first one to datetime format
for col in in_time_data.columns[1:]:
    in_time_data[col] = pd.to_datetime(in_time_data[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
for col in out_time_data.columns[1:]:
    out_time_data[col] = pd.to_datetime(out_time_data[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')

#function to remove columns depending on distinct values for relevance
def remove_col_depending_on_distinct_values(df, start_threshold=0, end_threshold=0):
    cols_to_remove = []
    for col in df.columns:
        if start_threshold <= df[col].nunique() <= end_threshold:
            cols_to_remove.append(col)
    df.drop(columns=cols_to_remove, inplace=True)
    return df

# merge in and out time data based on EmployeeID
time_data = pd.merge(in_time_data, out_time_data, on='EmployeeID', suffixes=('_in', '_out'))

# create a new column for each day calculating the difference between out and in time in hours
hours_columns = {}
day_of_week_columns = {}
for day in in_time_days.intersection(out_time_days):
    hours_columns[f'{day}_hours'] = (time_data[f'{day}_out'] - time_data[f'{day}_in']).dt.total_seconds() / 3600.0
    day_of_week_columns[f'{day}_day_of_week'] = time_data[f'{day}_in'].dt.dayofweek

# use pd.concat to avoid DataFrame fragmentation
# Concatenate all hours columns at once and create a new column called "duration_hours"
time_data = pd.concat([time_data, pd.DataFrame(hours_columns, index=time_data.index)], axis=1)
time_data = pd.concat([time_data, pd.DataFrame(day_of_week_columns, index=time_data.index)], axis=1)
time_data['duration_hours'] = time_data[list(hours_columns.keys())].sum(axis=1)

# aggregate by day of week
day_of_week_counts = {}
day_of_week_avg_hours = {}

for i in range(7): # 0=Monday through 6=Sunday
    count_cols = [col for col in time_data.columns if col.endswith('_day_of_week')]
    day_of_week_counts[f'worked_on_day_{i}'] = sum(
        (time_data[col] == i).astype(int) for col in count_cols
    )
    
    # avg hrs per day of week
    total_hours = 0
    for day in in_time_days.intersection(out_time_days):
        day_col = f'{day}_day_of_week'
        hours_col = f'{day}_hours'
        if day_col in time_data.columns and hours_col in time_data.columns:
            # only sum hours where the day of week matches
            mask = time_data[day_col] == i
            total_hours += time_data[hours_col].where(mask, 0)
    day_of_week_avg_hours[f'avg_hours_day_{i}'] = total_hours / day_of_week_counts[f'worked_on_day_{i}'].replace(0, 1)

time_data = pd.concat([time_data, pd.DataFrame(day_of_week_counts, index=time_data.index)], axis=1)
time_data = pd.concat([time_data, pd.DataFrame(day_of_week_avg_hours, index=time_data.index)], axis=1)

# remove columns with 0 distinct values
remove_col_depending_on_distinct_values(time_data)

# keep only columns: EmployeeID, duration_hours, worked_on_day_*, avg_hours_day_*
cols_to_keep = ['EmployeeID', 'duration_hours'] + [col for col in time_data.columns if col.startswith('worked_on_day_') or col.startswith('avg_hours_day_')]
time_data = time_data[cols_to_keep]
time_data = pd.concat([time_data, pd.DataFrame(hours_columns, index=time_data.index)], axis=1)

##############################
#Pipeline
##############################

# pipeline with all steps included above as parameters for easy reuse
def preprocess_data(dataset, impute_values=True, numeric_cols=None, categorical_cols=None, scale_data=True, encode_ordinal_cols=None, encode_onehot_cols=True, remove_constant_cols=True, remove_from_encoding=[]):
    #copy the dataset to avoid modifying the original data
    data = dataset.copy()
    # remove constant columns
    if remove_constant_cols:
        data = remove_col_depending_on_distinct_values(data, end_threshold=1)
    # identify numerical and categorical columns if not provided
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    # impute missing values
    if impute_values:
        if len(numeric_cols) > 0:
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        if len(categorical_cols) > 0:
            data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    # ordinal encoding
    if encode_ordinal_cols and len(categorical_cols) > 0:
        for col, categories in encode_ordinal_cols.items():
            if col in data.columns:
                data[col] = pd.Categorical(data[col], categories=categories, ordered=True).codes
                # Remove ordinally encoded columns from categorical_cols to avoid one-hot encoding them
                if col in categorical_cols:
                    categorical_cols.remove(col)
                # Add to numeric_cols since it's now numeric
                if col not in numeric_cols:
                    numeric_cols.append(col)
    # one-hot encoding
    if encode_onehot_cols:
        # If encode_onehot_cols is True, use the remaining categorical columns
        if encode_onehot_cols is True:
            cols_to_encode = categorical_cols
        else:
            cols_to_encode = encode_onehot_cols
        
        # Remove columns specified in remove_from_encoding
        cols_to_encode = [col for col in cols_to_encode if col not in remove_from_encoding]
        
        if len(cols_to_encode) > 0:
            data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)
    # scale numerical data
    if scale_data:
        scaler = StandardScaler(with_mean=True)
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data


##############################
#merge
##############################

from sklearn.model_selection import train_test_split
# merge employee and manager data first
employee_manager_data = pd.merge(employee_data, manager_data, on='EmployeeID', suffixes=('_emp', '_mgr'))
# merge all datasets into a final dataset on EmployeeID
final_dataset = pd.merge(general_data, employee_manager_data, on='EmployeeID')
final_dataset = pd.merge(final_dataset, time_data, on='EmployeeID')

# split dataset into training and testing sets
train_set, test_set = train_test_split(final_dataset, test_size=0.2, random_state=42)
#todo: cross-validation of train_set

# place the EmployeeID column at the front
cols = final_dataset.columns.tolist()
cols.insert(0, cols.pop(cols.index('EmployeeID')))
final_dataset = final_dataset[cols]

# print to verify 
# print(final_dataset.info())
# print(final_dataset.head())

#into pipeline
ordinal_mappings = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
    }

final_dataset = preprocess_data(final_dataset,
                                impute_values=True,
                                scale_data=True,
                                encode_ordinal_cols=ordinal_mappings,
                                encode_onehot_cols=True,
                                remove_constant_cols=True,
                                remove_from_encoding=['Attrition', 'Department','EducationField', 'JobRole']                                
                                )

#######################
# Analysis of Variance
#######################
numerical_columns = final_dataset.select_dtypes(include=[np.number]).columns.tolist()
ordinal_mappings_keys = list(ordinal_mappings.keys())

#######################
# Mutual Information
#######################
mi_columns = ["Department", "EducationField", "JobRole"]

##############################
#Correlation verification
##############################

def annova_mi_with_target(data, target_column, exclude_patterns=None):
    """
    Calculate feature importance using ANOVA F-statistic and Mutual Information
    """
    data_copy = data.copy()
    target_data = data_copy[target_column]
    anova_scores = {}
    mi_scores = {}
    
    # Get numeric columns for ANOVA
    anova_columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
    anova_columns = [col for col in anova_columns if col not in [target_column, "EmployeeID"]]
    
    # Get categorical columns for MI
    mi_columns = data_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    mi_columns = [col for col in mi_columns if col != target_column]
    
    # Calculate ANOVA scores
    for col in anova_columns:
        valid_idx = data_copy[[col, target_column]].dropna().index
        X_col = data_copy.loc[valid_idx, col].values.reshape(-1, 1)
        y_col = data_copy.loc[valid_idx, target_column].values
        f_stat, _ = f_classif(X_col, y_col)
        anova_scores[col] = f_stat[0]

    # Calculate MI scores
    for col in mi_columns:
        encoded_col = pd.Categorical(data_copy[col]).codes
        mi_score = mutual_info_classif(encoded_col.reshape(-1, 1), target_data, random_state=42)[0]
        mi_scores[col] = mi_score
    
    # Convert to sorted Series (easier to work with than dicts)
    anova_series = pd.Series(anova_scores).sort_values(ascending=False)
    mi_series = pd.Series(mi_scores).sort_values(ascending=False)
    
    # Filter based on exclude patterns
    if exclude_patterns:
        anova_series = anova_series[~anova_series.index.str.contains('|'.join(exclude_patterns), regex=True)]
        mi_series = mi_series[~mi_series.index.str.contains('|'.join(exclude_patterns), regex=True)]
    
    # No need to return both list and Series - caller can get list with .index.tolist()
    return anova_series, mi_series

anova_filtered_scores, mi_filtered_scores = annova_mi_with_target(
    final_dataset, 
    "Attrition",
    exclude_patterns=['EmployeeID', 'day_of_week', 'avg_hours_day_', r'\d{4}-\d{2}-\d{2}_hours']
)

print(f"\nANOVA Feature Importance:\n{anova_filtered_scores}")
print(f"\nMutual Information Feature Importance:\n{mi_filtered_scores}")

#########
# Display
#########

def order_correlation(corr_series , ascending=False):
    ordered_corr = corr_series.sort_values(ascending=ascending)
    return ordered_corr


# absolute_filtered_corr = filtered_corr.abs()
# # ordered correlation by absolute value
# absolute_filtered_corr = filtered_corr.sort_values(ascending=False)

# top_n_related = absolute_filtered_corr.where(absolute_filtered_corr > absolute_filtered_corr.mean(), None).dropna()
# print(f"\nTop {len(top_n_related)} related features with Attrition:\n{top_n_related}")

# day_of_week_corr = corr_values[[col for col in corr_values.index if 'day' in col and ('worked_on' in col or 'avg_hours' in col)]]
# if len(day_of_week_corr) > 0:
#     # Standardize the values
#     standardized_values = StandardScaler().fit_transform(day_of_week_corr.values.reshape(-1, 1)).flatten()
#     day_of_week_corr = pd.Series(standardized_values, index=day_of_week_corr.index)
#     print(f"\nDay-of-week correlations with Attrition:\n{day_of_week_corr}")

# final_dataset = final_dataset[ordered_cols]
target_col = 'Attrition'

# Ensure Attrition is numeric
if final_dataset[target_col].dtype == 'object':
    final_dataset[target_col] = final_dataset[target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1'] else 0)

# Prepare X and y from the processed dataset
X = final_dataset.drop(columns=[target_col], errors='ignore')
y = final_dataset[target_col]

# Split processed data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining with processed data. Shape: {X_train.shape}")

perceptron = Perceptron(max_iter=2000, random_state=42)
# Train the model
perceptron.fit(X_train, y_train)

# Cross validation
scores = cross_val_score(perceptron, X_train, y_train, cv=5, scoring='accuracy')
print("\n--- Cross Validation ---")
print("scores: ", scores)
print("Mean: ", scores.mean())
print("Standard Deviation: ", scores.std())

# Predictions
y_pred = perceptron.predict(X_test)
y_prob = perceptron._predict_proba_lr(X_test)[:, 1]

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# AUC-ROC
roc_auc = roc_auc_score(y_test, y_prob)
print("AUC-ROC: ", roc_auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)
print("AUC (computed): ", roc_auc_val)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


for i in final_dataset.columns:
    if ( i == "Department"):
        print(final_dataset[i])
    if ( i == "JobRole"):
        print(final_dataset[i])
    if ( i == "EducationField"):
        print(final_dataset[i])