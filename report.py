import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tarfile
import os
import sklearn
from sklearn.preprocessing import StandardScaler

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
def remove_col_depending_on_distinct_values(df,start_threshold=0, end_threshold=1):
    cols_to_remove = []
    for col in df.columns:
        if start_threshold <= df[col].nunique() <= end_threshold:
            cols_to_remove.append(col)
    df.drop(columns=cols_to_remove, inplace=True)
    return df

# merge in and out time data based on EmployeeID
time_data = pd.merge(in_time_data, out_time_data, on='EmployeeID', suffixes=('_in', '_out'))

# create a new column for each day calculating the difference between out and in time in hours
# use pd.concat to avoid DataFrame fragmentation
hours_columns = {}
for day in in_time_days.intersection(out_time_days):
    hours_columns[f'{day}_hours'] = (time_data[f'{day}_out'] - time_data[f'{day}_in']).dt.total_seconds() / 3600.0

# Concatenate all hours columns at once and create a new column called "duration_hours"
time_data = pd.concat([time_data, pd.DataFrame(hours_columns, index=time_data.index)], axis=1)
time_data['duration_hours'] = time_data[list(hours_columns.keys())].sum(axis=1)

# remove columns with only one or 0 distinct values
remove_col_depending_on_distinct_values(time_data, end_threshold=0)

# remove all the day columns keeping only duration_hours and EmployeeID
time_data = time_data[['EmployeeID', 'duration_hours']]
# insert the hour work by day columns back to time_data
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
        data = remove_col_depending_on_distinct_values(data,start_threshold=0, end_threshold=1)
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

# place the EmployeeID column at the front
cols = final_dataset.columns.tolist()
cols.insert(0, cols.pop(cols.index('EmployeeID')))
final_dataset = final_dataset[cols]

# print to verify 

# print(final_dataset.info())
# print(final_dataset.head())

#into pipeline
preprocess_data(final_dataset,
                impute_values=True,
                scale_data=True,
                encode_onehot_cols=True,
                remove_constant_cols=True,
                remove_from_encoding=['Attrition']
                )


##############################
#Correlation verification
##############################

print(final_dataset.info())


# Correlation verification between a target feature and others
def correlation_with_target(data, target_column):
    # Make a copy to avoid modifying original data
    data_copy = data.copy()
    
    # If target column is not numeric, try to encode it
    if target_column in data_copy.columns and data_copy[target_column].dtype == 'object':
        # Map Yes/No to 1/0, or use label encoding for other categorical values
        unique_vals = data_copy[target_column].unique()
        if set(unique_vals).issubset({'Yes', 'No', np.nan}):
            data_copy[target_column] = data_copy[target_column].map({'Yes': 1, 'No': 0})
        else:
            # For other categorical values, use numeric encoding
            data_copy[target_column] = pd.Categorical(data_copy[target_column]).codes
    
    # Select only numeric columns for correlation
    numeric_data = data_copy.select_dtypes(include=[np.number])
    
    # Check if target column exists in numeric data
    if target_column not in numeric_data.columns:
        raise ValueError(f"Target column '{target_column}' could not be converted to numeric or does not exist")
    
    correlation = numeric_data.corr()[target_column].sort_values(ascending=False)
    
    corelation_ordered = correlation.index.tolist()
    return corelation_ordered, correlation

# order dataset columns based on correlation with target 'Attrition'
ordered_cols, corr_values = correlation_with_target(final_dataset, 'Attrition')
print(f"Top 10 correlated features with Attrition:\n{corr_values.head(10)}")

final_dataset = final_dataset[ordered_cols]

print(final_dataset.info())