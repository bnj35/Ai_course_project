import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tarfile
import os
import sklearn
from sklearn.preprocessing import StandardScaler

# Configuring display settings
plt.rcParams['figure.figsize'] = (12, 9)
sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings("ignore", category=FutureWarning)

##############################################################
#import files
##############################################################

# Extract the .tgz file (use a local directory to avoid /mnt permission issues)
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

#display the info of each dataset

# print("Employee Data Info:")
# print(employee_data.info())
# print("\nGeneral Data Info:")
# print(general_data.info())
# print("\nManager Data Info:")
# print(manager_data.info())
# print("\nIn Time Data Info:")
# print(in_time_data.info())
# print("\nOut Time Data Info:")
# print(out_time_data.info())

################################################
#Clean and merge hour datasets
################################################

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

print(time_data.describe())

################################################
#Clean Manager Data
################################################

# merge employee and manager data on EmployeeID
employee_manager_data = pd.merge(employee_data, manager_data, on='EmployeeID', suffixes=('_emp', '_mgr'))

remove_col_depending_on_distinct_values(employee_manager_data)

#verification
# print(employee_manager_data.info())

################################################
#Clean General Data
################################################

col_before = general_data.columns.tolist()
remove_col_depending_on_distinct_values(general_data)
col_after = general_data.columns.tolist()
removed_cols = set(col_before) - set(col_after)
print(f"Removed columns: {removed_cols}")
# print(general_data.info())

################################################
#Get numeric and categorical columns
################################################

# get numerical and categorical columns for each dataset
#general data
general_numerical_cols = general_data.select_dtypes(include=[np.number]).columns.tolist()
general_categorical_cols = general_data.select_dtypes(include=['object']).columns.tolist()
#time data
time_numerical_cols = time_data.select_dtypes(include=[np.number, np.datetime64]).columns.tolist()
#in_time data
# in_time_numerical_cols = in_time_data.select_dtypes(include=[np.number, np.datetime64]).columns.tolist()
#out_time data
# out_time_numerical_cols = out_time_data.select_dtypes(include=[np.number, np.datetime64]).columns.tolist()

#only numerical
#employee_manager data
employee_manager_numerical_cols = employee_manager_data.select_dtypes(include=[np.number]).columns.tolist()
#manager data
manager_numerical_cols = manager_data.select_dtypes(include=[np.number]).columns.tolist()
#employee data
employee_numerical_cols = employee_data.select_dtypes(include=[np.number]).columns.tolist()

#print the numerical and categorical columns for each dataset
#print("General Data Numerical Columns:", general_numerical_cols)
#print("General Data Categorical Columns:", general_categorical_cols)
#print("Employee Data Numerical Columns:", employee_numerical_cols)
#print("In Time Data Numerical Columns:", in_time_numerical_cols)
#print("In Time Data Categorical Columns:", in_time_categorical_cols)
#print("Out Time Data Numerical Columns:", out_time_numerical_cols)
# print("Out Time Data Categorical Columns:", out_time_categorical_cols)
#print("Manager Data Numerical Columns:", manager_numerical_cols)
# print(time_numerical_cols)


#################################################
#Imputation
#################################################
#imputing missing values for numerical columns with mean
general_data[general_numerical_cols] = general_data[general_numerical_cols].fillna(general_data[general_numerical_cols].median())
employee_data[employee_numerical_cols] = employee_data[employee_numerical_cols].fillna(employee_data[employee_numerical_cols].median())
# in_time_data[in_time_numerical_cols] = in_time_data[in_time_numerical_cols].fillna(in_time_data[in_time_numerical_cols].median())
# out_time_data[out_time_numerical_cols] = out_time_data[out_time_numerical_cols].fillna(out_time_data[out_time_numerical_cols].median())
manager_data[manager_numerical_cols] = manager_data[manager_numerical_cols].fillna(manager_data[manager_numerical_cols].median())


#imputing missing values for categorical columns with mode
general_data[general_categorical_cols] = general_data[general_categorical_cols].fillna(general_data[general_categorical_cols].mode().iloc[0])

#verify no missing values remain 
# print("Missing values in General Data:\n", general_data.isnull().sum())
# print("Missing values in Employee Data:\n", employee_data.isnull().sum())
# print("Missing values in Manager Data:\n", manager_data.isnull().sum())
# print("Missing values in In Time Data:\n", in_time_data.isnull().sum())
# print("Missing values in Out Time Data:\n", out_time_data.isnull().sum())

#################################################
#Standardization and Encoding
#################################################

# Initialize the StandardScaler
# scaler = StandardScaler(with_mean=True)
# # Scale numerical columns
# general_data[general_numerical_cols] = scaler.fit_transform(general_data[general_numerical_cols])
# employee_data[employee_numerical_cols] = scaler.fit_transform(employee_data[employee_numerical_cols])
# manager_data[manager_numerical_cols] = scaler.fit_transform(manager_data[manager_numerical_cols])

#display the first few rows of each dataset after preprocessing
# print("Employee Data after preprocessing:")
# print(employee_data.head())
# print("\nGeneral Data after preprocessing:")
# print(general_data.head())
# print("\nManager Data after preprocessing:")
# print(manager_data.head())
# print("\nIn Time Data after preprocessing:")
# print(in_time_data.head())
# print("\nOut Time Data after preprocessing:")
# print(out_time_data.head())

##################################################
#encoding 
##################################################

# find unique values in BusinessTravel column
print( general_data['BusinessTravel'].unique())

# Define the order for ordinal encoding (only category names, no numeric values)
# categories parameter needs to be a list of lists - one list per feature
business_travel_categories = [['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']]
# ordinal encoding for categorical columns
general_data['BusinessTravel'] = pd.Categorical(general_data['BusinessTravel'], categories=business_travel_categories[0], ordered=True).codes
# remove 'BusinessTravel' from categorical columns list as it has been ordinal encoded
general_categorical_cols.remove('BusinessTravel')
# add 'BusinessTravel' to numerical columns list
general_numerical_cols.append('BusinessTravel')

# hot one encoding for categorical columns
general_data = pd.get_dummies(general_data, columns=general_categorical_cols, drop_first=True)


print("Categorical columns in General Data after Ordinal Encoding:")

# verify the new info of general_data
print(general_data.info())


# # in_time out_time not categorical or not using like that 

# # in_time_data = pd.get_dummies(in_time_data, columns=in_time_categorical_cols, drop_first=True)
# # out_time_data = pd.get_dummies(out_time_data, columns=out_time_categorical_cols, drop_first=True)


###################################################
#merge all datasets
###################################################

# merge all datasets into a final dataset on EmployeeID
final_dataset = pd.merge(general_data, employee_manager_data, on='EmployeeID')
final_dataset = pd.merge(final_dataset, time_data, on='EmployeeID')

# place the EmployeeID column at the front
cols = final_dataset.columns.tolist()
cols.insert(0, cols.pop(cols.index('EmployeeID')))
final_dataset = final_dataset[cols]

# print to verify 

print(final_dataset.info())
print(final_dataset.head())

####################################################
#Pipeline function
####################################################

test_business_travel_categories = [['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']]

dictionary_for_encoding = {
    'BusinessTravel': test_business_travel_categories[0]
}


# pipeline with all steps included above as parameters for easy reuse
def preprocess_data(dataset, impute_values=True, numeric_cols=None, categorical_cols=None, scale_data=True, encode_ordinal_cols=None, encode_onehot_cols=True, remove_constant_cols=True):
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
        
        if len(cols_to_encode) > 0:
            data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)
    # scale numerical data
    if scale_data:
        scaler = StandardScaler(with_mean=True)
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data

# Load fresh data for testing the pipeline
test_dataset = pd.read_csv(os.path.join(folder_path, general_file_name))

pipeline_test = preprocess_data(test_dataset,
                                    remove_constant_cols=True,
                                    impute_values=True,
                                    scale_data=True,
                                    encode_onehot_cols=True,
                                    encode_ordinal_cols=dictionary_for_encoding
                                    )


print(pipeline_test.info())
print(pipeline_test.head())

################################################
#learn and predict model
################################################
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Definition of independent variables (X) and dependent variable (y)
X = # TODO: define X based on the preprocessed dataset
y = # TODO: define y based on the preprocessed dataset

# add a constant to the model (intercept)
X = sm.add_constant(X)
# initialize and fit the model
model = sm.OLS(y, X).fit()

# verification
print(model.summary())

# Try another model from sklearn
A = #TODO: define A based on the preprocessed dataset
B = #TODO: define B based on the preprocessed dataset

# initialize the model
skmodel = LinearRegression()
# fit the model
skmodel.fit(A, B)
# make predictions
A["sklearn_preds"] = skmodel.predict(A)