#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt

def select_columns(df, columns_to_keep):
    """
    Inputs: a dataframe, a list of columns to retain
    Function: processes a dataframe and removes any unwanted columns
    Outputs: a new dataframe 
    """
    final_columns = [col for col in columns_to_keep if col in df.columns]
    new_df = df[final_columns]
    return new_df

def get_dataframe_shape(df):
    """
    Inputs: a dataframe
    Function: Provides information about the size and shape of the dataframe
    Outputs: A string describing the shape of the dataframe
    """
    num_columns = df.shape[1]
    num_rows = df.shape[0]
    print(f"The dataset has {num_columns} columns and {num_rows} rows")

def clean_up_missing_data(df, breakpoint):
    """
    Inputs: a dataframe, a data completeness breakpoint metric
    Function: drops any columns that are missing too many data points based on a breakpoint provided by the user
    Outputs: returns a dataframe with the columns with too many missing values removed 
    """
    
    nan_percentages = df.isna().mean() * 100
    nan_percentages.plot.hist()

    # set the plot title and labels
    plt.title("Histogram of Number of Variables with NaN Percentage")
    plt.xlabel("Percentage NaN / Blank / Missing")
    plt.ylabel("Count of Variables/Columns")

    # display the plot
    plt.show()
    
    # remove any variables that have a NaN below the breakpoint
    df = df.dropna(thresh=len(df) * breakpoint, axis=1)
    
    # Return a report of the nan_percentages by variable and the remainin columns 
    return nan_percentages, df

def drop_rows_with_na(df):
    """
    Inputs: a dataframe
    Function: drops NA records from the dataframe
    Outputs: a dataframe without NA records
    """
    
    # Count the number of rows in the original dataframe
    num_rows_before = len(df)

    # Drop rows with NaN values
    df= df.dropna()

    # Count the number of rows in the filtered dataframe
    num_rows_after = len(df)

    # Calculate the percentage of rows that were dropped
    percent_dropped = (num_rows_before - num_rows_after) / num_rows_before * 100

    # Print the number of rows before and after filtering, and the percentage dropped
    print("Number of rows before filtering:", num_rows_before)
    print("Number of rows after filtering:", num_rows_after)
    print(f"Percentage of rows dropped: {round(percent_dropped,2)}%")
    return df

def one_hot_encode_categorical_variables(df, col_list):
    """
    Inputs- a dataframe, a list of column names to be one-hot encoded
    Function: 1. creates a dummy binary variable for each level of the categorical variable. 
              2. Columns are named with the prefix of '1h_<original column name>'
              3. Original variable is dropped from dataframe
    Outpus: dataframe with all variables in col_list recoded 
    """
    for column in col_list:
        current_prefix = column
        one_hot_encoded = pd.get_dummies(df[column], prefix=f'1h_{current_prefix}')
        df = df.drop(column, axis=1)
        df = pd.concat([df, one_hot_encoded], axis=1)
    return df

def remove_leading_spaces(df):
    new_columns = {col: col.lstrip() for col in df.columns}
    return df.rename(columns=new_columns)

def run_data_cleanup_functions(df, data_dictionary, columns_to_keep, columns_to_recode, breakpoint):
    """
    Inputs: 
    1. our dataframe
    2. A data dictionary which has two colums "Name" and "Definition"
    3. A list of columns to keep
    4. A list of columns to recode using dummy variables
    5. A breakpoint for the minimum data completeness required for each column.
    
    Function:
    1. Prunes the dataframe to only keep the selected columns
    2. Removes columns with too many missing records
    3. Removes rows with missing values
    4. Recodes the categorical variables into dummies
    5. Prunes the data dictionary to retain only the variables use in the function.
    6. Prints the dataframe shape after each operation so you know how the dataframe has changed

    Outputs:
    1. resulting dataframe
    2. data dictionary
    """
    
    print("Original dataframe---")
    get_dataframe_shape(df)
    
    # Step 1 - Clean up column names
    # df = remove_leading_spaces(df)
    
    # Step 2 - Select Columns
    df = select_columns(df, columns_to_keep)
    
    # Step 3 - Remove columns with too many missing records
    nan_percentages, df_cleaned = clean_up_missing_data(df, breakpoint)
    print("After cleaning up columns with missing data---")
    get_dataframe_shape(df_cleaned)
    
    # Step 4 - Remove rows with NAs 
    df_cleaned = drop_rows_with_na(df_cleaned)
    print("After cleaning up rows with missing data---")
    get_dataframe_shape(df_cleaned)
    
    # Step 5 - Recode categorical variables 
    df_cleaned = one_hot_encode_categorical_variables(df_cleaned, columns_to_recode)
    print("After recoding categorical variables into one-hot variables---")
    get_dataframe_shape(df_cleaned)
    
    # Step 6 - Prune the data dictionary
    data_dictionary = data_dictionary[data_dictionary['Name'].isin(df_cleaned.columns.tolist())].reset_index(drop=True)
    
    return df_cleaned, data_dictionary


# In[ ]:


if __name__ == '__main__':
	print("Hello World")

