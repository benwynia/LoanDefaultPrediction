#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd

"""
Descriptive Statistics 
Function: Descriptive Statistics
This function takes a Pandas dataframe as input and computes various descriptive statistics of the variables. If the variable is numeric, it calculates the count, missing values, mean, median, mode, range, variance, standard deviation, skewness, kurtosis, minimum, maximum, and interquartile range. If the variable is not numeric, it calculates the frequency distribution, relative frequency, and mode.

Parameters: df: Pandas DataFrame Input data frame containing variables for which descriptive statistics are to be computed.

Output: A Pandas DataFrame containing descriptive statistics for each variable in the input data frame.

Required Libraries: pandas

Example Usage:

import pandas as pd df = pd.DataFrame({'var1': [1,2,3,4,5], 'var2': ['a','b','c','d','e'], 'var3': [1.1,2.2,3.3,4.4,5.5]}) descriptive_statistics(df)

Note: The function assumes that the input dataframe is cleaned and has no missing values except for NaNs.

Note on Data Types The function checks whether a variable is numeric or not based on its data type. It assumes that variables of type "int64" and "float64" are numeric and all other variables are not numeric. If you have variables that are numeric but have data types other than "int64" or "float64", you can modify the code to include those data types.


"""



def descriptive_statistics(df):
    """
    Input - takes a dataframe
    Function - computes the following information
    1. Count
    2. Missing values
    3. Data Type
    If the data is numeric, it will also:
    5-15. Mean, median, mode, range, variance, std dev, skewness, kurtosis, min, max, and interquartile range
    
    If the data is not numeric, it will calculate:
    1. frequency distribution
    2. relative frquencu
    4. mode 
    """

    stats = pd.DataFrame()

    for col in df.columns:
        col_stats = {
            'count': df[col].count(),
            'missing_values': df[col].isna().sum(),
            'data_type': df[col].dtype
        }

        #if pd.api.types.is_numeric_dtype(df[col]):
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            col_stats.update({
                'mean': df[col].mean(),
                'median': df[col].median(),
                'mode': df[col].mode().iloc[0],
                'range': df[col].max() - df[col].min(),
                'variance': df[col].var(),
                'std_dev': df[col].std(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurt(),
                'min': df[col].min(),
                'max': df[col].max(),
                'IQR': df[col].quantile(0.75) - df[col].quantile(0.25)
            })
        else:
            col_stats.update({
                'freq_dist': df[col].value_counts().to_dict(),
                'rel_freq': (df[col].value_counts(normalize=True) * 100).to_dict(),
                'mode': df[col].mode().iloc[0]
            })

        stats[col] = pd.Series(col_stats)

    # Transpose dataframe so that each variable is its own row
    stats = stats.transpose()

    # Round everything to three decimal places 
    stats = stats.round(3)
    stats.fillna('None', inplace=True)
    return  stats


# In[ ]:




