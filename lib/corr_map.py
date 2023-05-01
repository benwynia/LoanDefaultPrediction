#!/usr/bin/env python
# coding: utf-8

# In[ ]:
"""
Function: Correlation Heatmap
This function computes and visualizes pairwise correlation matrix of numeric variables in a Pandas dataframe using Spearman's rank correlation method. It drops variables that have too few samples or zero variance before computing the correlation matrix. It creates a heatmap of the correlation matrix using Seaborn library and saves the plot as an image file.

Parameters:
df: Pandas DataFrame
Input data frame containing numeric variables for which correlation heatmap is to be plotted.
Output:
A heatmap of pairwise correlations between numeric variables in the input data frame.

Required Libraries:
numpy
seaborn
pingouin
matplotlib.pyplot

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500000000

def correlation_heatmap(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove variables with too few samples or zero variance
    min_sample_size = 5
    small_vars = []
    for col in numeric_df.columns:
        if numeric_df[col].count() < min_sample_size:
            small_vars.append(col)
    if small_vars:
        print("Dropping variables due to small sample size:", small_vars)
    numeric_df = numeric_df.drop(columns=small_vars)

    zero_var_vars = []
    for col in numeric_df.columns:
        if numeric_df[col].nunique() == 1:
            zero_var_vars.append(col)
    if zero_var_vars:
        print("Dropping variables due to zero variance:", zero_var_vars)
    numeric_df = numeric_df.drop(columns=zero_var_vars)

    # Compute pairwise correlation matrix using Spearman's rank correlation
    corr_matrix = pg.pairwise_corr(numeric_df, method='spearman').pivot_table(index='X', columns='Y', values='r')

    # Calculate the number of variables in the correlation matrix
    num_vars = len(corr_matrix.columns)

    # Set the figure size based on the number of variables
    fig_size = (num_vars * 2, num_vars * 2)

    # Create a heatmap using Seaborn
    plt.figure(figsize=fig_size)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("corr_plot.jpg")
    plt.show()


# In[ ]:


if __name__ == '__main__':
    print("Hello World")

