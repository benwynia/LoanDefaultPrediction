#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Variable Histograms
Function: Create Histogram
This function creates a matrix of histograms for all numeric columns of the input dataframe. It calculates the optimal set of rows and columns for the histogram matrix based on the number of numeric columns in the dataframe. It uses the Seaborn library to plot histograms of each numeric variable in the dataframe.

Parameters:

df: Pandas DataFrame Input data frame containing numeric variables for which histograms are to be plotted. figsize: tuple, default=(15, 15) The figure size of the histogram matrix. bins: int, default=20 The number of bins to use for the histogram. color: str, default='steelblue' The color of the bars in the histogram. Output:

Displays a matrix of histograms for all numeric columns of the input dataframe. Required Libraries:

numpy matplotlib.pyplot seaborn

Example Usage:

python import pandas as pd import numpy as np import seaborn as sns import matplotlib.pyplot as plt

df = pd.DataFrame({'var1': [1,2,3,4,5], 'var2': [2,4,6,8,10], 'var3': [3,6,9,12,15], 'var4': [4,8,12,16,20]}) create_histogram(df)

Note: The function assumes that the input dataframe only contains numeric columns.
"""
def create_histogram(df, figsize=(15, 15), color='steelblue'):
    num_columns = df.select_dtypes(include=['int', 'float']).columns
    num_vars = len(num_columns)
    num_cols = 5
    num_rows = int(np.ceil(num_vars / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    def freedman_diaconis(data):
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        n = len(data)
        bin_width = 2 * iqr * (n ** (-1/3))
        return bin_width

    for idx, col in enumerate(num_columns):
        bin_width = freedman_diaconis(df[col].dropna())
        
        if bin_width == 0:
            bins = 1
        else:
            bins = int(np.ceil((df[col].max() - df[col].min()) / bin_width))
            bins = min(bins, 100)  # Add this line to limit the number of bins to a reasonable maximum

        sns.histplot(data=df, x=col, kde=False, bins=bins, color=color, ax=axes[idx])
        axes[idx].set_title(col, fontsize=12)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
        if (idx+1) % 5 == 0:
            for i in range(idx-3, idx+1):
                axes[i].set_visible(True)
            for i in range(idx+1, num_rows*num_cols):
                axes[i].set_visible(False)

    for idx in range(num_vars, num_rows * num_cols):
        fig.delaxes(axes[idx])

    plt.savefig("histograms.jpg")
    plt.show()
# In[ ]:


if __name__ == '__main__':
    print("Hello World")

