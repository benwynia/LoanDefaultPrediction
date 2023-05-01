#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, labels, target):
    low_value=labels[0]
    high_value=labels[1]
   
    # Split the original dataset into a training set and a temporary set
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)

    # Split the temporary set into a validation set and a test set
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Assess distribution of dependent variable
    print(f"""Train data dependent distribution:
    {train_data[target].value_counts()[0]} (0 - {low_value}) {train_data[target].value_counts()[1]}(1 - {high_value})""")

    print(f"""Test data dependent distribution:
    {test_data[target].value_counts()[0]} (0 - {low_value}) {test_data[target].value_counts()[1]}(1 - {high_value})""")

    print(f"""Validation data dependent distribution:
    {val_data[target].value_counts()[0]} (0 - {low_value}) {val_data[target].value_counts()[1]}(1 - {high_value})""")
    
    return train_data, test_data, val_data


# In[2]:


if __name__ == '__main__':
    print("Hello World")

