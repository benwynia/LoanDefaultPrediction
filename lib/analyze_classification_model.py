#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sklearn.metrics

from lib import model_pdf_report

def DrawConfusionMatrix(confusion_matrix, class_names):
    """
    Inputs:
    1. Confusion matrix
    2. Class names
    
    Function:
    Creates a formatted CM plot with a heatmap coloring thats easier to read and interpret
    
    Outputs:
    a figure object
    
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 4))
    
    # Custom color map
    cmap = sns.color_palette("Greens", as_cmap=True)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap=cmap)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), ha='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')

    # Add TP, TN, FP, and FN labels
    plt.text(0.5, 0.2, 'True Negative', fontsize=10, ha='center', va='center', color='white')
    plt.text(1.5, 0.2, 'False Positive', fontsize=10, ha='center', va='center', color='black')
    plt.text(0.5, 1.2, 'False Negative', fontsize=10, ha='center', va='center', color='black')
    plt.text(1.5, 1.2, 'True Positive', fontsize=10, ha='center', va='center', color='white')
    plt.tight_layout()

    fig = plt.gcf()
    plt.show()
    return fig

def specificity(y_actual, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def get_optimal_threshold(y_actual, y_pred_continuous):
    """
    Inputs:
    1. Actual y values
    2. Predicted y probabilities from model
    
    Function:
    1. Calculates youden_j stat
    2. Calculates optimal probability to maximize youden_j
    
    Outputs:
    1. youden_j_stat
    2. optimal cutoff threshold for maxing youden j
    3. FPR and TPR at maximal youden j
    
    """
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred_continuous)
    youden_j_stat = tpr - fpr
    optimal_idx = np.argmax(youden_j_stat)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
    youden_j_stat = youden_j_stat[optimal_idx]
    return youden_j_stat, optimal_threshold, optimal_fpr, optimal_tpr

def plot_pr_curve(model_name, y_actual, y_pred_proba):
    """
    Inputs:
    1. Actual y values
    2. Predicted y probabilities from model
    3. model name
    
    Function:
    1. Creates a precision recall curve
    
    Outputs:
    1. a figure object
    
    """
    precision, recall, _ = precision_recall_curve(y_actual, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig

def plot_tpr_tnr(model_name, y_actual, y_pred_proba):
    """
    Inputs:
    1. Actual y values
    2. Predicted y probabilities from model
    3. model name
    
    Function:
    1. Creates a plot showing the relationship between TPR, TNR, and Youden's J
    
    Outputs:
    1. a figure object
    
    """
    fpr, tpr, cutoffs = sklearn.metrics.roc_curve(y_actual, y_pred_proba)
    tnr = 1 - fpr
    youdens_curve = tpr + tnr - 1
    youdens_index_argmax = youdens_curve.argmax()
    youdens_index_cutoff = cutoffs[youdens_index_argmax]
    plt.figure(figsize=(6, 6))
    plt.plot(cutoffs, tpr, label='TPR')
    plt.plot(cutoffs, tnr, label='TNR')
    plt.plot(cutoffs, youdens_curve, label='Youden\'s Index Curve')
    plt.scatter(cutoffs[youdens_index_argmax], tpr[youdens_index_argmax])
    plt.scatter(cutoffs[youdens_index_argmax], tnr[youdens_index_argmax])
    plt.scatter(cutoffs[youdens_index_argmax], youdens_curve[youdens_index_argmax])
    plt.xlabel('Cutoff')
    plt.ylabel('Rate')
    plt.title(f'TPR / TNR vs Youdens Index - {model_name}')
    plt.legend()
    plt.xlim(0, 1)
    fig = plt.gcf()
    plt.show()
    return fig

def plot_roc_curve(model_name, y_actual, y_pred_proba):
    
    """
    Inputs:
    1. Actual y values
    2. Predicted y probabilities from model
    3. model name
    
    Function:
    1. Creates an ROC curve 
    
    Outputs:
    1. a figure object
    
    """
    
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Youden's J statistic
    youden_j_stat, optimal_threshold, optimal_fpr, optimal_tpr = get_optimal_threshold(y_actual, y_pred_proba)

    # Plot ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')

    # Annotate the plot with the optimal threshold and Youden's J statistic
    plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', label='Optimal Threshold')
    plt.annotate(f'Optimal Threshold = {optimal_threshold:.3f}\nYouden\'s J = {youden_j_stat:.3f}',
                 xy=(optimal_fpr, optimal_tpr), xycoords='data',
                 xytext=(70, -40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3,rad=0.2",
                                 lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="gainsboro"),
                 fontsize=10)
    plt.legend(loc="lower right")
    fig = plt.gcf()
    plt.show()
    return fig

def analyze_model(model, model_name, class_names, datasets, profit_list, result_type=None):
    
    """
    Inputs:
    1. A "fit" model ("model")
    2. model name
    3. dependent variable class names
    4. A list of datasets including: train data, testing data, features list, and target list
    
    Function:
    Runs all the functions above to analyze and report on the model performance
    2. Makes predictions using model
    3. Calculates key model statistics including accuracy, precision, recall, specificity, F1, and youden J
    4. Generates a PR Curve
    5. Generates a ROC Curve
    6. Generates a TPR, TNR, Youden J plot
    7. Plots the confusion matrix
    8. Saves a summary of the model performance to a pdf file
    
    Outputs:
    1. a "results" dataframe with all the pertinent statistics
    2. the trained model
    
    """
    
    # Unpack datasets
    train_data = datasets[0]
    test_data = datasets[1]
    features = datasets[2]
    target = datasets[3]
    
    # Make predictions on the test dataset
    y_pred_proba = model.predict_proba(test_data[features])[:, 1]
    y_pred = model.predict(test_data[features])

    # Calulcate key model statistics
    conf_matrix = confusion_matrix(test_data[target], y_pred)
    accuracy = accuracy_score(test_data[target], y_pred)
    precision = precision_score(test_data[target], y_pred)
    recall = recall_score(test_data[target], y_pred)
    spec = specificity(test_data[target], y_pred)
    f1 = f1_score(test_data[target], y_pred)
    youden_j_stat, optimal_threshold, optimal_fpr, optimal_tpr = get_optimal_threshold(test_data[target], y_pred_proba)
    
    # Generate a PR curve plot
    pr_curve_plot = plot_pr_curve(model_name, test_data[target], y_pred_proba)
    pr_curve_plot
    
    # Generate an ROC curve plot 
    roc_curve_plot = plot_roc_curve(model_name, test_data[target], y_pred_proba)
    roc_curve_plot
    
    # Generate a TPR TNR Plot 
    tpr_tnr_plot = plot_tpr_tnr(model_name, test_data[target], y_pred_proba)
    tpr_tnr_plot
    
    # Create confusion matrix 
    conf_matrix_plot = DrawConfusionMatrix(conf_matrix, class_names)
    conf_matrix_plot
    
    
    # Summarize model results 
    results = pd.DataFrame({
        'Model Name': [model_name],
        'Confusion Matrix': [conf_matrix],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'Specificity': [spec],
        'F1 Score': [f1],
        'Youden J Stat': [youden_j_stat],
        'Optimal P Threshold': [optimal_threshold],
    })
    
    return model_name, pr_curve_plot, roc_curve_plot, tpr_tnr_plot, conf_matrix, conf_matrix_plot, results


# In[2]:


if __name__ == '__main__':
    print("Hello World")

