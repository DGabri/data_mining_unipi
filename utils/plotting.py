import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_nans_stacked(df, title):
    nan_count = df.isnull().sum()
    nan_pct = (nan_count / len(df)) * 100
    non_nan = (df.notnull().sum() / len(df)) * 100

    plot_data = pd.DataFrame({
        'NaN Values %': nan_pct,
        'NaN Values': nan_count,
        'Non-NaN Values %': non_nan
    })

    plot_data = plot_data.sort_values(by=['NaN Values'], ascending=False)
    display(plot_data)

    plot_data.drop("NaN Values", axis=1, inplace=True)
    
    # stacked bar to see the proportion of NaN vs non NaN in each column for 
    ax = plot_data.plot(kind='bar', stacked=True, 
                        figsize=(12, 6),
                        color=['#e74c3c', '#2ecc71'])
    ax.set_title(str(title), fontsize=12, fontweight='bold')
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def plot_bar_chart_distribution(df, col_to_plot, xlabel, ylabel, title):

    plot_data = df[col_to_plot].value_counts().head(10)
    plt.figure(figsize=(12, 6)) 
    plt.bar(range(len(plot_data)), plot_data.values)
    plt.xticks(range(len(plot_data)), plot_data.index, rotation=45, ha='right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
def plot_scatter(df, x_col, y_col, xlabel, ylabel, title):
    plt.figure(figsize=(12, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_histogram(df, column, xlabel, ylabel, title, nbins):
    plt.figure(figsize=(12, 6))
    plt.hist(df[column].dropna(), bins=nbins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

