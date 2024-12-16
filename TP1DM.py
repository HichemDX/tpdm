import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the data
data = pd.read_csv('./DatasetExos.csv', sep=';')

#  Write a function to display basic information about the dataset
def basic_info(df):
    print("The dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]))
    print("The columns are: ", df.columns)
    print("The data types are: ", df.dtypes)
    print("The first 5 rows are: ", df.head())

# call funtion basic_info


#  Write a function to calculate the central tendencies of an attribute
def mean_data(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')

    valid_values = df[col].dropna()  # prendre les valeurs non nulles
    some = sum(valid_values)
    count = len(df[col])
    mean = some / count
    print(f"The mean of '{col}' is:", mean)

# Call the function for the 'Acc_x' column

# Median
def median_data(df, col):
    sorted_values = df[col].sort_values()
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_values.iloc[mid - 1] + sorted_values.iloc[mid]) / 2
    else:
        median = sorted_values.iloc[mid]
    print(f"The median of '{col}' is:", median)

# Mode
def mode_data(df, col):
    values = df[col].dropna()
    freq = {}
    for val in values:
        if val in freq:
            freq[val] += 1
        else:
            freq[val] = 1
    max_freq = max(freq.values())
    modes = [key for key, value in freq.items() if value == max_freq]
    print(f"The mode of '{col}' is:", modes)




# Write a function to calculate the quartiles (Q0, Q1, Q2, Q3, Q4) of an attribute sans utiliser la fonction quantile de pandas
def quartiles_data(df, col):
    sorted_values = df[col].sort_values()
    count = len(sorted_values)
    q0 = sorted_values.iloc[0]
    q1 = sorted_values[count // 4]
    q2 = sorted_values[count // 2]
    q3 = sorted_values[3 * count // 4]
    q4 = sorted_values[count-1]
    print('the count is:', count)
    print(f"The quartiles of '{col}' are:")
    print(f"Q0: {q0}")
    print(f"Q1: {q1}")
    print(f"Q2: {q2}")
    print(f"Q3: {q3}")
    print(f"Q4: {q4}")


#  missing_values_info
def missing_values_info(df, col):
    values = df[col].dropna()
    count = len(values)
    total_count = len(df[col])
    poursantage = (total_count - count) * 100 / total_count


    print(total_count - count, "missing values found in column", col)
    print(f"Pourcentage of missing values: {poursantage:.2f}%")

#  Write a function to display the number of unique values for an attribute sans utiliser les fonctions nunique et unique de pandas et set
def unique_values_info(df, col):
    values = df[col].dropna()
    unique_values = []
    for val in values:
        if val not in unique_values:
            unique_values.append(val)
    print(f"The number of unique values in '{col}' is:", len(unique_values))

def scatter_plot(df, x_col, y_col):

    # Convert columns to numeric and drop rows with NaNs in these columns
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df_clean = df.dropna(subset=[x_col, y_col])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df_clean[x_col], df_clean[y_col], alpha=0.5, edgecolors='k')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatter Plot of {y_col} vs {x_col}')
    plt.grid(True)
    plt.show()


# Call the function for the 'Acc_x' and 'Acc_y' columns
scatter_plot(data, 'Acc_x', 'Acc_y')


#  Write a function to generate a histogram (and Bar chart) for an attribute
def histogram(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df_clean = df.dropna(subset=[col])

    plt.figure(figsize=(10, 6))
    plt.hist(df_clean[col], bins=20, edgecolor='k')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col}')
    plt.grid(True)
    plt.show()

# Call the function for the 'Acc_x' column
histogram(data, 'Acc_x')

# Write a function to generate a box plot for an attribute, both with and without outliers
def box_plot(df, col, show_outliers=True):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df_clean = df.dropna(subset=[col])

    plt.figure(figsize=(10, 6))
    if show_outliers:
        plt.boxplot(df_clean[col], vert=False, patch_artist=True, showfliers=True)
    else:
        plt.boxplot(df_clean[col], vert=False, patch_artist=True, showfliers=False)
    plt.xlabel(col)
    plt.title(f'Box Plot of {col} {"with" if show_outliers else "without"} outliers')
    plt.grid(True)
    plt.show()

# Call the function for the 'Acc_x' column
box_plot(data, 'Acc_x', show_outliers=False)


#Discretization into Equal-Width Intervals
def discretize_ewi_manual(df, col):

    # Convert column to numeric and remove rows with missing values in this column
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df_clean = df.dropna(subset=[col]).copy()  # Create a copy to avoid modifying the original

    # Calculate the number of non-missing values in the column
    n = len(df_clean[col])
    if n == 0:
        raise ValueError(f"No valid data in column {col} to discretize.")

    # Calculate the number of intervals (k) using Huntsberger's formula
    k = int(1 + (10 / 3) * np.log10(n))
    print(f"K:{k}")
    # Determine min and max values in the column
    min_val = df_clean[col].min()
    max_val = df_clean[col].max()

    # Handle case where min and max values are the same
    if min_val == max_val:
        raise ValueError(f"All values in column {col} are the same: {min_val}. Cannot discretize.")

    # Calculate the width of each interval
    width = (max_val - min_val) / k
    print(f"Width:{width}")
    # Generate intervals
    intervals = [(min_val + i * width, min_val + (i + 1) * width) for i in range(k)]

    # Create a new column for discretized values
    discretized_values = []

    # Assign categories based on intervals
    for value in df_clean[col]:
        assigned = False
        for i, (lower, upper) in enumerate(intervals):
            if lower <= value < upper:
                discretized_values.append(f'Interval {i + 1}')
                assigned = True
                break
        if not assigned:
            # If the value is equal to the max_val, it should belong to the last interval
            discretized_values.append(f'Interval {k}') if value == max_val else discretized_values.append(None)

    # Add the discretized column to the DataFrame
    df_clean['Discretized'] = discretized_values

    # Print the result for visibility
    print(df_clean[['Discretized', col]])

    return df_clean[['Discretized', col]]


# Usage example for the 'Acc_x' column
discretized_data = discretize_ewi_manual(data, 'Acc_x')




################################
def discretize_ewi_with_averages(df, col):

    # Convert column to numeric and remove rows with missing values
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df_clean = df.dropna(subset=[col]).copy()

    # Calculate the number of non-missing values
    n = len(df_clean[col])
    if n == 0:
        raise ValueError(f"No valid data in column {col} to discretize.")

    # Calculate the number of intervals (k) using Huntsberger's formula
    k = int(1 + (10 / 3) * np.log10(n))

    # Determine min and max values in the column
    min_val = df_clean[col].min()
    max_val = df_clean[col].max()

    # Handle case where min and max values are the same
    if min_val == max_val:
        raise ValueError(f"All values in column {col} are the same: {min_val}. Cannot discretize.")

    # Calculate the width of each interval
    width = (max_val - min_val) / k

    # Generate intervals and calculate the average of each interval
    intervals = [(min_val + i * width, min_val + (i + 1) * width) for i in range(k)]
    interval_averages = [(lower + upper) / 2 for lower, upper in intervals]

    # Create a new column for discretized values
    discretized_values = []

    # Assign the average value of the interval based on where the value falls
    for value in df_clean[col]:
        assigned = False
        for i, (lower, upper) in enumerate(intervals):
            if lower <= value < upper:
                discretized_values.append(interval_averages[i])
                assigned = True
                break
        if not assigned:
            # If the value is equal to the max_val, assign it to the last interval's average
            discretized_values.append(interval_averages[-1]) if value == max_val else discretized_values.append(None)

    # Add the discretized column to the DataFrame
    df_clean['Discretized'] = discretized_values

    # Print the result for visibility
    print(df_clean[[col, 'Discretized']])

    return df_clean[[col, 'Discretized']]

discretized_data = discretize_ewi_with_averages(data, 'Acc_x')


# funtion for normalisation discretized_data using Min-Max Normalization

def min_max_normalization(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df_clean = df.dropna(subset=[col]).copy()

    # Calculate the min and max values of the column
    min_val = df_clean[col].min()
    max_val = df_clean[col].max()

    # Perform Min-Max Normalization
    df_clean['Normalized'] = (df_clean[col] - min_val) / (max_val - min_val)

    # Print the result for visibility
    print(df_clean[[col, 'Normalized']])

    return df_clean[[col, 'Normalized']]

# Usage example for the 'Acc_x' column
normalized_data = min_max_normalization(discretized_data, 'Acc_x')

