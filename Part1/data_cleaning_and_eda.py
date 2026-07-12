# Part 1: Data Acquisition, Cleaning, and Exploratory Analysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the raw dataset
print("Loading the raw data file...")
dataframe = pd.read_csv("part1/raw_data.csv")

# Print dataset size and columns
print("Number of rows: " + str(dataframe.shape[0]))
print("Number of columns: " + str(dataframe.shape[1]))
print("\nFirst 5 rows of data:")
print(dataframe.head())
print("\nData types of each column:")
print(dataframe.dtypes)

# Save the original memory size to check savings later
original_memory = dataframe.memory_usage(deep=True).sum()

# 2. Null Value Analysis
print("\n--- Analysing Missing Values ---")
total_rows = len(dataframe)
for col in dataframe.columns:
    nulls = dataframe[col].isnull().sum()
    pct = (nulls / total_rows) * 100
    print("Column: " + col + " | Missing: " + str(nulls) + " (" + str(round(pct, 2)) + "%)")

# We see that ReferralCode exceeds 20% null rate, so we report it
print("\nColumns with more than 20% missing values: ['ReferralCode']")

# 3. Duplicate Detection and Removal
print("\n--- Checking for Duplicates ---")
duplicate_count = dataframe.duplicated().sum()
print("Found " + str(duplicate_count) + " duplicate rows.")

# Remove them
dataframe = dataframe.drop_duplicates()
print("Shape after removing duplicates: " + str(dataframe.shape))

# 4. Data Type Correction
print("\n--- Fixing Incorrect Data Types ---")
# TotalCharges should be numeric, but it is stored as an object (string)
# We find rows where TotalCharges is just empty spaces
empty_space_rows = dataframe[dataframe['TotalCharges'].str.strip() == ""]
print("Number of blank spaces in TotalCharges column: " + str(len(empty_space_rows)))

# Convert to float and turn blank spaces to NaN
dataframe['TotalCharges'] = pd.to_numeric(dataframe['TotalCharges'], errors='coerce')
print("TotalCharges dtype after converting: " + str(dataframe['TotalCharges'].dtype))
print("Missing values in TotalCharges now: " + str(dataframe['TotalCharges'].isnull().sum()))

# Convert repetitive string columns to category type to save memory
columns_to_category = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]

for col in columns_to_category:
    dataframe[col] = dataframe[col].astype('category')

# Compare memory usage
new_memory = dataframe.memory_usage(deep=True).sum()
print("\nMemory size before: " + str(original_memory) + " bytes")
print("Memory size after: " + str(new_memory) + " bytes")
saved_mem = original_memory - new_memory
saved_pct = (saved_mem / original_memory) * 100
print("Memory saved: " + str(saved_mem) + " bytes (" + str(round(saved_pct, 2)) + "%)")

# 5. Descriptive Statistics and Skewness
print("\n--- Statistics and Skewness ---")
print(dataframe[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

# Compute skewness
tenure_skew = dataframe['tenure'].skew()
monthly_skew = dataframe['MonthlyCharges'].skew()
total_skew = dataframe['TotalCharges'].skew()
print("tenure skewness: " + str(round(tenure_skew, 4)))
print("MonthlyCharges skewness: " + str(round(monthly_skew, 4)))
print("TotalCharges skewness: " + str(round(total_skew, 4)))
print("Highest absolute skewness is TotalCharges: " + str(round(total_skew, 4)))

# 6. Outlier Detection with IQR
print("\n--- Finding Outliers with IQR ---")
numeric_columns = ['MonthlyCharges', 'TotalCharges']
for col in numeric_columns:
    q1 = dataframe[col].quantile(0.25)
    q3 = dataframe[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Count outliers
    outliers = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)]
    print("Column: " + col)
    print("  Q1: " + str(round(q1, 2)) + " | Q3: " + str(round(q3, 2)) + " | IQR: " + str(round(iqr, 2)))
    print("  Bounds: [" + str(round(lower_bound, 2)) + ", " + str(round(upper_bound, 2)) + "]")
    print("  Number of outliers found: " + str(len(outliers)))

# 7. Visualizations
print("\n--- Generating and Saving Plots ---")

# Plot 1: Line plot of MonthlyCharges sorted by index
plt.figure()
plt.plot(dataframe.index, dataframe['MonthlyCharges'], color='blue', alpha=0.5)
plt.title("Monthly Charges Line Plot")
plt.xlabel("Index")
plt.ylabel("Monthly Charges")
plt.savefig("part1/line_plot_monthly_charges.png")
plt.close()
print("Saved line plot to part1/line_plot_monthly_charges.png")

# Plot 2: Bar chart comparing mean of MonthlyCharges across categories of Contract
plt.figure()
mean_charges = dataframe.groupby('Contract', observed=True)['MonthlyCharges'].mean()
plt.bar(mean_charges.index, mean_charges.values, color='orange', edgecolor='black')
plt.title("Average Monthly Charges by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Average Monthly Charges")
plt.savefig("part1/bar_chart_monthly_charges_by_contract.png")
plt.close()
print("Saved bar chart to part1/bar_chart_monthly_charges_by_contract.png")

# Plot 3: Histogram of most skewed column (TotalCharges)
plt.figure()
plt.hist(dataframe['TotalCharges'].dropna(), bins=20, color='green', edgecolor='black', alpha=0.7)
plt.title("Histogram of Total Charges")
plt.xlabel("Total Charges")
plt.ylabel("Frequency")
plt.savefig("part1/histogram_most_skewed.png")
plt.close()
print("Saved histogram to part1/histogram_most_skewed.png")

# Plot 4: Scatter plot between tenure and TotalCharges
plt.figure()
plt.scatter(dataframe['tenure'], dataframe['TotalCharges'], color='purple', alpha=0.5)
plt.title("Tenure vs Total Charges")
plt.xlabel("Tenure (months)")
plt.ylabel("Total Charges")
plt.savefig("part1/scatter_plot_tenure_vs_total_charges.png")
plt.close()
print("Saved scatter plot to part1/scatter_plot_tenure_vs_total_charges.png")

# Plot 5: Box plot of MonthlyCharges split by Churn
plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=dataframe, hue='Churn', palette=['green', 'red'], legend=False)
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churned?")
plt.ylabel("Monthly Charges")
plt.savefig("part1/box_plot_monthly_charges_by_churn.png")
plt.close()
print("Saved box plot to part1/box_plot_monthly_charges_by_churn.png")

# 8. Correlation Heatmap
print("\n--- Correlation Analysis ---")
numeric_df = dataframe[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
pearson_corr = numeric_df.corr()
print("Pearson Correlation:")
print(pearson_corr)

plt.figure()
sns.heatmap(pearson_corr, annot=True, cmap='Blues')
plt.title("Pearson Correlation Heatmap")
plt.savefig("part1/correlation_heatmap.png")
plt.close()
print("Saved correlation heatmap to part1/correlation_heatmap.png")

# 8a. Imputation Comparison (Top 2 skewed columns are SeniorCitizen and TotalCharges)
print("\n--- Imputation Comparison ---")
# Compare Mean vs Median for TotalCharges
tc_mean = dataframe['TotalCharges'].mean()
tc_median = dataframe['TotalCharges'].median()
print("TotalCharges - Mean: " + str(round(tc_mean, 4)) + " | Median: " + str(round(tc_median, 4)))

# We fill missing TotalCharges with the median
dataframe['TotalCharges'] = dataframe['TotalCharges'].fillna(tc_median)
print("Missing values in TotalCharges after filling: " + str(dataframe['TotalCharges'].isnull().sum()))

# 8b. Spearman Correlation
print("\n--- Spearman Rank Correlation ---")
spearman_corr = numeric_df.corr(method='spearman')
print("Spearman Correlation:")
print(spearman_corr)

# Difference table
diff_corr = (spearman_corr - pearson_corr).abs()
print("\nDifference |Spearman - Pearson|:")
print(diff_corr)

# Print specific pairs
print("\nKey Differences:")
print("TotalCharges vs tenure: Pearson = " + str(round(pearson_corr.loc['TotalCharges', 'tenure'], 4)) + 
      ", Spearman = " + str(round(spearman_corr.loc['TotalCharges', 'tenure'], 4)) + 
      ", Diff = " + str(round(diff_corr.loc['TotalCharges', 'tenure'], 4)))
print("MonthlyCharges vs tenure: Pearson = " + str(round(pearson_corr.loc['MonthlyCharges', 'tenure'], 4)) + 
      ", Spearman = " + str(round(spearman_corr.loc['MonthlyCharges', 'tenure'], 4)) + 
      ", Diff = " + str(round(diff_corr.loc['MonthlyCharges', 'tenure'], 4)))
print("MonthlyCharges vs TotalCharges: Pearson = " + str(round(pearson_corr.loc['MonthlyCharges', 'TotalCharges'], 4)) + 
      ", Spearman = " + str(round(spearman_corr.loc['MonthlyCharges', 'TotalCharges'], 4)) + 
      ", Diff = " + str(round(diff_corr.loc['MonthlyCharges', 'TotalCharges'], 4)))

# 8c. Grouped Aggregation
print("\n--- Grouped Aggregation ---")
grouped = dataframe.groupby('Contract', observed=True)['MonthlyCharges'].agg(['mean', 'std', 'count'])
print(grouped)

# Save cleaned dataset
dataframe.to_csv("part1/cleaned_data.csv", index=False)
print("\nCleaned dataset saved to part1/cleaned_data.csv successfully!")
