Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis 
=================================================================

## Steps to run:
Download data_cleaning_and_eda.ipynb, raw_data.csv and cleaned_data.csv from part 1 folder of repo. Open the data_cleaning_and_eda.ipynb in google collab. Add the raw_data.csv and cleaned_data.csv in the files of collab and run it step by step.

## 1. Dataset Description:
This project focuses on identifying the underlying drivers of customer churning using the IBM Telco Customer Churn dataset by evaluating comprehensive customer profiles.
The data set has total of 7043 rows and 22 columns. The columns are gender, senior citizen status, partners, dependents, tenure, phone service, internet, security, backup, streaming, contract type, paperless billing, payment method,MonthlyCharges, TotalCharges, etc. This data set contains mixed data types, nul values, missing values, class imbalance and related to market.

## 2. NUll value Analysis
The referal code column has 84.61% (5959 rows) null values having no impact on the dataset's overall distribution.

## 3. Duplicate detection and removal
The dataset has 3 duplicate rows which has been indentified and removed. The shape has been changed form (7046,22) to (7043,22)

## 4. Data type correction
The total charges once the datatype has been changed from object to numeric has identified 0.16% (11 rows), were replaced by median in the data set. 16 columns were converted from object to category. Memory Before Conversion is 7409294 bytes and Memory After Conversion is 1068116 bytes which has reduction of 85.58%

## 5. skewness:
In the dataset TotalCharges is highly right skewed (+0.9616). When a distribution is positively skewed, the mean is pulled upward by extreme high values, making it unrepresentative of the typical central tendency. Imputing with the mean would falsely assign a high cumulative charge to new customers with. The median is robust to outliers and represents a more realistic central value. other skewness were tenure is 0.2395 and monktmonthly skewness is -0.2205.

## 6. Outlier Detection with IQR
No outliers were detected. We will retain all data points as they reflect valid customer charges.
Column: MonthlyCharges
  Bounds: [-46.02, 171.38] | Outliers: 0
Column: TotalCharges
  Bounds: [-4688.48, 8884.67] | Outliers: 0

## 7. Visualizations
1. **Line Plot (line_plot_monthly_charges.png)**: From this line graph we could understand that the MonthlyCharges ordered by row index and it shows fluctuations across the entire sequence, reflecting customer diversity.
2. **Bar Chart (bar_chart_monthly_charges.png)**: Bar charat compares mean monthly charges across contract categories from which we can inerpret Month-to-month contracts have the highest mean charge, while two-year contracts have the lowest.
3. **Histogram (histogram_most_skewed.png)**: Histogram helps us visualizes the right-skewed distribution of TotalCharges. The distribution is heavily clustered near $0 to $2,000, tapering off gradually.
4. **Scatter Plot (scatter_plot_tenure_vs_total_charges.png)**: Scatter plot is plots tenure against TotalCharges grouped by contract. There is a strong positive relationship. The pattern shows that higher monthly charges lead to higher total charges.
5. **Box Plot (box_plot_monthly_charges_by_churn.png)**: Box plot helps to compare MonthlyCharges distribution for churned vs retained customers. Churned customers have a higher median monthly charge compared to retained customers, indicating a correlation between high prices and churn.

## 8. Correlation heat map:
There's a strong positive correlation between tenure and TotalCharges, the longer a customer stays, the more they will pay in total. A moderately strong positive correlation exists between MonthlyCharges and TotalCharges,higher monthly charges would lead to higher total charges over time.SeniorCitizen shows weak positive correlations with MonthlyCharges.Alternative is customers who subscribe to more services would naturally have higher MonthlyCharges.These customers, especially if satisfied, might also have a longer tenure. Both higher MonthlyCharges and longer tenure would, in turn, lead to a higher TotalCharges.

## 9. a. Imputation strategy comparison
Based on the skewness analysis, TotalCharges is the most skewed column which is 0.9616 positively skewed. Since TotalCharges is positively skewed, its mean would be pulled upwards by high values, making the median a more robust. Therefore, the median will be used to impute the missing values in the TotalCharges column.

## 9. b. Spearman rank correlation:
Absolute Difference Matrix (|Spearman - Pearson|):
| | SeniorCitizen | tenure | MonthlyCharges | TotalCharges |
| :--- | :--- | :--- | :--- | :--- |
| **SeniorCitizen** | 0.000000 | 0.001989 | 0.000893 | 0.004838 |
| **tenure** | 0.001989 | 0.000000 | 0.028517 | 0.063297 |
| **MonthlyCharges** | 0.000893 | 0.028517 | 0.000000 | 0.013033 |
| **TotalCharges** | 0.004838 | 0.063297 | 0.013033 | 0.000000 |


TotalCharges and tenure has an diff of 0.0611. The Spearman correlation is higher because the relationship is monotonic but non-linear. MonthlyCharges and tenure has an diff of 0.0285. Suggests a weak, non-linear monotonic trend. MonthlyCharges and TotalCharges has an diff of 0.0136. The Pearson correlation is higher, suggesting the relation is closer to linear.we can rely on Spearman Rank Correlation to guide feature selection. Because Spearman captures monotonic, it prevents us from discarding features that have a strong relationship with the target but are not strictly linear which is better than pearson.

## 9. c. Grouped Aggregation

We grouped MonthlyCharges by Contract as below values,


| **Month-to-month** | 66.398490 | 26.926599 | 3875 |
| **One year** | 65.048608 | 31.840539 | 1473 |
| **Two year** | 60.770413 | 34.678865 | 1695 |

Highest Mean is for Month-to-month contracts ($66.398490).Highest Standard Deviation is for Two-year contracts ($34.678865). Implication of High Within-Group Std Dev is the high standard deviation indicates high variance within each contract group. This is a concern because it shows that contract type alone is insufficient to accurately predict monthly charges. The ratio of the highest mean to the lowest mean is $66.40 / $60.77 = 1.09. This ratio is close to 1.0, which indicates that the contract type alone does not carry a strong predictive signal for monthly charges. Additional features must be included to build a reliable predictive model.

## 10. Save the clean dataset
Final clean data set is saved as cleaned_data.csv
