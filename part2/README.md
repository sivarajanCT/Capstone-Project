Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate
=========================================================================

## steps to run
Download model_training.ipynb and cleaned_data.csv from part 1 folder of repo. Open the model_training.ipynb in google collab. Add the cleaned_data.csv in the files of collab and run it step by step.

## 1. Label Definitions
Regression Label (y_reg) is MonthlyCharges which represents the monthly amount charged to the customer in USD. This is useful for predicting revenue and customer account value. Classification Label (y_clf) is Churn which represents whether the customer left the company within the last month.

## 2. Encode categorical columns
Ordinal Encoding are applied to `Contract` and `InternetService`.Contract was mapped as: Month-to-month as 0, One year as 1, Two year as 2. This has a natural order since longer contract periods represent higher customer commitment. InternetService was mapped as: No as 0, DSL as 1, Fiber optic as 2. This represents a natural progression in speed and pricing tiers. One-Hot Encoding is applied to all other nominal categorical features gender,Partner,Dependents,PhoneService,MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, PaymentMethod). We set drop_first=True to drop the first category, avoiding multicollinearity.

Why One-Hot Encoding Avoids False-Ordinal Relationships : 
If we had applied label encoding to categories such as PaymentMethod, a machine learning model would falsely assume a numeric order. A linear model would interpret Credit card (3) as "greater than" that Bank transfer (2) is midway between them. This false numerical structure forces the model to search for linear relationships that do not exist, leading to poor generalization. One-hot encoding creates independent binary dummy variables.

## 3. Leak-free train-test split and scaling

We split the features X_encoded and labels y_reg and y_clf into training (80%) and testing (20%) sets using train_test_split with a fixed random_state=42. We fitted the StandardScaler only on the training features (X_train) and then transformed both X_train and X_test using that fitted scaler.Fitting the scaler on the full dataset before splitting would constitute data leakage. The scaler calculates the mean and standard deviation of the features to standardize them. If the test set is included in these calculations, information about the distribution, mean, and range of the unseen test data is leaked into the training process. The model would train on scaled values that are influenced by the test set, leading to overfitting.

## 4. Regression Model — Linear Regression
Regression Results:

**Linear Regression** : Test MSE = 1.0980 | R² = 0.9988
**Ridge Regression** (alpha=1.0): Test MSE = 1.0977 | R² = 0.9988

### Linear regression Coefficients Interpretation:

1. InternetService (coeff = 19.3690)
2. StreamingTV_Yes (coeff = 4.8517)
3. StreamingMovies_Yes (coeff = 4.8404)
4. PhoneService_Yes (coeff = 2.9724)
5. MultipleLines_No phone service (coeff = -2.9724)

Large Positive Coefficient: The standard deviation increase in the scaled feature is associated with a corresponding unit increase in the predicted monthly charge equal to the coefficient.
Large Negative Coefficient: The standard deviation increase in the feature is associated with a corresponding unit decrease in the predicted monthly charge.

Ridge vs Linear Coefficient Profiles
Ridge regression adds an L2 regularization penalty to the linear loss function. This penalty shrinks the coefficients toward zero, reducing model variance by preventing any single feature from dominating, which is especially helpful when features are highly correlated. The alpha parameter controls this trade-off: alpha = 0 is standard linear model, while larger values force stronger regularization. Because monthly charges are almost completely determined by service configurations, the linear model fits the training set with extremely low variance already (R² = 0.9988), resulting in almost identical performance.

## 5. Classification Model — Logistic Regression

Class Imbalance Handling
The training set contains, No Churn as 4,138 samples (73.45%) and Churn as 1,496 samples (26.55%)

Because the minority class represents 26.55% of the data (< 35%), we addressed this imbalance using class_weight='balanced' in the Logistic Regression constructor.

Confusion Matrix:
  [[749  287]   <- [True Negatives, False Positives]
   [ 65  308]]   <- [False Negatives, True Positives]

Accuracy: 0.75
Precision: 0.52
Recall: 0.83
F1-Score: 0.64
ROC-AUC: 0.8621

Precision and Recall Formulas:

**Precision**: Precision= True Positives (TP)/(True Positives (TP)+False Positives (FP))
**Recall**: Recall = True Positives (TP)/(True Positives (TP)+False Negative (FN))

In customer churn prediction, Recall is more important than Precision. A false negative is highly costly, as the customer leaves and the company loses their revenue. A false positive only incurs the minor cost of a retention campaign. Therefore, we prioritize a high recall even if it leads to a lower precision.

### AUC Interpretation :

The AUC of 0.8621 indicates that there is an 86.21% probability that a randomly selected customer who churned will have a higher predicted churn probability than a randomly selected customer who stayed.

## 5b .Decision-Threshold Sensitivity

| Threshold | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| 0.3 | 0.4291 | 0.941 | 0.5894 |
| 0.4 | 0.4668 | 0.8847 | 0.6111 |
| 0.5 | 0.5176 | 0.8257 | 0.6364 |
| 0.6 | 0.5756 | 0.7453 | 0.6495 |
| 0.7 | 0.6503 | 0.6381 | 0.6441 |

F1-Maximizing Threshold: 0.60 (F1 = 0.6495).
Threshold Optimization: To optimize for Recall, we would lower the threshold (0.40). At a threshold of 0.40, recall rises to 88.47%. The cost of doing this is a lower precision (46.68%), which means a higher rate of false positives (wasted marketing expenses on customers who were not going to churn).

## 6. Regularization experiment on Logistic Regression

Regularization Comparison (C=0.01): 
| Model | Precision | Recall | ROC-AUC |
| :--- | :--- | :--- | :--- |
| C=1.0 | 0.5176 | 0.8257 | 0.8621 |
| C=0.01 | 0.5193 | 0.8311 | 0.8599 |

The parameter C represents the inverse of regularization strength. A smaller C (0.01) imposes stronger L2 regularization, constraining coefficient to prevent overfitting. In this case, reducing C to 0.01 slightly decreased the ROC-AUC score to 0.8599.

## 7. Bootstrap Confidence Interval for AUC Difference

**Mean AUC Difference**: 0.002261
**95% Confidence Interval**: [-0.000549, 0.004756]

Because the 95% confidence interval includes zero, spanning from -0.000549 to 0.004756, the performance advantage of the baseline model (C=1.0) is not statistically significant. This means that we cannot confidently assert that the C=1.0 model is superior to the C=0.01 model; the minor difference could be due to random variation in the test sample.
