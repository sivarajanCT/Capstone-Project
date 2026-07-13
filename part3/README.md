Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline
=====================================================================

## Steps to run:
Download advanced_models.ipynb and cleaned_data.csv from part 3 folder of repo. Open the advanced_models.ipynb in google collab. Add the cleaned_data.csv in the files of collab and run it step by step.

## 1. Decision Tree baseline
Unconstrained Decision Tree: Training Accuracy is 99.86% and Test Accuracy is 72.25%. Decision tree exhibits severe overfitting. It achieves nearly 100% accuracy on the training set but drops drastically to 72.25% on the test set.Decision trees are described as high-variance models because they grow recursively by greedily splitting nodes based on the feature that maximizes information gain or Gini impurity reduction at that specific point. A single deep tree fits the training data noise, making the final structure highly sensitive to small variations in the training sample.

## 2. Controlled Decision Tree
Controlled Decision Tree: Training Accuracy is 79.64% and Test Accuracy is 80.20% .The controlled tree successfully resolves this overfitting. By restricting tree growth, the train-test accuracy gap is virtually eliminated, and test set accuracy rises from 72.25% to 80.20%. max_depth=5, limits the maximum height of the tree, preventing it from creating complexity and min_samples_split=20, Prevents a node from splitting if it contains fewer than 20 samples.

## 3. Gini vs. Entropy Comparison
Gini impurity test accuracy is 80.20% and entropy test accuracy is 80.13%

Formulas:

- **Gini Impurity**: G=1−∑Ci=1(pi)2
- **Entropy**: H=−∑Ci=1pilog2(pi)

### Gini = 0 Interpretation
A node has Gini impurity = 0 when it is perfectly pure. This occurs when all samples in that node belong to a single class. This represents a leaf node that can make a 100% certain classification.

---

## 4. Random Forest

RF Test Acc: 0.8055 | Test AUC: 0.8612

### Top 5 Features by Random Forest Importance
| Rank | Feature | Importance Score |
| :--- | :--- | :--- |
| 1 | `tenure` | 0.1990 |
| 2 | `TotalCharges` | 0.1898 |
| 3 | `Contract` | 0.1295 |
| 4 | `InternetService` | 0.0815 |
| 5 | `PaymentMethod_Electronic check` | 0.0472 |

### Feature Importance in Random forest vs Linear Coefficients
Random Forest feature importance measures the average reduction in Gini impurity brought by splits on a given feature, weighted by the number of samples routing through those splits, averaged across all 100 trees.Unlike linear regression coefficients, which indicate the direction and magnitude of a linear relationship, RF importance is non-negative and represents how informative a feature is for making classification splits.

### Bagging
Bagging reduces model variance by building a collection of decorrelated, deep decision trees in parallel using
- **Bootstrap Sampling**: Each tree is trained on a random sample of the training data drawn with replacement.
- **Feature Subspacing**: At each split, only a random subset of features is considered. This decorrelates the trees, preventing a single highly dominant feature from being selected at the root of every tree.
- **Ensemble Averaging**: By averaging the predictions of these diverse, high-variance trees, their individual variances cancel out, leaving the ensemble with low.

## 4. a.   Gradient Boosting

GB Test Acc: 0.8119 | Test AUC: 0.8613

## 4. b.   Feature Ablation Study

Using Random Forest feature importances, we identified the 5 features with the lowest importance: OnlineSecurity_No internet service, TechSupport_No internet service, DeviceProtection_No internet service, MultipleLines_No phone service, and PhoneService_Yes.

- **Full Model Test AUC (all 27 features)**: **0.86119**
- **Reduced Model Test AUC (5 lowest features removed)**: **0.86148**
- **AUC Difference**: **-0.00029** (The reduced model performed slightly better).

### Production Trade-off Discussion
The feature ablation study reveals that the 5 lowest-importance features were contributing noise rather than predictive signal, since removing them led to a minor AUC increase (+0.00029). 
In production, deploying the reduced 22-feature model is highly advantageous:
- **Lower Inference Cost**: Processing fewer features reduces API payload size, data validation pipelines, and computation time.
- **Maintenance and Monitoring**: Fewer features mean a smaller surface area for data drift, schema breaks, and data pipelines failing.

## 5. Cross-validated Comparison

The data for 5-fold cross-validation using StratifiedKFold on the training set:

Logistic Regression - CV Mean AUC: 0.8404 | Std: 0.0115
Controlled Tree - CV Mean AUC: 0.8347 | Std: 0.0129
Random Forest - CV Mean AUC: 0.8341 | Std: 0.0152
Gradient Boosting - CV Mean AUC: 0.8431 | Std: 0.0159

### Why Cross-Validation is More Reliable
A single train-test split evaluates the model on a single static subset of the data, which may be easy or hard to predict by chance. Cross-validation partitions the dataset into 5 distinct sets and rotates them so every sample is used for testing exactly once. The mean and standard deviation of these sets provide an unbiased result of the model's true performance.

## 6. Hyperparameter Tuning with GridSearchCV

- **Best Hyperparameters**: {'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 200}
- **Best 5-Fold CV Mean AUC**: 0.8402

### Grid Search vs. Randomized Search Trade-off
Grid Search is exhaustive and guarantees finding the optimal parameter combination within the defined grid, but it scales exponentially with the number of parameters and is highly computationally expensive. Randomized Search samples configurations randomly from a defined distribution.

## 7. Manual Learning Curve

| Fraction | Train AUC | Test AUC |
| :--- | :--- | :--- |
| 0.2 | 0.9351 | 0.8617 |
| 0.4 | 0.9264 | 0.8622 |
| 0.6 | 0.9162 | 0.8622 |
| 0.8 | 0.9111 | 0.8645 |
| 1.0 | 0.9077 | 0.8659 |

### Learning Curve Interpretation
1. **Training AUC Trend**: Decreases as the dataset grows (from 0.9351 to 0.9077). This is expected because a smaller dataset is easier for a model to fit perfectly, whereas a larger dataset introduces more variance.
2. **Test AUC Trend**: Increases with more training data (from 0.8617 to 0.8659), showing that the model works from more samples to learn general patterns.
3. **Data vs. Capacity Limitation**: The model is currently limited by model capacity rather than data quantity. Although the test AUC is still rising slightly, the improvement is very small (+0.004). The gap between Training AUC and Test AUC shows some remaining variance. To improve, we need to increase model capacity or engineer more informative features.

## Model Recommendation

We recommend the **Tuned Random Forest Pipeline** to the client. This model achieved the highest overall Test AUC of 0.8659. By averaging predictions, it provides excellent stability and is robust to minor data changes. 
