20231652 Tomás Pires

## Week 2
DT: Train accuracy: 0.829
Test accuracy:  0.629
Gap (train - test): +0.199

LR: Train accuracy: 0.680
Test accuracy:  0.677
Gap (train - test): +0.003

Conclusion: The Logistic Regression demonstrates better generalization than the Decision Tree. Although the Decision Tree achieves higher training accuracy, it has lower test accuracy giving a large train-test gap indicating overfitting. While the Logistic Regression achieves a lower training accuracy, but with a train-test gap of only 0.003 points.

## Week 3

DT: Train accuracy: 0.789
Test accuracy:  0.611
Gap (train - test): +0.178

LR: Train accuracy: 0.672
Test accuracy:  0.667
Gap (train - test): +0.005

This week the file data_diagnostic.py was added. This module provides generic, configuration-driven data quality diagnostics that run before preprocessing. It is designed to work across datasets without dataset-specific column names or assumptions, those can be changed through config.yaml.
It provides three main checks:
- Missingness mechanism testing: Tests whether missing values in a target column are associated with other predictors using a chi-square test and bias-corrected Cramér’s V, helping identify potential missingness patterns.
- Invalid-value detection: Applies configurable minimum and maximum domain rules to identify invalid numeric values. Detected violations are converted to NaN, and a report records the number of violations per column.
- Duplicate detection: Checks for both exact duplicate rows and repeated identifiers, allowing duplicates to be detected even when records differ in other fields.

The diagnostics section of config.yaml that defines dataset-specific rules used by the data diagnostics stage, keeping the Python code generic and reusable across datasets,it allows the same pipeline to be applied to other datasets by changing the YAML configuration rather than modifying the Python code.

The preprocessing.py module was improved to include dataset cleaning doing:
- Category standardization: Converts inconsistent categorical values to canonical labels defined in config.yaml and converts placeholder tokens to missing values.
- Numeric conversion: Converts numeric columns that were loaded as text into numeric types, treating configured placeholder tokens as missing values.
- Validity and missing-value handling: Applies configured domain rules and converts invalid values to NaN.
- Duplicate removal: Removes exact duplicate rows and repeated records based on the configured identifier column.
- Redundant-column removal: Drops configured columns that are considered redundant or unsuitable due to multicollinearity.
- Missingness indicators: Can create a column that indicates where the missingness pattern is considered informative, preserving that information through later imputation. (not used yet)

Some additonal (basic)preprocessing had to be done to the imputation to run the logistic regression
- Numeric preprocessing: Missing numeric values are median-imputed and features are standardized using StandardScaler.
- Categorical preprocessing: Missing categorical values are filled using the most frequent category, then categorical variables are one-hot encoded with unknown categories handled safely.

Conclusion this week results show a small decrease in both training and test accuracy compared to the previous week. 
- For the Decision Tree (DT), training accuracy decreased from 0.829 to 0.789, while test accuracy decreased from 0.629 to 0.611. The train-test gap also became slightly smaller (0.199 to 0.178). This may suggest that the cleaning steps removed or corrected some patterns that the tree was able to recognise. However, the smaller gap indicates slightly less overfitting.
- For Logistic Regression (LR), training accuracy decreased from 0.680 to 0.672, and test accuracy decreased from 0.677 to 0.667.


## Week 4 

Dummy: Cross-validation (5 stratified folds, metric: accuracy)

Train       mean = 0.549   std = 0.000
Validation  mean = 0.549   std = 0.000
Gap         mean = -0.000   std = 0.000

LR: Cross-validation (5 stratified folds, metric: accuracy)

Train       mean = 0.677   std = 0.003
Validation  mean = 0.674   std = 0.012
Gap         mean = +0.003   std = 0.014

DT: Cross-validation (5 stratified folds, metric: accuracy)

Train       mean = 0.696   std = 0.010
Validation  mean = 0.607   std = 0.012
Gap         mean = +0.090   std = 0.020

RF: Cross-validation (5 stratified folds, metric: accuracy)

Train       mean = 0.736   std = 0.014
Validation  mean = 0.648   std = 0.019
Gap         mean = +0.089   std = 0.020


This week, the pipeline was improved by replacing the single train-test evaluation with 5-fold stratified cross-validation. Having the models become more reliable 

Another changes added this week were:

- The Preprocessing is included inside the model pipeline. This ensures that imputation, encoding and scaling are fitted only on the training portion of each fold, reducing the risk of data leakage. 
- The enconding used was target encoding.
- The scaler used was robust scaling, being less affected by extreme values.
- For numeric imputation: Missing numerical values are filled using the median.
- For categorical imputation: Missing categorical values are filled using the most frequent category.
- Training and validation accuracy are recorded for every fold, together with the train-validation gap. This makes it easier to identify potential overfitting and compare how well the models generalize.
- Predictions are generated for each observation using a model that was not trained on that observation. These predictions can be used for classification and fairness evaluation without using the locked test set.
- The test set is kept separate and is not used during cross-validation or model selection. It can therefore be reserved for the final evaluation of the selected model.
- A Dummy Classifier was added as a baseline, achieving a validation accuracy of 0.549. This provides a minimum benchmark that the machine-learning models should outperform.
- The Random Forest model was added.
 
Regarding this week Results:

- Dummy Classifier, both Training and validation accuracy are 0.549, with essentially no gap.
- Logistic Regression: Achieved the highest mean validation accuracy at 0.674, with a mean training accuracy of 0.677 and a very small gap of 0.003. The similar training and validation performance indicates stable generalization across the folds.
- Decision Tree: Achieved a training accuracy of 0.696 and validation accuracy of 0.607, reducing the gap to 0.090. Although the gap is smaller, it still shows considerably more overfitting than Logistic Regression.
- Random Forest: Achieved a training accuracy of 0.736 and validation accuracy of 0.648, with a gap of 0.089. It performs better on validation data than the Decision Tree, but also shows a noticeable overfitting.

Conclusion

The Week 4 results reinforce the pattern observed in previous weeks. Logistic Regression continues to show the strongest generalization, with the highest validation accuracy and the smallest train-validation gap. The Decision Tree continues to show signs of overfitting, although its gap has decreased compared with Week 3. Random Forest improves validation performance compared with the individual Decision Tree, but still has a substantially larger train-validation gap than Logistic Regression.
More relevant is the fact this week improves the reliability of the models. Instead of drawing conclusions from one train-test split, the models are now compared using five stratified folds, while the final test set remains untouched for later evaluation.

## Weeks 5/6

Weeks 5 and 6 were done together, focusing on hyperparameter tuning and ensemble learning methods. Hyperparameter tuning was introduced to optimize model configurations and improve predictive performance(tuning.py was added). Additionally, two new ensemble models, Oblique Random Forest and AdaBoost, were implemented to expand the range of models evaluated and compare their performance against the existing classifiers.

### Table of results
| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Logistic regression | 0.674 ± 0.012 (+0.003) | 0.677 ± 0.013 (+0.001) | 0.678 | +0.001 |'model__C': 0.4145086347025647, 'model__solver': 'liblinear', 'model__max_iter': 1328 |
| Decision tree | 0.602 ± 0.020 (+0.094) | 0.673 ± 0.010 (+0.010) | 0.679 | +0.003 |'model__max_depth': 9, 'model__min_samples_leaf': 112, 'model__criterion': 'entropy' |
| Random forest | 0.642 ± 0.019 (+0.094) | 0.680 ± 0.014 (+0.008 ) | 0.682 | +0.003 | 'model__max_depth': 7, 'model__min_samples_leaf': 24, 'model__max_features': 'sqrt' |
| Oblique random forest | 0.645 ± 0.011 (+0.120) | 0.682 ± 0.014 (+0.009) | 0.683 | -0.001 | model__tree__max_depth': 5, 'model__tree__min_samples_leaf': 1, 'model__estimator__project__n_components': 10 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.681 ± 0.012 (+0.006) | 0.685 | +0.001 |`model__max_depth': 4, 'model__n_estimators': 176, 'model__learning_rate': 0.14640181866811797|

From these results, the best model for now is AdaBoost, which achieves a nested cross-validation accuracy of 0.681 ± 0.012, with a small train-validation gap of +0.006. Although Oblique Random Forest achieves a slightly higher accuracy (0.682), AdaBoost has a lower standard deviation and a smaller train-validation gap, being more consistent regarding performance with less overfitting.

## Week 7
This week, the pipeline was expanded with new models and improvements to model evaluation, hyperparameter tuning, and results reporting.
 - Additional models such as Gradient Boosting, Histogram Gradient Boosting, and Naive Bayes were added, allowing for more algorithms to be compared.
 - The evaluation process was improved to support different scoring metrics instead of relying only on accuracy, having the selection_metric as "roc_auc".
 - Cross-validation reports, out-of-fold classification, fairness evaluations as well as tuning and nested cross-validation function were updated to support the new evaluation framework.
 - data/run.csv was added registering the performance of the runs of the different models.
Overall, Week 7 focused on expanding the available models and making model evaluation more flexible and comprehensive, allowing comparisons based on multiple performance metrics rather than accuracy alone.

### Scoreboard
| Model | Default: Roc_auc mean ± std | Tuned: Roc_auc mean ± std | Best hyperparameters |
|---|---|---|---|
| Dummy(floor) | 0.500 | 0.500 |   |
| naive_bayes | 0.689 ± 0.011 | 0.707 ± 0.011 |'var_smoothing': 0.09069648209152663 |
| logistic_regression | 0.719 ± 0.014 | 0.718 ± 0.013 |'model_C': 0.19069966103000435, 'solver': 'lbfgs', 'max_iter': 1271 |
| decision_tree | 0.600 ± 0.017 | 0.712 ± 0.016 |'max_depth': 12, 'min_samples_leaf': 107, 'criterion': 'gini' |
| random_forest | 0.686 ± 0.018 | 0.726 ± 0.015 |'max_depth': 30, 'min_samples_leaf': 33, 'max_features': 'sqrt' |
| oblique_random_forest | 0.687 ± 0.017 | 0.727 ± 0.014 |'estimator__tree__max_depth': 13, 'estimator__tree__min_samples_leaf': 91, 'estimator__project__n_components': 8 |
| adaboost | 0.723 ± 0.016 | 0.726 ± 0.014 |'estimator__max_depth': 4, 'n_estimators': 55, 'learning_rate': 0.16738085788752127 |
| gradient_boosting | 0.726 ± 0.015 | 0.728 ± 0.014 |'n_estimators': 216, 'learning_rate': 0.016814344160113716, 'max_depth': 3, 'min_samples_leaf': 1, 'subsample': 0.5781493080610196 |
| hist_gradient_boosting | 0.715 ± 0.022 | 0.729 ± 0.015 |'max_iter': 364, 'learning_rate': 0.07844942622426945, 'max_leaf_nodes': 2, 'min_samples_leaf': 6, 'l2_regularization': 0.0026935425710306635 |


From these results and using ROC-AUC as the selected metric, the Gradient Boosting, Adaboost and Histogram Gradient Boosting models achieved the highest reported performance(~0.728), though their advantage over the other models is not that significant, with Naive bayes having the worst performance with(0.707).

## Week 8
This week, the pipeline was extended with a new model and a feature selection step, along with updates to configuration and results reporting.
- XGBoost was added to the available models, with its own hyperparameter search space in config.yaml.
- A feature selection step was added to the pipeline through a new VoteSelector class. It keeps only the features that receive at least a minimum number of votes (min_votes) from the selection methods. This minimum is tuned together with the model’s hyperparameters, so the number of selected features is optimized rather than fixed by hand.
- The scoreboard now includes a feature_selection column.

Overall, Week 8 focused on adding a stronger gradient boosting model and on reducing the feature set to the most relevant variables.

### Scoreboard
| Model | feature_selection | roc_auc (std) | untuned |
|---|---|---|---|
|naive_bayes|min_votes 1|0.708 (0.013)|0.704|
|logistic_regression|min_votes 2|0.716 (0.011)|0.716|
|decision_tree|min_votes 2|0.718 (0.012)|0.657|
|random_forest|min_votes 3|0.723 (0.014)|0.672|
|oblique_random_forest|min_votes 2|0.724 (0.012)|0.672|
|adaboost|min_votes 1|0.723 (0.013)|0.721|
|gradient_boosting|min_votes 2|0.726 (0.013)|0.723|
|hist_gradient_boosting|min_votes 1|0.725 (0.013)|0.710|
|xgboost|min_votes 3|0.725 (0.013)|0.721|

With feature selection, XGBoost kept 2 of the 10 features ([numeric__age, numeric__priors_count]) and reached a tuned ROC-AUC of 0.725 ± 0.013, compared with 0.727 ± 0.016 using all features. This 0.002 difference is not that significant and the same pattern holds for the other models, whose scores changed by at most about 0.006 with selection.

## Week 9
(Option A: Adult income) This week the pipeline was adapted to run on a new dataset, the UCI Adult dataset to analyse (does a person earn more than 50,000 USD a year?).

### How to run
- python main.py                      # COMPAS, (reads config.yaml)
- python main.py config_adult.yaml    # Adult income


### What broke, and how it was fixed
| What broke | Why | Fix |
|---|---|---|
|main.py could only run COMPAS| load_config() always read config.yaml |main.py takes a config path as an argument (sys.argv), defaulting to config.yaml|
|? stayed in the data as a category |	placeholder tokens were only turned into NaN inside _canonicalize_categories(), which only touches the columns listed in canonical_categories |	clean_dataset() now turns placeholder tokens into NaN in every text column|
|KeyError: 'score_text' in the fairness report|	COMPAS's own score column was hardcoded in split_features_target() and fairness_report() | The report compares against it only when it is set (COMPAS), and the wording is now generic|
| COMPAS validity rules did not fit new dataset | rules are not the same for every dataset |Adult's own rules in config_adult.yaml|
|No way to predict new data| the pipeline only evaluated and refit |new src/predict.py: same cleaning and feature split as training, no drop_duplicate_rows|
|Scoreboard would mix datasets|	both datasets would write to results/runs.csv|	separate output folder in each config|

### Scoreboard
| Model | feature_selection | roc_auc (std) | untuned roc_auc | F1 | Accuracy |
|---|---|---|---|---|---|
| dummy |  | 0.5 | 0.5 | 0 | 0.76 |
| xgboost | min_votes 2 | 0.909 (0.014)| 0.913 | 0.675 | 0.860 |

## Week 10
This week the imputer became a setting in config.yaml: median, mean, knn, iterative or random_forest. They were added in a new _NUMERIC_IMPUTERS dictionary in src/preprocessing.py, and build_preprocessor() now scales the numeric columns first and then imputes, because KNN needs all columns on the same scale.

Main.py now saves the final fitted pipeline to models/model.joblib (output.model_path). 
And pred.py loads it, cleans the new file with the training rules, without drop_duplicate_rows(), so every row gets a prediction and writes id, probability and prediction (threshold 0.5):

### Scoreboard (XGBOOST untuned)
| imputer | roc_auc (std) | 
|---|---|
| median | 0.721 (0.013) |
| mean | 0.720 (0.014) |
| knn | 0.720 (0.011) |
| iterative | 0.721 (0.014) |
| random_forest | 0.721 (0.013) |

### prediction results

|id|probability|prediction|
|---|---|---|
|900001|0.5152289|1|
|900002|0.2895547|0|
|900003|0.3504241|0|
|900004|0.5089558|1|
|900005|0.3771013|0|
|900006|0.61310786|1|
|900007|0.43267715|0|
|900008|0.42228296|0|
|900009|0.19221552|0|
|900010|0.34199584|0|
|900011|0.1954701|0|
|900012|0.88729805|1|
|900013|0.47601312|0|
|900014|0.7074971|1|
|900015|0.4801668|0|
|900016|0.4801668|0|
|900017|0.72893476|1|
|900018|0.2844232|0|
|900019|0.34277713|0|
|900020|0.44525927|0|

