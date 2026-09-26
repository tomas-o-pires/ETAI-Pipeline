20231652 Tomás Pires

week 2
DT: Train accuracy: 0.829
Test accuracy:  0.629
Gap (train - test): +0.199

LR: Train accuracy: 0.680
Test accuracy:  0.677
Gap (train - test): +0.003

Conclusion: The Logistic Regression demonstrates better generalization than the Decision Tree. Although the Decision Tree achieves higher training accuracy, it has lower test accuracy giving a large train-test gap indicating overfitting. While the Logistic Regression achieves a lower training accuracy, but with a train-test gap of only 0.003 points.

week 3

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

