Important notice: the 'unseen-data' file made available by the professor has formatting errors. My CSV files are already fixed.

# Task 1
## Make a preliminary statistical analysis of the credit dataset

The following tables describe the statistical analysis for both classes of features, namely numerical features and 
categorical features, that were made available in the original (raw) dataset before any data wrangling done.

**Statistical summary of the numerical features**:

|  | loan_amnt |  funded_amnt |  funded_amnt_inv |    int_rate |  installment |
| --- | --- | --- | --- | --- | --- |
| count | 331304.0000 |  331304.0000 |      331304.0000 | 331304.0000 |  331304.0000 |   
| mean |   15485.6705 |   15485.6705 |       15479.0312 |     12.5709 |     452.0909 |   
| std |     9182.7264 |    9182.7264 |        9181.2607 |      4.6853 |     264.2243 |   
| min |     1000.0000 |    1000.0000 |         725.0000 |      5.3200 |      14.7700 |   
| 25% |     8100.0000 |    8100.0000 |        8100.0000 |      8.8100 |     259.2400 |   
| 50% |    14000.0000 |   14000.0000 |       14000.0000 |     11.9900 |     387.1500 |   
| 75% |    20700.0000 |   20700.0000 |       20675.0000 |     15.3100 |     602.3000 |   
| max |    40000.0000 |   40000.0000 |       40000.0000 |     30.9900 |    1618.2400 |   

|  | annual_inc |         dti |  delinq_2yrs |  inq_last_6mths |    open_acc |
| --- | --- | --- | --- | --- | --- |
| count |  331304.0000 | 331138.0000 |  331304.0000 |     331303.0000 | 331304.0000 |   
| mean |   80425.2580 |     19.0286 |       0.3369 |          0.6083 |     11.8598 |   
| std |     91150.7789 |     12.5358 |       0.9184 |          0.8880 |      5.7909 |   
| min|         0.0000 |     -1.0000 |       0.0000 |          0.0000 |      0.0000 |   
| 25% |     48000.0000 |     12.0500 |       0.0000 |          0.0000 |      8.0000 |   
| 50% |     67314.0000 |     18.1900 |       0.0000 |          0.0000 |     11.0000 |   
| 75% |     95000.0000 |     25.0900 |       0.0000 |          1.0000 |     15.0000 |   
| max |   9757200.0000 |    999.0000 |      21.0000 |          5.0000 |     81.0000 |   

|   |       pub_rec |    revol_bal |  revol_util |   total_acc |   out_prncp | 
| --- | --- | --- | --- | --- | --- |
| count | 331304.0000 |  331304.0000 | 331074.0000 | 331304.0000 | 331304.0000 |  
| mean    |   0.2468 |  16049.1716   |  48.5172   |  24.8545 |   678.8532   |
| std     |   0.6682 |  23183.3434   |  24.8231   |  12.3119  | 3493.3046   |
| min     |   0.0000 |      0.0000   |   0.0000   |   2.0000   |   0.0000   |
| 25%     |   0.0000 |   5507.0000   |  29.6000   |  16.0000   |   0.0000   |
| 50%     |   0.0000 |  10583.0000   |  48.0000   |  23.0000   |   0.0000   |
| 75%     |   0.0000 |  19084.0000   |  67.3000   |  32.0000   |   0.0000   |
| max     |  86.0000 | 1044210.0000  |  182.8000  |  176.0000 | 40000.0000   |

|    |   total_pymnt |      target |
| --- | --- | --- |
| count | 331304.0000 | 310679.0000  |
| mean  |  13235.1261 |     0.3337  |
| std   |  10045.6296 |     0.4715  |
| min   |      0.0000 |     0.0000  |
| 25%   |   5593.8933 |     0.0000  |
| 50%   |  10584.4760 |     0.0000  |
| 75%   |  18391.8971 |     1.0000  |
| max   |  59808.2621 |     1.0000  |

**Statistical summary of the numerical features**:

|   |             term |   grade | emp_title | emp_length | home_ownership |
| --- | --- | --- | --- | --- | --- |
| count  |    331304 | 331304 |   300514  |   307479   |      331304  |  
| unique |         2  |     7  |   90441  |       11    |          4  |  
| mode   |  36 months |      B  | Teacher | 10+ years   |    MORTGAGE  |  
| freq   |    230667  | 98290    |  6412  |   109514    |     164411  |  

|   |        verification_status | issue_d   |          purpose | addr_state |
| --- | --- | --- | --- | --- |
| count  |             331304 | 331304      |        331304   |  331304   |
| unique |                  3  |    11      |            13   |      50   |
| mode    |    Source Verified | Mar-16 | debt_consolidation  |       CA   |
| freq    |            133327 |  56649   |           189096  |    46585   |

|   |        earliest_cr_line | loan_status |  
| --- | --- | --- |
| count  |          331304   |   310704  |  
| unique |             685   |        6  |  
| mode  |            Sep-04  | Fully Paid |  
| freq |              2603   |   207036 |

# Preliminary work
## Before training models
Some adjustments to the dataset must be made before this data can be used to train the models. Verification and adjustment of missing values, correct assignment of data type for some features that are incorrect (dates and integers), correction of some target observations and creation of new features (feature engineering) are done beforehand.

Once the dataset is wrangled and cleaned, the categorical variables undergo One Hot Encoding process using the scikit-learn library, in order to transform the observations in these categories into dummy variables so that they can be used correctly for fitting and predictions by the model algorithms.

The data is then segregated into three distinct groups: validation data (10% of the total), training data (72% of the total), and test data (18% of the total). The training data is used for the effective training of the models, and the test data is used to estimate the predictive power of each model and make the necessary adjustments to the algorithms' hyperparameters. The validation data, in turn, serves to compare the results obtained with predictions simulating real-world data, once the hyperparameters are aligned, in order to avoid overfitting resulting from the over-tuning of the hyperparameters.

# Task 2
## Develop a logistic regression model

The logistic regression model uses the LogisticRegression algorithm from the scikit-learn library, obtaining Accuracy metric results of 0.7831 (test data) and 0.7833 (validation data). The confusion matrices and the resulting metrics for each dataset are shown below.

**Confusion Matrix for test data**
![Confusion Matrix for test data](images/lr_test.png)

Model analytics for test data:__
Accuracy: 0.783182__
Sensitivity/Recall: 0.530680__
Specificity: 0.909449__
Precision: 0.745592__

**Confusion Matrix for validation data**
![Confusion Matrix for validation data](images/lr_valid.png)

Model analytics for validation data:__
Accuracy: 0.783335__
Sensitivity/Recall: 0.539480__
Specificity: 0.907787__
Precision: 0.749107__