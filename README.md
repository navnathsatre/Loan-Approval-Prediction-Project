# Loan-Approval-Prediction-Project

![Web App GIF](https://github.com/navnathsatre/Loan-Approval-Prediction-Project/blob/3c21c80124f85b35b3726b9e4d920192aa87b5d2/Loan_Approval_GIF.gif)

# __Business Objective :__<br/>
To predict the impact of the incident raised by the customer.

# __Problem :__<br/>
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.

# __Data Set Details :__<br/>
It’s not at all easy to get a loan from the bank. Getting a loan approved requires a complex mix of factors not the least of which is a steady income! So this ML project aims to create a model that will classify how much loan the user can obtain based on various factors such as the user’s marital status, income, education, employment prospects, number of dependents, etc. The dataset attached provides details about all these factors which can then be used to create an ML model that demonstrates the amount of loan that can be approved.

# __Methodology :__<br/>
![Methodology](https://user-images.githubusercontent.com/75266852/133732015-771aabd1-647a-4e12-89ea-42f4a395b50c.png)
# __Exploratory Data Analysis (EDA) :__<br/>
## Univariate Analysis - 
__Categorical Variables__<br/>
Mostly, those who Male, Married, Graduated, Not Self_Employed they applied for Loan.<br/>
Those who a high Credit_History (1.0) they mostly applied for Loan.<br/>
From any Property_Area people applied for Loan.<br/>
![UnivariateAnalysis](https://user-images.githubusercontent.com/75266852/133725566-a40cea24-8de2-4fce-a7fd-093543f0de40.png)

__Continuous Variables__<br/>
1. Our data is Not Normal (Right Skewed) and Huge Outliers in ApplicantIncome.<br/>
2. Our data is Not Normal (Right Skewed) and There are Outliers in CoapplicantIncome.<br/>
3. Our data is Not Normal (Right Skewed) and Huge Outliers in LoanAmount.<br/>
4. Our data is Not Normal (Right Skewed) and Huge Outliers in LoanAmount.
Below figure is the one example of numeric column (LoanAmount)
![UnivariateNumericCol](https://user-images.githubusercontent.com/75266852/133725421-39b68ddc-69d3-46ab-b709-7e475c293159.png)

## Bi-variate Analysis -
* From HeatMap - The most correlated variables are (ApplicantIncome - LoanAmount) and LoanAmount is also correlated with CoapplicantIncome.
* From Two-way table - We can see those who NOT Self_Employed and NOT dependent mostly they apply for loan and most male Graduated apply for loan.
* From below figure - The People who have good Credit History they got loan most of the Time and other variables plays less importance role.
![BivariateAnalysis](https://user-images.githubusercontent.com/75266852/133726685-1431bf75-ff25-4c50-9302-ecbad1cd722b.png)
* From Chi-Square Test Credit_History, Property_Area, Married, Education, are dependent on Loan_Status

## Missing Value Treatment - 
* We ues missing value imputation for numeric variable using backfill (use next valid observation to fill gap)
* We use missing value imputation for categorical variable using mode
![MissingValues](https://user-images.githubusercontent.com/75266852/133727947-9f24958e-950c-4685-b066-92834e6097d1.png)

### __Variable Creation :__<br/>
We Add ApplicantIncome and CoapplicantIncome to create TotalIncome.
### __Variable Transformation(Outliers Treatment) :__<br/>
We use log and cube root function for transforming numerical variable and then normalized it using MinMaxScaler.<br/>
And for chategorical variables create dummies using get_dummies.
### __Balancing the data :__<br/>
From left below figure, data is imbalanced we need to balance it.<br/>
We balanced the data set using SMOTETomek<br/>
Original data set shape Counter({Y: 422, N: 192})<br/>
Resample data set shape Counter({Y: 383, N: 383})<br/>

 ![DataBalancing](https://user-images.githubusercontent.com/75266852/133730655-4f3ace78-f765-486c-873a-861629f0f8e6.png)

# Model Building - 
__We used different different models but following are finalized__<br/>
### KNeighborsClassifier:
The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. <br/>
for more info [click here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

### Support Vector Machines:
Support Vector Machine” (SVM) is a supervised machine learning algorithm that can be used for both classification or regression challenges. However,  it is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.<br/>
for more info [click here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

### ExtraTreesClassifier:
ExtraTreesClassifier is an ensemble learning method fundamentally based on decision trees. ExtraTreesClassifier, like RandomForest, randomizes certain decisions and subsets of data to minimize over-learning from the data and overfitting.<br/>
for more info [click here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

### XGBClassifier:
It is a gradient boosting algorithm which forms strong rules for the model by boosting weak learners to a strong learner. It is a fast and efficient
algorithm which recently dominated machine learning because of its high performance and speed.<br/>
for more info [click here](https://xgboost.readthedocs.io/en/latest/)


**Parameter setting for machine learning models:**<br/>
| Model | Parameter Setting |
| ------- | ------- |
| KNeighborsClassifier | n_neighbors=5, |
| Support Vector Machines | C=0.7,kernel='poly',gamma=16 |
| ExtraTreesClassifier | criterion='entropy',max_features='sqrt',n_estimators=90,max_depth=8 |
| XGBClassifier | learning_rate =0.005, n_estimators=96,eval_metric='mlogloss',max_depth=8 |


# Deployment :
Finally I used Streamlit App framework to deploy my application.

## Challenges faced ?
* Balancing the Imbalance Data
* Missing Values
* To increase the Model Accuracy

## How did you overcome ?
* By using SMOTETomek
* Impute Cat data using Mode and Num data using Bfill Method
* We tuned hyperparameters  

# Conclusion -
- We did Univariate Analysis and bivariate analysis to see imapct of one another on their features using charts.
- We did Exploratory data Analysis on the features of this dataset and saw how each feature is distributed.
- We analysed each variable to check if data is cleaned and normally distributed.
- We cleaned the data and filled NA values with mode and bfill.
- We calculated correaltion between independent variables and found that applicant income and loan amount have significant relation.
- We created dummy variables for constructing the model.
- We constructed models taking different variables into account and credit history is creating the most impact on loan giving decision.
- We treat outliers as variable transformation and normalized it.
- For data balanced we used SMOTETomek.
- Finally, from model building using Encemble Model we finalized KNN, SVM, Extra Tree Classifier, XGBM and we got training accuracy of 85.06% with validation score 81.85%

*You can reach on my web application*
[click here](https://share.streamlit.io/navnathsatre/loan-approval-prediction-project/main/LoanApprovalPredictionDeployment.py)
