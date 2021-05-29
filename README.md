# AI-project
What kind of problem is this
This is a binary classification problem with a highly imbalanced dataset, the resutls of different supervised learning methods are compared against eachother.

Details
This project compares 5 different supervised learning methods to try predict whether a firm will go bankrupt, the 5 different methods I used were logistic regression, KNN, decision trees, random forest and support vector machines. The dataset provided is highly imbalanced where less than 7% of the instances provided were of the bankrupt class while the other 93% were of the not bankrupt class. In order to mitigate this I performed SMOTE and random undersampling in order to increase the split between bankrupt and non bankrupt class at 25:75 respoectively. I performed all 5 methods on the dataset both before and after the SMOTE and random undersampling was conducted to compare how the methods performed with 

Dataset used
For this project I used the 5th year from the UCL machine learning repository, "Polish companies bankruptcy data"
Dataset at https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
