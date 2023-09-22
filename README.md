# Credit-card-faurd-detection-using-ML
Credit Card Fraud Detection Project
Hello!

I'm excited about machine learning and have decided to take on this project to become more comfortable with the models used. This project focuses on credit card fraud detection, where we aim to predict whether a transaction is fraudulent or not based on a dataset of transactions.

Getting Started
Prerequisites
Make sure you have the following libraries installed:

NumPy
Pandas
Matplotlib
Seaborn
Scipy
Scikit-learn
You can install them using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scipy scikit-learn
Save to grepper
Data Importing
We begin by importing the dataset:

python
Copy code
import numpy as np
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv')
Save to grepper
Exploring the Dataset
Let's explore the dataset to understand its structure:

python
Copy code
print(data.columns)
print(data.shape)
data.hist(figsize=(20, 20))
Save to grepper
Data Analysis
We analyze the data to understand the distribution of fraudulent and valid transactions:

python
Copy code
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(valid)))
Save to grepper
Correlation Analysis
Visualizing the correlation matrix to identify relationships between features:

python
Copy code
corrmat = data.corr()
sns.heatmap(corrmat, vmax=0.8, square=True)
Save to grepper
Organizing the Data
Preparing the data for model training:

python
Copy code
columns = data.columns.tolist()
columns = [c for c in columns if c not in ['Class']]
target = 'Class'
X = data[columns]
Y = data[target]
Save to grepper
Applying Machine Learning Algorithms
We use various algorithms for fraud detection:

Isolation Forest and Local Outlier Factor
python
Copy code
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Define a random state
state = 1

classifiers = {
    'Isolation Forest': IsolationForest(max_samples=len(X), contamination=outlier_fraction, random_state=state),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
}

# Fit the model and evaluate
for clf_name, clf in classifiers.items():
    # ... (code for fitting and evaluating the model)
Save to grepper
Naive Bayes
python
Copy code
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Select columns and split the data
# ... (code for data preparation)

# Instantiate and train the Naive Bayes classifier
# ... (code for fitting and evaluating the model)
Save to grepper
DBSCAN
python
Copy code
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Select columns and preprocess data
# ... (code for data preparation)

# Apply DBSCAN clustering and evaluate performance
# ... (code for fitting and evaluating the model)
Save to grepper
Decision Tree
python
Copy code
from sklearn.tree import DecisionTreeClassifier

# Select columns and split the data
# ... (code for data preparation)

# Create and train the Decision Tree classifier
# ... (code for fitting and evaluating the model)
Save to grepper
Conclusion
This project explores various machine learning techniques for credit card fraud detection. Each algorithm has its strengths and weaknesses, and the choice of the algorithm depends on specific use cases and requirements.

Feel free to experiment with these models and improve the fraud detection rates!
