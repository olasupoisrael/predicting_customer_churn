# Predicting Customer Churn With Synthetic Data

## Imagine you are a data analyst at a telecom company. The marketing department has noticed an increase in customer churn and needs your help to identify which customers are most likely to leave next month.


---

### Dataset Description

The dataset contains the following columns:

- **CustomerID**: Unique identifier for each customer
- **Age**: The age of the customer
- **MonthlyCharge**: The monthly bill amount for the customer
- **CustomerServiceCalls**: Number of customer service calls
- **Churn**: Target variable, indicating if the customer churned (`Yes`/`No`)

---

### Step-by-Step Instructions

#### 1. Setup the Environment

```python
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

warnings.filterwarnings('ignore')

```

#### 2. Create the Synthetic Dataset

```python
data = {
      'CustomerID': range(1, 101),  # Unique ID for each customer
      'Age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]*10,  # Age of customers
      'MonthlyCharge': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]*10,  # Monthly bill amount
      'CustomerServiceCalls': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0]*10,  # Number of customer service calls
      'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']*10  # Churn status
}

df = pd.DataFrame(data)
```

#### 3. Data Preparation

```python
# Define features and target
X = df[['Age', 'MonthlyCharge', 'CustomerServiceCalls']]
y = df['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### 4. Build the Decision Tree Model

```python
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
```

#### 5. Evaluate the Model

```python
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

#### 6. Visualize the Decision Tree

```python
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=['Age', 'MonthlyCharge', 'CustomerServiceCalls'], class_names=['No Churn', 'Churn'])
plt.title('Decision Tree for Predicting Customer Churn')
plt.show()
```
![image](https://github.com/user-attachments/assets/ea016d69-be63-482d-9ed0-513cf246129e)

---

### Results and Interpretation
- **Gini**
The Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
The Gini impurity ranges from 0 to 0.5, where 0 indicates that all elements in the subset belong to the same class (perfect purity), and 0.5 means the data is randomly distributed across various classes.
In decision trees, a lower Gini impurity is generally preferred as it indicates greater purity of the node.
- **Samples**
This value represents the number of samples (or records) that reach the node.
It gives an idea of how much of the training data is affected by the conditions leading to this node.
A high number of samples in a node means that the condition or rule associated with that node is relevant for a significant portion of the dataset.
- **Value**
This shows the distribution of the samples in different classes at that particular node.
For a binary classification problem (like churn prediction with 'Yes' or 'No'), the value is presented as a list of two numbers. The first number indicates the count of samples in the first class, and the second number indicates the count of samples in the second class.
This distribution helps in understanding which class is predominant at a particular node.
- **Class**
This indicates the class that would be predicted if the decision tree traversal ends at that node.
It is determined based on the majority class of the samples that reach the node. For instance, if most samples at a node belong to the 'No Churn' class, the node will predict 'No Churn'.
Feature Name (e.g., 'Monthly Charge')
This is not a standard part of the decision tree node description, but it may appear in the tree's branches.
It represents the feature (or attribute) used to split the data at that node.
For example, if you see "MonthlyCharge <= 80", it means that the tree is splitting the data at this node based on whether the monthly charge is less than or equal to 80.

---
