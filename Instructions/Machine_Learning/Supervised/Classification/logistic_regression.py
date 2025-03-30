import matplotlib.pyplot as plt
import numpy as np




logistic_regression_explanation = r"""


Logistic regression is a statistical method used for binary classification problems. It predicts the probability of a binary outcome (1/0, Yes/No, True/False) based on one or more predictor variables (features). 

## Key Concepts

### 1. **The Logistic Function**
Logistic regression uses the logistic function (also known as the sigmoid function) to model the relationship between the input features and the output probabilities. The logistic function is defined as:



$$
f(z) = \frac{1}{1 + e^{-z}}
$$

where $$ z $$ is a linear combination of the input features, given by:

$$
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n
$$

Here:
- $$( \beta_0 )$$ is the intercept.
- $$( \beta_1, \beta_2, \ldots, \beta_n )$$ are the coefficients for the features $$ x_1, x_2, \ldots, x_n $$.

### 2. **Interpretation of Output**
The output of the logistic function is a probability value between 0 and 1, which can be interpreted as follows:
- If $$ P(y=1|X) > 0.5 $$, predict class 1.
- If $$ P(y=1|X) \leq 0.5 $$, predict class 0.

### 3. **Cost Function**
The cost function for logistic regression is derived from the likelihood function. The goal is to minimize the cost function, which for logistic regression is the **log loss**:

$$
J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h(x^{(i)})) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) \right]
$$

where:
- $$ m $$ is the number of training examples.
- $$ h(x^{(i)}) $$ is the predicted probability for the $$ i $$-th example.

### 4. **Training the Model**
The model is trained using optimization algorithms like **Gradient Descent** or **Newton-Raphson** to find the best coefficients $$ \beta $$ that minimize the cost function.

### 5. **Assumptions**
Logistic regression makes several assumptions:
- The relationship between the features and the log odds of the outcome is linear.
- The outcome variable is binary.
- The observations are independent of each other.

### 6. **Advantages and Disadvantages**

**Advantages:**
- Simple and interpretable.
- Efficient for binary classification.
- Provides probabilities for outcomes.

**Disadvantages:**
- Assumes linearity in the log odds.
- Sensitive to outliers.
- Can underperform with highly imbalanced datasets.

## Implementation in Python

Here’s a simple implementation of logistic regression using the `scikit-learn` library:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 9, 8, 7, 6],
    'label': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Split the dataset
X = df[['feature1', 'feature2']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

"""
