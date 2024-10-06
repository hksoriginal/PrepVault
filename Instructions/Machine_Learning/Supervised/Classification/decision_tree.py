decision_tree_explanation = r"""


## Overview
A Decision Tree is a flowchart-like structure used for both classification and regression tasks in machine learning. It splits the dataset into subsets based on the value of input features, leading to a decision at each leaf node.

## Structure
- **Root Node**: Represents the entire dataset, which is split into two or more homogeneous sets.
- **Decision Nodes**: Nodes that represent the feature on which the data is split.
- **Leaf Nodes**: Terminal nodes that represent the output or class label.

## How Decision Trees Work
1. **Select Feature**: Choose a feature to split the data based on a specific criterion (e.g., Gini impurity, information gain).
2. **Split Dataset**: Divide the dataset into subsets based on the selected feature.
3. **Repeat**: Recursively repeat the process for each subset until:
   - All data points in a subset belong to the same class.
   - There are no features left to split.
   - A predefined depth of the tree is reached.

## Criteria for Splitting
- **Gini Impurity**: Measures the impurity of a node, with lower values indicating better purity.
- **Entropy**: Measures the disorder or unpredictability in the data.
- **Information Gain**: The reduction in entropy after the dataset is split.

### Gini Impurity Formula
$$ [ Gini(D) = 1 - \sum_{i=1}^{C} (p_i)^2 ] $$
Where $$( p_i )$$ is the probability of class $$( i )$$ in dataset $$( D )$$ and $$( C )$$ is the number of classes.

### Entropy Formula
$$[ Entropy(D) = -\sum_{i=1}^{C} p_i \log_2(p_i) ] $$

## Advantages of Decision Trees
- **Easy to Interpret**: Visual and intuitive representation.
- **No Need for Feature Scaling**: Works well with both numerical and categorical data.
- **Handles Non-linear Relationships**: Can model complex relationships.

## Disadvantages of Decision Trees
- **Overfitting**: Tends to create overly complex trees that do not generalize well.
- **Instability**: Small changes in the data can lead to different splits and structures.
- **Biased towards Dominant Classes**: Can be biased if one class dominates the dataset.

## Pruning Techniques
To combat overfitting, decision trees can be pruned:
- **Pre-pruning**: Stops the tree from growing when certain conditions are met (e.g., maximum depth).
- **Post-pruning**: Removes nodes after the tree has been created to reduce complexity.

## Implementation Example (using Python's scikit-learn)
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
```

"""