# Logistic Regression & Performance Metrics

Logistic Regression is a supervised learning algorithm used for **classification** tasks (despite its name). It predicts the probability of an instance belonging to a particular class using the logistic function.

## 1. Core Foundations

### Sigmoid Function (Logistic Function)

The core of logistic regression is the sigmoid function, which maps any real-valued number into a value between 0 and 1.
$$P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n$.

### Decision Boundary

- If $P(y=1|X) \ge 0.5$, predict class 1.
- If $P(y=1|X) < 0.5$, predict class 0.

### Cost Function: Log Loss (Binary Cross-Entropy)

Linear regression uses MSE, but Logistic Regression uses Log Loss because MSE results in a non-convex function for logistic regression.
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]$$

---

## 2. Assumptions

1. **Binary/Ordinal Output**: The dependent variable should be binary (0/1) or ordinal.
2. **Independence of Observations**: Observations should not come from repeated measurements.
3. **No Multicollinearity**: Independent variables should not be highly correlated.
4. **Linearity of Independent Variables and Log Odds**: While it doesn't require a linear relationship between X and Y, it requires a linear relationship between independent variables and the **logit** (log-odds).
5. **Large Sample Size**: Typically requires a larger sample size than linear regression.

---

## 3. Performance Metrics

### Confusion Matrix

|               | Predicted: 0        | Predicted: 1        |
| ------------- | ------------------- | ------------------- |
| **Actual: 0** | True Negative (TN)  | False Positive (FP) |
| **Actual: 1** | False Negative (FN) | True Positive (TP)  |

### Key Metrics

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
  - **Definition**: The ratio of correctly predicted observations to the total observations. It indicates overall correctness.
  - **Use Case**: Best used when the target classes are well-balanced.

- **Precision (Positive Predictive Value)**: $\frac{TP}{TP + FP}$
  - **Definition**: The ratio of correctly predicted positive observations to the total predicted positives.
  - **Use Case**: Focus here when the cost of False Positives is high (e.g., Spam detection).

- **Recall (Sensitivity / True Positive Rate)**: $\frac{TP}{TP + FN}$
  - **Definition**: The ratio of correctly predicted positive observations to all actual positives.
  - **Use Case**: Focus here when the cost of False Negatives is high (e.g., Cancer detection, Fraud detection).
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
  - **Definition**: The harmonic mean of Precision and Recall. It provides a balance between the two.
  - **Use Case**: Best used when you have an imbalanced dataset or when you want to seek a balance between Precision and Recall.

- **Specificity (True Negative Rate)**: $\frac{TN}{TN + FP}$
  - **Definition**: The ratio of correctly predicted negative observations to all actual negatives.
  - **Use Case**: Important in clinical tests where we want to ensure healthy people are not misdiagnosed.

### ROC and AUC

- **ROC Curve (Receiver Operating Characteristic)**:
  - **Definition**: A probability curve that plots the **True Positive Rate (Recall)** against the **False Positive Rate (1 - Specificity)** at various threshold settings. It shows the performance of a classification model at all classification thresholds.
- **AUC (Area Under the Curve)**:
  - **Definition**: Represents the probability that the model ranks a random positive example more highly than a random negative example.
  - **Interpretation**:
    - AUC = 1: Perfect classifier.
    - AUC = 0.5: No better than random guessing.
    - Higher AUC means the model is better at distinguishing between classes.

---

## 4. Hyperparameter Tuning

- **C**: Inverse of regularization strength (smaller values specify stronger regularization).
- **Penalty**:
  - `l1` (Lasso): Can lead to sparse coefficients (feature selection).
  - `l2` (Ridge): Standard regularization.
  - `elasticnet`: Combination of L1 and L2.
- **Solver**: Algorithms to use for optimization (`liblinear`, `lbfgs`, `newton-cg`, `sag`, `saga`).
