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

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ (Overall correctness)
- **Precision**: $\frac{TP}{TP + FP}$ (Quality of positive predictions) — _Focus here if FP is costly._
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ (Ability to find all positives) — _Focus here if FN is costly (e.g., Cancer detection)._
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$ (Harmonic mean of Precision & Recall).
- **Specificity**: $\frac{TN}{TN + FP}$ (Ability to find all negatives).

### ROC and AUC

- **ROC (Receiver Operating Characteristic)**: A plot of True Positive Rate (Recall) vs. False Positive Rate (1 - Specificity) at various thresholds.
- **AUC (Area Under Curve)**: Represents the degree or measure of separability. Higher AUC means the model is better at predicting 0s as 0s and 1s as 1s.

---

## 4. Hyperparameter Tuning

- **C**: Inverse of regularization strength (smaller values specify stronger regularization).
- **Penalty**:
  - `l1` (Lasso): Can lead to sparse coefficients (feature selection).
  - `l2` (Ridge): Standard regularization.
  - `elasticnet`: Combination of L1 and L2.
- **Solver**: Algorithms to use for optimization (`liblinear`, `lbfgs`, `newton-cg`, `sag`, `saga`).
