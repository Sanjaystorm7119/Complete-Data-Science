# Ridge, Lasso, and Elastic Net Regression (Regularization)

Regularization is a technique used to reduce the complexity of a machine learning model to prevent **overfitting** (high variance). It adds a penalty term to the cost function.

---

## 1. Ridge Regression (L2 Regularization)

Ridge regression adds a penalty equivalent to the **square of the magnitude** of coefficients.

### Mathematical Intuition

**Cost Function:**
$$Cost = \sum(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

- where $\lambda$ is the regularization parameter (shrinkage factor).

### Key Characteristics

- **Shrinkage:** It shrinks the coefficients towards zero but **never exactly to zero**.
- **Multicollinearity:** It is particularly useful when features are highly correlated; it distributes the weight among them.
- **Handling Overfitting:** It reduces model complexity while keeping all features.

---

## 2. Lasso Regression (L1 Regularization)

Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty equivalent to the **absolute value** of the magnitude of coefficients.

### Mathematical Intuition

**Cost Function:**
$$Cost = \sum(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

### Key Characteristics

- **Feature Selection:** Unlike Ridge, Lasso can shrink some coefficients **exactly to zero**, effectively acting as an automated feature selection tool.
- **Sparsity:** It produces sparse models (models with fewer features).
- **Limitation:** If there are a group of highly correlated variables, Lasso tends to pick one arbitrarily and ignore the others.

---

## 3. Elastic Net Regression

Elastic Net is a hybrid approach that combines both **L1 (Lasso)** and **L2 (Ridge)** penalties.

### Mathematical Intuition

**Cost Function:**
$$Cost = \sum(y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$$

### Why Use Elastic Net?

- It overcomes the limitations of Lasso when dealing with highly correlated features.
- It maintains the feature selection property of Lasso while adding the stability of Ridge.

---

## 4. Lambda ($\lambda$) / Alpha ($\alpha$) Parameter

| $\lambda$ Value    | Effect                                              |
| :----------------- | :-------------------------------------------------- |
| **$\lambda = 0$**  | Equivalent to OLS (Ordinary Least Squares).         |
| **High $\lambda$** | Higher penalty, leads to underfitting (High Bias).  |
| **Low $\lambda$**  | Lower penalty, might still overfit (High Variance). |

---

## 5. Comparison Table

| Feature                   | Ridge (L2)             | Lasso (L1)                   | Elastic Net          |
| :------------------------ | :--------------------- | :--------------------------- | :------------------- |
| **Penalty Term**          | $\lambda \sum \beta^2$ | $\lambda \sum \mid\beta\mid$ | Both L1 & L2         |
| **Coefficient Shrinkage** | Shrinks towards 0      | Shrinks to exactly 0         | Shrinks towards/to 0 |
| **Feature Selection**     | No                     | Yes                          | Yes (selective)      |
| **Use Case**              | Correlated features    | Many useless features        | Correlated + Useless |

---

## 6. Interview Specific Q&A

### Q1: Why do we use Regularization?

**Ans:** To prevent overfitting. Overfitting occurs when a model performs exceptionally well on training data but poorly on unseen data (High Variance). Regularization penalizes high coefficients, simplifying the model.

### Q2: Which one to choose: Ridge or Lasso?

**Ans:**

- Use **Ridge** if you think all features are important and you want to handle multicollinearity.
- Use **Lasso** if you suspect only a few features are actually contributing to the target (Feature Selection).
- Use **Elastic Net** if you have many features and many of them are correlated.

### Q3: What happens if $\lambda$ is too high?

**Ans:** The model becomes too simple. It leads to **High Bias** and **Underfitting**, as the penalty term dominates the loss function, forcing coefficients to be very small.

### Q4: Can Ridge Regression perform feature selection?

**Ans:** No. Ridge shrinks coefficients but they theoretically never reach zero. Hence, all features remain in the model.

### Q5: What is the geometric interpretation of Lasso vs Ridge?

**Ans:**

- **Ridge:** The constraint region is a **circle** (hypersphere). The OLS solution usually hits the circle at a point where coefficients are non-zero.
- **Lasso:** The constraint region is a **diamond** (rhombus). The OLS solution often hits the corners of the diamond, where one or more coefficients are exactly zero.

---

## 7. Python Implementation (Sklearn)

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

> **Note:** In Scikit-Learn, the hyperparameter $\lambda$ is referred to as **alpha**.
