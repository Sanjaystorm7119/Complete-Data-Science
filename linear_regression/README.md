# Simple Linear Regression (SLR) - Interview Notes

This document provides a comprehensive summary of Simple Linear Regression, based on the practical implementation using the Height-Weight dataset. It is structured for quick review and interview preparation.

---

## 1. Introduction to Simple Linear Regression

Simple Linear Regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables:

- **Independent Variable (X):** The predictor or feature (e.g., Weight).
- **Dependent Variable (y):** The outcome or target we want to predict (e.g., Height).

### The Equation

The relationship is modeled through a linear function:
$$y = \beta_0 + \beta_1x + \epsilon$$

- **$\beta_0$ (Intercept):** The value of $y$ when $X = 0$.
- **$\beta_1$ (Slope/Coefficient):** The change in $y$ for a one-unit change in $X$.
- **$\epsilon$ (Error Term):** The difference between the actual and predicted values.

---

## 2. Practical Implementation Steps

### A. Exploratory Data Analysis (EDA)

1. **Visualization:** Use Scatter Plots to check for a linear relationship.
2. **Correlation:** Check the strength and direction of the relationship using `df.corr()`. A value close to 1 or -1 indicates a strong linear relationship.
3. **Distribution:** Use Pairplots (`seaborn.pairplot`) to see the distribution and relationships of all features.

### B. Feature Selection

- The independent feature $X$ should be a 2D array or DataFrame in Scikit-Learn: `X = df[['Weight']]`.
- The dependent feature $y$ can be a 1D array or Series: `y = df['Height']`.

### C. Data Preprocessing

1. **Train-Test Split:** Divide the data (e.g., 75% training, 25% testing) to evaluate model performance on unseen data.
2. **Standardization (Scaling):**
   - **Why?** To bring features to a similar scale, making optimization (Gradient Descent) faster and more stable.
   - **Important:** Use `fit_transform()` on the training set and only `transform()` on the test set to avoid **Data Leakage**.

### D. Model Training

Using `sklearn.linear_model.LinearRegression`:

- `.fit(X_train, y_train)`: Trains the model.
- `.coef_`: Retrieves the slope ($\beta_1$).
- `.intercept_`: Retrieves the intercept ($\beta_0$).

---

## 3. Performance Metrics

| Metric          | Formula                                | Description                                                                                 |
| :-------------- | :------------------------------------- | :------------------------------------------------------------------------------------------ | --- | ---------------------------------------- |
| **MAE**         | $\frac{1}{n} \sum                      | y_i - \hat{y}\_i                                                                            | $   | Mean Absolute Error. Robust to outliers. |
| **MSE**         | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Mean Squared Error. Penalizes larger errors.                                                |
| **RMSE**        | $\sqrt{MSE}$                           | Root Mean Squared Error. Same units as the target variable.                                 |
| **R-Squared**   | $1 - \frac{SSR}{SST}$                  | Coefficient of Determination. Represents the % of variance explained by the model (0 to 1). |
| **Adjusted R2** | $1 - \frac{(1-R^2)(n-1)}{n-k-1}$       | Penalizes the addition of non-significant predictors. Always $\le R^2$.                     |

---

## 4. Key Interview Questions (Q&A)

> [!IMPORTANT]
> **Q: Why do we use standardized data in Linear Regression?**
> **A:** It helps in faster convergence of Gradient Descent and ensures that the model doesn't give undue importance to features with larger raw values.

> [!TIP]
> **Q: What is the difference between R2 and Adjusted R2?**
> **A:** R2 always increases or stays the same when you add more features, even if they are irrelevant. Adjusted R2 increases only if the new feature improves the model more than what would be expected by chance, penalizing unnecessary complexity ($k$).

> [!WARNING]
> **Q: What are the main assumptions of Linear Regression?**
>
> 1. **Linearity:** The relationship between $X$ and $y$ is linear.
> 2. **Independence:** Observations are independent of each other.
> 3. **Homoscedasticity:** Constant variance of error terms.
> 4. **Normality:** Residuals (errors) are normally distributed.

---

## 5. Statistical Modeling with `statsmodels`

While Scikit-Learn is great for prediction, `statsmodels` provides a more detailed statistical summary:

```python
import statsmodels.api as sm
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
```

- **P-value:** Used to test the significance of features (usually $< 0.05$ is significant).
- **F-statistic:** Checks if the overall model is significant.
