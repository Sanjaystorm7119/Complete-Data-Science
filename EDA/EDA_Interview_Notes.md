# EDA & Feature Engineering - Interview Notes 📝

## 1. What is EDA (Exploratory Data Analysis)?

**Definition:** EDA is the process of analyzing datasets to summarize their main characteristics, often using visual methods. It helps understand data patterns, spot anomalies, test hypotheses, and check assumptions.

**Interview Answer:**

> "EDA is the first step in any data science project where we explore the dataset to understand its structure, identify patterns, detect outliers, and discover relationships between variables before building any models."

---

## 2. Essential EDA Steps & Code 🔧

### Step 1: Data Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV with different separators
df = pd.read_csv('file.csv')              # comma-separated
df = pd.read_csv('file.csv', sep=';')     # semicolon-separated

# Excel files (requires openpyxl)
df = pd.read_excel('file.xlsx')
```

**Interview Tip:** Always check the delimiter when CSV isn't loading correctly!

---

### Step 2: Initial Data Exploration

```python
df.head()          # First 5 rows
df.tail()          # Last 5 rows
df.shape           # (rows, columns)
df.info()          # Data types & non-null counts
df.describe()      # Statistical summary
df.columns         # Column names
df.dtypes          # Data types of each column
```

**Interview Question:** _"How would you quickly understand a new dataset?"_

> "I use `df.info()` for data types and missing values, `df.describe()` for statistical summary, and `df.head()` to see sample data."

---

### Step 3: Missing Values Analysis

```python
# Check missing values
df.isnull().sum()                    # Count per column
df.isnull().sum() / len(df) * 100    # Percentage

# Handle missing values
df.dropna()                          # Remove rows with NaN
df.dropna(subset=['column'])         # Drop if specific column is NaN
df.fillna(value)                     # Fill with specific value
df.fillna(df.mean())                 # Fill with mean
df.fillna(df.median())               # Fill with median
df.fillna(df.mode()[0])              # Fill with mode
```

**Interview Question:** _"How do you handle missing values?"_

> "It depends on the data:
>
> - **Numerical:** Use mean/median (median for skewed data)
> - **Categorical:** Use mode or create 'Unknown' category
> - **Too many missing (>50%):** Consider dropping the column
> - **Random missing:** Use imputation techniques like KNN"

---

### Step 4: Duplicate Records

```python
# Find duplicates
df.duplicated().sum()         # Count duplicates
df[df.duplicated()]           # View duplicate rows

# Remove duplicates
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['col1', 'col2'])  # Based on specific columns
```

---

### Step 5: Data Type Conversion

```python
# String to numeric
df['column'] = df['column'].astype(int)
df['column'] = df['column'].astype(float)

# To datetime
df['date'] = pd.to_datetime(df['date'])

# Check if values are numeric
df['column'].str.isnumeric()
```

**Common Pitfall:** Always check for non-numeric values before conversion!

```python
df[~df['column'].str.isnumeric()]  # Find non-numeric values
```

---

## 3. Feature Engineering Techniques 🛠️

### Date Feature Extraction

```python
# From string date (format: DD/MM/YYYY)
df['Date'] = df['Date_of_Journey'].str.split('/').str[0]
df['Month'] = df['Date_of_Journey'].str.split('/').str[1]
df['Year'] = df['Date_of_Journey'].str.split('/').str[2]

# Using datetime
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
```

### Text/String Feature Extraction

```python
# Extract from time strings (e.g., "22:20")
df['Hour'] = df['Time'].str.split(':').str[0]
df['Minute'] = df['Time'].str.split(':').str[1]

# Clean text by removing extra parts
df['column'] = df['column'].apply(lambda x: x.split(' ')[0])
```

### Size Conversion (e.g., "19M" → numeric)

```python
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan

df['Size_MB'] = df['Size'].apply(convert_size)
```

---

## 4. Correlation Analysis 📊

```python
# Correlation matrix
df.corr()

# Heatmap visualization
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

**Interview Question:** _"What does correlation tell you?"_

> "Correlation measures the linear relationship between variables:
>
> - **+1:** Perfect positive correlation
> - **-1:** Perfect negative correlation
> - **0:** No linear correlation
> - **> 0.7 or < -0.7:** Strong correlation
> - Focus on features highly correlated with the target variable"

---

## 5. Data Visualization Techniques 📈

### Univariate Analysis

```python
# Distribution of numerical variables
df['column'].hist()
df['column'].plot(kind='box')
sns.distplot(df['column'])

# Count of categorical variables
df['category'].value_counts().plot(kind='bar')
sns.countplot(x='category', data=df)
```

### Bivariate Analysis

```python
# Scatter plot
plt.scatter(df['x'], df['y'])

# Box plot by category
sns.boxplot(x='category', y='value', data=df)

# Pair plot
sns.pairplot(df)
```

---

## 6. Handling Categorical Variables 🏷️

```python
# Check unique values
df['column'].unique()
df['column'].nunique()
df['column'].value_counts()

# One-hot encoding
pd.get_dummies(df, columns=['column'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['column_encoded'] = le.fit_transform(df['column'])
```

---

## 7. Outlier Detection & Treatment 🎯

```python
# Using IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]

# Remove outliers
df = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Visualization
sns.boxplot(df['column'])
```

**Interview Question:** _"How do you handle outliers?"_

> "First, I investigate if they're data errors or genuine extreme values:
>
> - **Errors:** Remove or correct
> - **Genuine:** Cap/floor (Winsorization), transform (log), or keep if domain-relevant"

---

## 8. Interview Questions & Answers ❓

### Q1: What's the difference between `df.info()` and `df.describe()`?

> - `info()`: Shows data types, non-null counts, memory usage
> - `describe()`: Shows statistical summary (mean, std, min, max, quartiles)

### Q2: How do you handle imbalanced classes?

> - Oversampling (SMOTE)
> - Undersampling
> - Class weights
> - Stratified sampling

### Q3: Why use median over mean for imputation?

> Median is robust to outliers. For skewed distributions, median better represents the central tendency.

### Q4: What's the purpose of feature engineering?

> Create new features from existing ones to improve model performance by capturing patterns that raw data doesn't explicitly show.

### Q5: How do you decide which features to drop?

> - Constant or near-constant values
> - High percentage of missing values (>50-70%)
> - High correlation with other features (multicollinearity)
> - Irrelevant to the target variable

---

## 9. Quick Checklist ✅

- [ ] Load data correctly (check delimiter)
- [ ] Check shape and data types
- [ ] Identify missing values
- [ ] Handle/remove duplicates
- [ ] Convert data types appropriately
- [ ] Extract features from dates/text
- [ ] Analyze correlations
- [ ] Detect and handle outliers
- [ ] Visualize distributions
- [ ] Encode categorical variables

---

## 10. Common Libraries 📚

```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns        # Statistical visualization
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline           # For Jupyter notebooks
```

---

_Based on: Wine Quality EDA, Flight Price Prediction, Google Playstore Analysis_
