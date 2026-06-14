# house-prices-ames
Exploratory Data Analysis and Machine Learning modeling on 1,460 home sales to identify key price drivers in Ames, Iowa.

# 🏠 What Makes a Home Worth More?
### A Real Estate Market Intelligence Analysis — Ames, Iowa

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)
![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF?style=flat&logo=kaggle)

---

## The Business Problem

Buying or selling a home is one of the biggest financial decisions a person makes, yet most people have little clarity on what actually drives a property's value. Is it the size? The neighborhood? The age of the house?

This project analyzes **1,460 residential property sales** in Ames, Iowa across **80 features** to answer five core questions:

| # | Business Question |
|---|---|
| 1 | What property features have the strongest impact on final sale price? |
| 2 | How much is a 1-point increase in overall quality worth in dollars? |
| 3 | Is square footage or neighborhood a better predictor of price? |
| 4 | Do newer or recently remodeled homes command a significant price premium? |
| 5 | Which "invisible" features (garage, basement, fireplaces) add the most unexpected value? |

> Dataset: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## Notebook Structure

```
A. Import Libraries
B. Load Dataset
C. Data Cleaning         
D. Feature Engineering   
E. Exploratory Data Analysis
F. Dashboard
G. Modeling              
```

---

## Data Cleaning

**Problem:** 19 out of 81 columns had missing values, ranging from 1 missing row to 99.5% missing.

**Key insight:** Most "missing" values were not data collection errors, they meant the home simply didn't have that feature.

| Category | Columns | Missing % | Action |
|---|---|---|---|
| No feature categorical | PoolQC, Alley, Fence, FireplaceQu, MiscFeature, all Garage\*, all Bsmt\*, MasVnrType | 5% - 99.5% | Filled with `'None'` |
| No feature numerical | MasVnrArea, GarageYrBlt | <1% | Filled with `0` |
| Real missing | LotFrontage | 17.7% | Median per Neighborhood (not global median, lot size is neighborhood-dependent) |
| Single row | Electrical | 0.07% | Mode fill |

**Additional fixes:**
- `MSSubClass` was stored as `int64` but is actually category codes (20, 30, 60...), converted to string
- `Id` column dropped : row index, not a feature

**Result:** 0 nulls remaining across all 80 columns.

---

## Feature Engineering

**Problem:** Raw columns contained redundant, weakly-structured, or string-encoded information that models couldn't use effectively.

**16 new features created across 4 strategies:**

### 1. Size Combinations
Combining related area columns into single, stronger signals:

| New Feature | Formula | Why |
|---|---|---|
| `TotalSF` | BsmtSF + 1stFlrSF + 2ndFlrSF | Combined home size outperformed all individual area columns (r = 0.782) |
| `TotalBath` | FullBath + BsmtFullBath + 0.5×HalfBaths | Single bathroom count metric |
| `TotalPorchSF` | Sum of all 4 porch columns | Combined outdoor space |
| `LivingAreaRatio` | GrLivArea ÷ TotalSF | How much of total area is livable above ground |

### 2. Age & Time Features
Raw years (1950, 2003...) are less meaningful than age:

| New Feature | Formula | Why |
|---|---|---|
| `HouseAge` | YrSold - YearBuilt | Direct age signal |
| `YearsSinceRemodel` | YrSold − YearRemodAdd | Recency of updates |
| `WasRemodeled` | YearRemodAdd ≠ YearBuilt | Binary yes/no flag |
| `IsNewHome` | YrSold = YearBuilt | Was it brand new when sold? |

### 3. Ordinal Encoding
Quality columns stored as strings (Poor, Fair, Average, Good, Excellent) were mapped to a 0–5 numeric scale so models understand their order. Without this, `'Ex'` and `'Po'` would appear as two random unrelated categories.

Columns encoded: `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`, `BsmtExposure`, `GarageFinish`, `PavedDrive`, `LotShape`, `LandSlope`, `Functional`, `BsmtFinType1`, `BsmtFinType2`

### 4. Boolean Presence Flags
Simple yes/no signals that are often stronger than raw area columns:
`HasPool`, `HasGarage`, `HasBasement`, `HasFireplace`, `Has2ndFloor`, `HasDeck`, `HasPorch`

Dataset shape after feature engineering: (1460, 95)

New column count: 95 (was 80 after cleaning)

---

## Exploratory Data Analysis

### Chart 1 : How are home prices distributed?

**Question:** What does the typical home sale look like in Ames?

![Chart 1](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Chart_1_Sale_Distribution.png)

**Finding:** Sale prices are right-skewed, most homes sold between **$100k–$250k**, but a small number of luxury properties pull the mean above the median. The median sale price is a more honest representation of the "typical" home than the mean.

| Stat | Value |
|---|---|
| Minimum | ~$35,000 |
| Median | ~$163,000 |
| Mean | ~$181,000 |
| Maximum | ~$755,000 |

> The gap between mean and median confirms right skew, a few luxury homes inflate the average. For buyers, the typical Ames home is well under $200k.
> 
> Apply Log transformation to make the skewed data more normal.

---

### Chart 2 : Which features drive price the most?

**Question:** Out of 80 features, which ones actually move the needle on price?

![Chart 2](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Chart_2_feature_correlation_raw_and_clean.png)

**Finding:** The top price drivers are dominated by construction quality, living space, and functional home features.

**Top correlations with SalePrice (engineered dataset):**

| Feature | Correlation | What it means |
|---|---|---|
| `OverallQual` | 0.79 | Quality of finish is the single strongest predictor |
| `TotalSF` *(engineered)* | 0.78 | Combined size beats any individual floor column |
| `GrLivArea` | 0.71 | Above-ground living area |
| `GarageCars` | 0.64 | Garage capacity |
| `TotalBath` *(engineered)* | 0.63 | Combined bathroom count |

> **Key insight:** Quality of finish matters more than size. A well-finished smaller home can outsell a larger but average-quality one.

> **Feature engineering payoff:** `TotalSF` (r = 0.782) outperformed the raw `GrLivArea` (r = 0.71), combining floor areas into one signal gave the model more information than any individual column.

---

### Chart 3 : The quality premium: how much is each tier worth?

**Question:** If a seller upgrades their home's finish quality, how much more can they ask for?

![Chart 3](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Chart_3_quality_worth.png)

**Finding:** The price premium accelerates at the top end, luxury finishes yield disproportionately higher returns.

| Quality Tier | Approximate Median Price | Tier-to-Tier Jump |
|---|---|---|
| 4 (Below Average) | ~$100,000 |  |
| 5 (Average) | ~$130,000 | +$30,000 |
| 6 (Above Average) | ~$150,000 | +$20,000 |
| 7 (Good) | ~$200,000 | +$50,000 |
| 8 (Very Good) | ~$250,000 | +$50,000 |
| 9–10 (Excellent) | $350,000+ | +$100,000+ |

> **For sellers:** targeted quality upgrades: kitchen remodels, flooring, and fixtures before listing can yield significant return on investment, especially when moving from tier 5 to 6 or tier 7 to 8.

---

### Chart 4 : Neighborhood price comparison

**Question:** Which neighborhoods offer the best value in Ames?

![Chart 4](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Chart_4_neighborhood.png)

**Finding:** There is a dramatic price gap across Ames neighborhoods, the most expensive area commands nearly **4× the median price** of the most affordable one.

| Position | Neighborhood | Median Price |
|---|---|---|
| Highest | NorthRidge Heights / Stone Brook | $300,000+ |
| Near median | Gilbert, Northwest Ames, Somerset | ~$175,000 |
| Most affordable | Meadow Village, Iowa DOT & RR, Briardale | ~$85,000 |

> **For buyers:** neighborhoods sitting just below the overall median line offer similar amenities at lower price points so it's a potential undervalued zones.
>
> **For agents:** this chart serves as a quick-reference benchmarking tool for setting listing expectations.

---

### Chart 5 : Size vs. price: does more space always pay off?

**Question:** Is the relationship between size and price linear, and where do outliers sit?

![Chart 5](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Chart_5_size_vs_price.png)

**Finding:** Size and price are positively correlated, but **quality amplifies the relationship significantly.**

- A 2,000 sqft high-quality home regularly outsells a 2,500 sqft average-quality home
- The relationship is roughly linear up to ~3,500 sqft, after which returns diminish
- Two clear outliers exist: very large homes (4,000+ sqft) that sold at unusually low prices, likely estate sales or foreclosures that would distort model predictions

> **Takeaway:** Size matters, but quality is the multiplier. Buyers maximizing value should prioritize finish quality over raw square footage.

---

### Chart 6 : Age and renovation: do newer homes command a premium?

**Question:** Is it worth paying more for a newer home, and does renovation close the gap?

![Chart 6](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Chart_6_homeage_renovation.png)

**Finding:** Homes built after 1990 command a clear price premium. However, renovation is equally powerful.

- Post-1990 homes trade at a significant premium over pre-1960 stock
- **Remodeled older homes close a large portion of that age-related discount** and renovation year is nearly as predictive as build year
- The biggest price drops are in homes built 1940–1960 with no remodeling

> **For sellers of older homes:** a strategic remodel can recover much of the age-related price discount without the cost of new construction.

---

### Chart 7 : Price per square foot by neighborhood

**Question:** Which neighborhoods are truly value-dense vs. overpriced relative to what you get?

![Chart 7](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Chart_7_pricepersSqF_neighborhood.png)

**Finding:** Absolute median price is misleading, some "expensive" neighborhoods are actually reasonable per square foot, while some "affordable" ones offer poor space-for-money.

- **Value zones** (below city median $/sqft): neighborhoods where buyers get meaningfully more space for their dollar
- **Premium zones** (above city median $/sqft): buyers pay for prestige address, not just space

> Price per sqft is the great equalizer, it removes home size from the equation and reveals true neighborhood pricing power. Buyers prioritizing space over address should target value-zone neighborhoods.

---

## Key Takeaways

1. **Quality beats size**, `OverallQual` is the strongest single predictor of price (r = 0.79). A well-finished smaller home consistently outsells a larger average-quality one.

2. **Neighborhood matters enormously**, a 4× price gap exists between the most and least expensive neighborhoods. Location isn't just about prestige; it's the second-biggest price lever after quality.

3. **Most "missing" data isn't missing**, 99.5% of homes don't have a pool. Treating that as missing data would corrupt the analysis. Understanding *why* data is absent is as important as knowing how much is absent.

4. **Renovation partially offsets age**, older homes that have been remodeled trade significantly closer to newer construction than unmodified older stock.

5. **Engineered features outperformed raw columns**,  `TotalSF` (combining 3 area columns) showed higher correlation with price than any individual area column, confirming that domain knowledge in feature design pays off.

---
## Dashboard

An interactive Power BI dashboard was built to communicate findings
to a non-technical audience, focusing on neighborhood comparisons,
quality premiums, and price distribution.

![Dashboard Preview](dashboard_preview.png)

> 📁 Download `house_prices_dashboard.pbix` to explore the
> interactive version in Power BI Desktop.
---

---
## Machine Learning Modeling Results Summary

This section outlines the process of training and evaluating various machine learning models to predict home sale prices, from baseline performance to hyperparameter tuning and final selection.

### Baseline Model Performance

Eight regression models were initially trained and evaluated on a validation set. The performance is summarized below, ranked by R² Score:

| Model | R² Score | MAE ($) | CV RMSE (log) | RMSE ($) |
|---|---|---|---|---|
| **Lasso** | **0.9167** | **14,456** | **0.12030** | **20,660** |
| **Gradient Boosting** | **0.9116** | **14,076** | **0.12450** | **19,376** |
| Ridge | 0.9057 | 15,459 | 0.12768 | 22,010 |
| Linear Regression | 0.8942 | 15,681 | 0.13127 | 22,606 |
| Random Forest | 0.8769 | 16,175 | 0.13770 | 23,307 |
| SVR | 0.8574 | 18,373 | 0.14040 | 26,592 |
| Decision Tree | 0.7672 | 24,869 | 0.19070 | 39,198 |
| KNN | 0.7439 | 24,655 | 0.19200 | 36,694 |

*Note: MAE and RMSE are converted back to actual dollar values for business interpretability.*

### Justification for Tuned Model Selection

While Lasso initially showed a slightly higher R² on the log-transformed target, **Gradient Boosting** was chosen for further tuning alongside Lasso due to its superior performance in dollar-based metrics (lower MAE and RMSE in dollars), indicating better real-world prediction accuracy, especially for capturing non-linear pricing premiums.

### Tuned Model Performance

Both Gradient Boosting and Lasso underwent hyperparameter tuning. The results on the validation set are as follows:

| Model | R² Score | MAE ($) | RMSE ($) |
|---|---|---|---|
| **Lasso (Tuned)** | **0.9238** | **$13,936** | **$19,418** |
| Gradient Boosting (Tuned) | 0.9205 | $14,030 | $19,757 |

### Overfitting Check for Tuned Models

| Model | Train R² | Validation R² | Gap (Train - Val) | Overfit Status |
|---|---|---|---|---|
| Lasso (Tuned) | 0.9351 | 0.9238 | 0.0113 | Healthy |
| Gradient Boosting (Tuned) | 0.9673 | 0.9205 | 0.0468 | Possible overfit (acceptable for ensembles) |

Both models show acceptable generalization. Lasso (Tuned) has a very small gap, indicating strong generalization, while Tuned Gradient Boosting's larger gap is typical for ensemble models and still within an acceptable range given its performance.

![Plot](https://github.com/Mr-Mata/house-prices-ames/blob/main/Images/Plot_models.png)

### Final Model Selection and Performance

Although **Lasso (Tuned)** achieved a slightly higher R² score and marginally lower MAE and RMSE in dollars, the **Tuned Gradient Boosting** model was ultimately favored for its demonstrated superior consistency and reliability in error distribution, evidenced by a significantly lower standard deviation of residuals and smaller maximum underprediction. This makes Tuned Gradient Boosting a more robust choice for real-world valuation where consistent and less volatile predictions are crucial.

**Selected Best Model: Gradient Boosting (Tuned)**
*   **R² Score**: 0.9205 (model explains 92.05% of price variation)
*   **MAE**: $14,030 (average prediction is off by this amount)
*   **RMSE**: $19,757 (penalizes large misses more heavily)
*   **Residual Std**: $13,344 (lower standard deviation indicates higher consistency in predictions)
---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Mr-Mata/house-prices-ames.git
cd house-prices-ames
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**3. Add the dataset**

Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in a `dataset/` folder.

**4. Run the notebook**

Open `house_prices_eda_cells.ipynb` and run all cells top to bottom.

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn`

---

## Author

**GAV** — [GitHub](https://github.com/Mr-Mata) · [LinkedIn](https://linkedin.com/in/gav08)

*Built as a data analytics and machine learning portfolio project.*
