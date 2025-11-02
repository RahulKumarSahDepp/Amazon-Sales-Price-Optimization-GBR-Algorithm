# Amazon-Sales-Price-Optimization-GBR-Algorithm
Tuned GBR model (R² 0.637) predicts Amazon demand ($\log(\text{Sales})$) from price, reviews &amp; status. Optimizes pricing strategy by simulating revenue; confirmed social proof is 7x stronger than price.
# 💰 Amazon Product Pricing Optimization Model

This project is an end-to-end Machine Learning implementation focused on maximizing revenue for Amazon sellers by predicting product demand (sales volume) as a function of pricing, quality, and promotional status.

The final model is a **Tuned Gradient Boosting Regressor (GBR)**, which achieved a high $R^2$ score and provides robust feature importance and simulation capabilities for strategic pricing decisions.

---

## 🛠️ 1. Data Cleaning and Preparation

The initial dataset was messy, containing string representations of numbers, complex date formats, and redundant data.

| Task | Outcome |
| :--- | :--- |
| **Sales Volume** | Converted strings like "10K+ bought" to quantified integer values (`purchased_last_month`). |
| **Reviews & Rating**| Extracted numerical values from string formats and converted columns to `float64`. |
| **Feature Engineering**| Created the target variable: **`log_purchased_last_month`** to correct for sales data skewness. |
| **Encoding** | Binary encoded all promotional status columns (`is_best_seller`, `is_sponsored`, `has_coupon`) to 1s and 0s. |

---

## 🔎 2. Exploratory Data Analysis (EDA)

The EDA focused specifically on identifying the true linear and non-linear drivers of demand to inform feature selection.

### Key Insights:

* **Social Proof Dominance:** The **Total Reviews** column showed the strongest linear correlation with sales ($\text{r} = 0.33$), significantly higher than price.
* **Price Elasticity:** The relationship between discounted price and sales is highly **non-linear**. Sales volume is overwhelmingly concentrated below the **$40 price point**.
* **Promotional Lift:** The **"Best Seller"** status was found to provide an approximate $\mathbf{7\times}$ multiplier in median sales volume compared to non-badged products.
* **Multicollinearity:** The `original_price` column was dropped due to perfect correlation with `discounted_price`.

---

## 🧠 3. Model Building and Evaluation

The problem was structured as a **Regression Task** to predict the continuous variable $\log(\text{Sales})$.

### Model Comparison:

| Algorithm | $\mathbf{R^2}$ Score (Before Tuning) | Rationale |
| :--- | :--- | :--- |
| **Ridge Regression** | Low | Cannot handle the complex, non-linear demand curve. |
| **Random Forest** | $\approx 0.47$ | Good baseline, but struggles to model the specific sequential error patterns. |
| **Gradient Boosting**| $\approx 0.45$ (Initial) | Strong potential due to sequential learning. |

### Final Tuned Model: Gradient Boosting Regressor (GBR)

The GBR was selected and fine-tuned using `RandomizedSearchCV`, focusing on improving the learning rate and tree depth.

| Metric | Tuned Result | Improvement |
| :--- | :--- | :--- |
| **R² Score** | $\mathbf{0.6372}$ | **35% increase** over the best baseline (RF). |
| **MAE** | $\mathbf{1136.95}$ units | Average error in predicting the original sales volume. |
| **Best Params** | `{'subsample': 0.9, 'n_estimators': 300, 'max_depth': 7, ...}` | Optimized for best predictive power. |

---

## 💡 4. Final Pricing Strategy Recommendations

The final model provides the intelligence to shift from static pricing to a dynamic, revenue-maximizing strategy.

1.  **Reputation First, Price Second:** Allocate resources to driving **reviews** and achieving **Best Seller** status, as the GBR model confirms these factors have the highest predictive importance.
2.  **Revenue Simulation:** Use the final GBR model to run **What-If Scenarios**. By simulating Predicted Sales $\times$ Price across a range, the model pinpoints the specific $\mathbf{Revenue-Maximizing Price}$ for any product configuration (e.g., $50\text{K}$ reviews, sponsored, no badge).
3.  **Promotional Strategy:** Use price cuts strategically to **trigger promotional badges** (Best Seller, Coupon) rather than relying on the discount percentage alone to boost sales.
4.  **Avoid Suboptimal Zones:** Ensure high-volume products remain priced within the $\mathbf{\$15 - \$40}$ affordability band to maintain mass-market demand.

---

### Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn (Ridge, RandomForest, GradientBoosting)
* Matplotlib, Seaborn
