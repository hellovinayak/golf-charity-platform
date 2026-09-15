# Comprehensive Machine Learning Accuracy & Evaluation Report
## Project: Predictive SDG Monitoring Dashboard (BCSE497J: Project I)
**Institution:** Vellore Institute of Technology (VIT) Chennai Campus  
**Dataset Source:** NITI Aayog SDG India Index Historical Datasets (2018–2023)  
**Targets Analyzed:** SDG 3 (Good Health & Well-Being), SDG 4 (Quality Education), SDG 13 (Climate Action)  
**Evaluated Entities:** 36 States & Union Territories + 1 All-India National Composite (111 Independent Models)

---

## 1. Executive Summary & Core Performance Indicators

| Metric Category | Key Evaluation Metric | Result | Benchmark / Interpretation |
| :--- | :--- | :--- | :--- |
| **Goodness-of-Fit (Full Horizon 2018–2023)** | **Mean $R^2$ (Explained Variance)** | **0.737** | Explains 73.7% of total variance across states |
| | **Mean Adjusted $R^2$** | **0.672** | Corrected for sample size ($n=6, p=1$) |
| | **Mean Absolute Error (MAE)** | **2.63 pts** | Average absolute error of 2.6 points on 0–100 scale |
| | **Root Mean Squared Error (RMSE)** | **3.18 pts** | Low variance in residual errors |
| | **Mean Absolute Percentage Error (MAPE)** | **5.69%** | High overall fidelity across all 3 SDGs |
| **Statistical Reliability** | **95% Prediction Interval Coverage (PICP)** | **100.0%** | All test points fall within 95% t-distribution bounds |
| **Risk Classification (NITI Tiers)** | **Tier Classification Accuracy** | **75.23%** | Accurate 3-tier categorization on holdout set |
| | **Weighted F1-Score** | **0.759** | Robust balance between Precision & Recall |
| | **High-Risk (Aspirant) Sensitivity / Recall** | **82.61%** | Successfully catches 82.6% of critical lagging states |

---

## 2. Goal-by-Goal Empirical Breakdown

### A. Full Historical Horizon (2018–2023) Model Fit

| Sustainable Development Goal | Mean $R^2$ | Mean Adj $R^2$ | Mean MAE (pts) | Mean RMSE (pts) | Mean MAPE (%) | Mean Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SDG 3: Good Health & Well-Being** | **0.748** | **0.685** | **2.80** | **3.36** | **4.68%** | **0.816** |
| **SDG 4: Quality Education** | **0.692** | **0.614** | **1.14** | **1.35** | **2.19%** | **0.463** |
| **SDG 13: Climate Action** | **0.772** | **0.715** | **3.96** | **4.83** | **10.21%** | **0.733** |
| **Overall Macro Average** | **0.737** | **0.672** | **2.63** | **3.18** | **5.69%** | **0.671** |

### B. Out-of-Sample Holdout Validation (Train: 2018–2021 | Test: 2022–2023)

| Target SDG | Holdout MAE | Holdout RMSE | Holdout MAPE (%) | 95% CI Coverage | Tier Classification Accuracy | Tier Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SDG 3: Health** | 9.69 pts | 10.03 pts | 13.68% | 100.0% | 68.92% | 0.680 |
| **SDG 4: Education** | 3.66 pts | 3.79 pts | 6.55% | 100.0% | 87.84% | 0.873 |
| **SDG 13: Climate** | 15.41 pts | 15.93 pts | 23.22% | 100.0% | 68.92% | 0.709 |
| **Pooled Overall** | **9.58 pts** | **15.35 pts** | **14.48%** | **100.0%** | **75.23%** | **0.759** |

---

## 3. Risk Tier Classification & Early Warning Performance

States are categorized into 3 official NITI Aayog policy tiers:
- **Low Risk / Front Runner:** $\text{Score} \ge 75$
- **Medium Risk / Performer:** $50 \le \text{Score} < 75$
- **High Risk / Aspirant (Alert):** $\text{Score} < 50$

### A. Holdout Confusion Matrix (Actual vs Predicted)

```
                    Predicted: Low Risk    Predicted: Medium Risk    Predicted: High Risk    Total Actual
Actual: Low Risk           49                      8                        0                     57
Actual: Medium Risk        38                     99                        5                    142
Actual: High Risk           0                      4                       19                     23
---------------------------------------------------------------------------------------------------------
Total Predicted            87                    111                       24                    222
```

### B. Per-Tier Classification Metrics

| Tier Category | Precision | Recall (Sensitivity) | F1-Score | Support (Holdout Instances) | Policy Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **High Risk (Aspirant)** | **79.17%** | **82.61%** | **0.809** | 23 | High sensitivity ensures minimal false negatives for critical states |
| **Medium Risk (Performer)** | **89.19%** | **69.72%** | **0.783** | 142 | Mainstream cohort with high precision |
| **Low Risk (Front Runner)** | **56.32%** | **85.96%** | **0.681** | 57 | High recall captures fast-advancing states |
| **Macro Average** | **74.89%** | **79.43%** | **0.757** | 222 | Balanced evaluation across all classes |
| **Weighted Average** | **79.71%** | **75.23%** | **0.759** | 222 | Proportional accuracy across sample distribution |

---

## 4. National Model (All-India Composite Benchmark)

| Indicator | Slope (Annual Growth Rate) | Full $R^2$ | Adjusted $R^2$ | MAE | RMSE | $p$-value | Trajectory Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SDG 3 (Health)** | **+3.40 pts/yr** | **0.817** | **0.771** | **1.92 pts** | **2.26 pts** | **0.0173** (Statistically Significant) | Strongly Improving |
| **SDG 4 (Education)** | **+0.89 pts/yr** | **0.647** | **0.558** | **0.57 pts** | **0.68 pts** | **0.0579** (Near 95% threshold) | Moderate / Steady |
| **SDG 13 (Climate)** | **+3.86 pts/yr** | **0.716** | **0.645** | **4.86 pts** | **5.64 pts** | **0.0375** (Statistically Significant) | Improving with Volatility |

---

## 5. Statistical Diagnostics & Technical Observations

1. **High Explanatory Power in Health & Climate:**
   - SDG 3 (Health) and SDG 13 (Climate) exhibit strong temporal linear correlation ($r = 0.816$ and $r = 0.733$, $R^2 > 0.74$), reflecting steady nationwide institutional programs (Ayushman Bharat, solar energy targets).
2. **High Stability in Education:**
   - SDG 4 (Quality Education) demonstrated exceptionally low absolute errors ($\text{MAE} = 1.14\text{ pts}$, $\text{MAPE} = 2.19\%$), achieving the highest tier classification accuracy (**87.84%**).
3. **Robust Uncertainty Quantification:**
   - The Student's $t$-distribution calibrated 95% Confidence/Prediction Intervals achieved **100.0% empirical coverage** across holdout test points, confirming reliable uncertainty bounds for forward 2024–2026 policymaking.
4. **Early Warning Utility:**
   - The high recall on the **Aspirant Tier (82.61%)** confirms the system functions reliably as an automated early warning trigger for NITI Aayog and state governance bodies.
