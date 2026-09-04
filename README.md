
# 👥 Employee Attrition Prediction — HR Analytics & Machine Learning

Predicting which employees are about to leave — and why — from 11,991 HR records, using Python, Pandas, and a Random Forest classifier (98.1% accuracy, 90.3% recall).

## 📌 TL;DR

Replacing an employee costs far more than retaining one, so knowing who's about to leave — before they hand in notice — is worth real money to an HR team. This project analyzes 11,991 employee records (after cleaning 3,008 duplicate rows out of a raw 14,999), identifies the strongest drivers of attrition through EDA and correlation analysis, and trains a Random Forest classifier that predicts attrition with 98.08% accuracy, 90.27% recall, and 98.10% precision on held-out data — while also surfacing which features actually drive that prediction, not just the score.

### 🎯 The Story Behind This Project

The accuracy score isn't the point — here's the thinking behind it.

### What problem was I solving? 

Attrition is expensive, but "employees leave sometimes" isn't an actionable insight — HR needs to know which employees, and why, early enough to intervene. The real question wasn't "build a classifier" — it was whether attrition is driven by a handful of identifiable, addressable factors (satisfaction, workload, promotion history, compensation) or whether it's effectively random. If it's the former, a model isn't just a prediction tool, it's a prioritization tool for a retention budget that can't cover everyone.

### Why did I choose these metrics?

Correlation and feature importance, not just prediction accuracy — because a model that predicts well but can't say why is far less useful to an HR team than one that also ranks the drivers. Accuracy tells you the model works; feature importance tells you what to actually do about it.
Recall on the "left" class specifically, not just overall accuracy — because in attrition, a false negative (predicting someone will stay when they actually leave) is the costlier mistake: it's a preventable resignation nobody saw coming. Overall accuracy alone can look great while quietly missing the cases that matter most, so recall on the positive class was tracked deliberately alongside precision and F1.
Attrition rate broken out by department and salary band, not just an overall rate — because "16.6% of employees leave" doesn't tell HR where to focus a retention program, but "HR and Accounting have the highest attrition, low-salary employees leave 4x more than high-salary employees" does.
Cross-validation over a single train/test split — because a single 98% accuracy score could be a lucky split; 5-fold cross-validation (consistently ~98.3–98.7% across folds) confirms the model's performance is stable, not a fluke.

### What did I find — and what decision does it enable? 

See Key Findings and Insights & Recommendations for the full breakdown, but the short version: satisfaction level is, by a wide margin, the single strongest predictor of attrition (both in raw correlation and in the trained model's feature importance) — which turns "reduce attrition" into a much narrower, more actionable question: what's driving satisfaction down for the employees most likely to leave. The model itself also earns its place here — 90.27% recall means it catches the large majority of actual leavers, which is exactly the property an early-warning tool needs.

## 🧩 The Business Problem
Attrition is happening, but there's no clear read on which employees are actually at risk before they've already decided to leave.
HR doesn't have a ranked view of which factors — pay, workload, promotion history, department, tenure — actually predict attrition versus which just correlate loosely with it.
Retention budget and manager attention are limited resources; without a risk score, they get spread evenly instead of where they're needed most.
Some commonly assumed drivers of attrition may not hold up under scrutiny once the data is actually analyzed — assuming instead of checking risks building a retention strategy around the wrong problem.

## ❓ Stakeholder Questions This Project Answers

HR Leadership

What's the overall attrition rate, and is it evenly spread or concentrated in specific groups?
Which departments and salary bands have the highest attrition, and by how much?
Does promotion history actually reduce attrition, and by how much?

People Managers

Which of my team's characteristics (workload, tenure, recent evaluation score) are associated with higher flight risk?
Can I get an early-warning signal for at-risk employees rather than finding out after they've resigned?

Data Science / People Analytics

Which features are actually predictive of attrition, ranked by importance — not just correlated with it?
How reliable is a trained model's performance — does it hold up across cross-validation, or is it overfit to one split?
What would it take to productionize this as an ongoing risk score rather than a one-off analysis?

## 🏗️ Project Workflow
1. Data Loading & Profiling — load 14,999 raw HR records (10 columns: satisfaction, evaluation score, project count, monthly hours, tenure, work accidents, promotions, department, salary band, and the left target) and inspect structure, types, and duplicates.
2. Data Cleaning — remove exact duplicate rows and confirm there are no missing values in the remaining data.
3. Exploratory Data Analysis — target distribution, cross-tabs of attrition against salary and department, boxplots of key numeric features against attrition, a full correlation matrix and heatmap, and department/salary attrition-rate rankings.
4. Feature Engineering — encode categorical fields (Department, salary) and scale numeric features for modeling.
5. Model Training — a Random Forest classifier (200 trees) trained on an 80/20 train/test split.
6. Evaluation — confusion matrix, accuracy, precision, recall, and F1 score on the held-out test set.
7. Feature Importance — rank which features the trained model actually relies on most.
8. Validation & Tuning — 5-fold cross-validation to confirm stability, followed by a GridSearchCV hyperparameter search (see Notes on the Modeling Code for an honest caveat on this step).

## 🧹 Data Cleaning & Preparation
. Source data: 14,999 employee records × 10 columns, loaded with Pandas.

. Duplicates: 3,008 exact duplicate rows were found and removed, leaving 11,991 unique employee records for all analysis and modeling — confirmed directly from the post-

. cleaning class counts (10,000 retained + 1,991 left = 11,991).

. Missing values: none — a full .isnull().sum() check on the deduplicated data returned zero missing values across all 10 columns.

. Encoding: salary (low/medium/high) and Department were label-encoded to numeric codes for the model. Note: label encoding imposes an implicit order on Department, which has no natural ranking — see the notes section below for why one-hot encoding would be the more methodologically correct choice here.

. Scaling: numeric features were standardized with StandardScaler ahead of cross-validation.


## Key Findings

📉 Overall attrition rate: 16.61% (1,991 of 11,991 employees left) — a meaningful minority, not a marginal edge case.

🎯 Satisfaction level is the strongest single driver of attrition, by a wide margin — it has the largest correlation with attrition of any feature (−0.35) and is the top-ranked feature in the trained model (35.86% of total feature importance, more than double the next-highest feature). Employees who left averaged a 0.44 satisfaction score vs. 0.67 for those who stayed.

💰 Salary band shows a clear, steep gradient: employees on low salary churn at 20.45%, medium at 14.62%, and high salary at just 4.85% — a more than 4x difference between the lowest and highest bands.

🏢 HR (18.80%) and Accounting (17.55%) have the highest departmental attrition; Management (11.93%) and R&D (12.25%) have the lowest — a real spread worth investigating rather than treating attrition as uniform across the business.

🏆 Promotion history is one of the strongest retention levers found: employees promoted in the last 5 years left at just 3.94%, versus 16.82% for those who weren't — over a 4x difference.

⚠️ A counterintuitive finding: employees who'd had a workplace accident left less often (5.68%) than those who hadn't (18.60%), a negative correlation (−0.125) opposite to what might be assumed. Worth investigating further rather than dismissing — it may reflect increased support/attention after an incident rather than a causal protective effect.

⏰ Employees who left worked more hours and had longer tenure than those who stayed (208 vs. 199 average monthly hours; 3.88 vs. 3.26 years tenure) — suggesting attrition risk may be concentrated among overworked, longer-tenured staff rather than new hires.

🤖 The trained Random Forest model performs strongly: 98.08% accuracy, 90.27% recall, 98.10% precision, 94.03% F1 on the test set (2,399 held-out employees), and this holds up under 5-fold cross-validation (mean accuracy 98.60%, ranging 98.33%–98.85% across folds) — the performance isn't a lucky split.

## 🤖 Model Performance
. Model: Random Forest Classifier, 200 trees, trained on an 80/20 train/test split (9,592 train / 2,399 test rows).

. Confusion matrix (test set, 2,399 employees):

. True Negatives (correctly predicted "stayed"): 1,991

. False Positives (predicted "left," actually stayed): 7

. False Negatives (predicted "stayed," actually left): 39

. True Positives (correctly predicted "left"): 362

Accuracy: 98.08%
Precision (on the "left" class): 98.10%
Recall (on the "left" class): 90.27% — the model correctly flags roughly 9 out of every 10 employees who actually leave.
F1 Score: 94.03%
5-fold cross-validation accuracy: 98.60% average (98.33%–98.85% range) — confirms the test-set result is stable, not a one-off split.
Top 5 features by importance: satisfaction_level (35.86%), number_project (17.69%), time_spend_company (17.21%), average_montly_hours (15.31%), last_evaluation (11.39%) — together accounting for over 97% of the model's decision-making, with Department, salary, Work_accident, and promotion_last_5years contributing comparatively little to the model's predictions despite showing meaningful patterns in the raw EDA.

## 🧠 Insights & Recommendations

1. Satisfaction is the lead indicator — treat it as one Satisfaction level is both the strongest raw correlate of attrition and the model's top feature by a wide margin, yet it's rarely tracked as a leading indicator the way performance metrics are.

Recommendation: Run a regular (e.g. quarterly) employee satisfaction pulse survey and feed the score into whatever system tracks attrition risk — right now this is the single highest-leverage number the business isn't systematically watching.
Owner: HR Leadership / People Analytics
Priority: 🔴 High — the strongest lever identified in the entire analysis

2. Low-salary employees are churning at 4x the rate of high-salary employees 20.45% attrition at the low salary band vs. 4.85% at the high band is a steep, consistent gradient — not a marginal difference.

Recommendation: Review compensation bands specifically at the low tier for market competitiveness, prioritizing roles and departments where attrition is already elevated (HR, Accounting). A targeted compensation review is more cost-effective than an across-the-board raise.
Owner: Compensation & Benefits
Priority: 🔴 High — large, clearly quantified effect

3. Promotion pathways are a proven retention lever — but rarely used Promoted employees leave at 3.94% versus 16.82% for those who weren't — a bigger effect than salary band alone — yet the data shows promotions are rare overall.

Recommendation: Build clearer, more visible promotion pathways, especially in the departments with both high attrition and low promotion rates (HR, Accounting, Technical) — this is a retention lever that's currently underused relative to how strong its effect is.
Owner: HR Leadership / People Managers
Priority: 🔴 High — strong effect, currently underutilized

4. Attrition risk skews toward overworked, longer-tenured employees, not new hires Employees who left averaged more monthly hours and longer tenure than those who stayed — the opposite of the "new hires are the flight risk" assumption that often drives onboarding-focused retention efforts.

Recommendation: Extend retention check-ins and workload reviews beyond the typical first-90-days window — the data points to sustained overwork among established employees as a bigger risk factor here, not early-tenure disengagement.
Owner: People Managers
Priority: 🟠 Medium — reframes where retention effort should focus

5. The work-accident finding needs investigation before action, not a policy change Employees with a work-accident history left less often, which runs counter to intuition. Acting on this directly (e.g., assuming accidents are "good" for retention) would be a misuse of a correlational finding.

Recommendation: Treat this as a flag for further investigation (e.g., checking whether post-accident employees receive more manager attention or support) rather than a lever to pull — the honest next step is understanding the mechanism, not acting on the correlation alone.
Owner: People Analytics
Priority: 🟢 Low — interesting, but needs more evidence before it's actionable

6. The model is ready to support prioritization, not yet ready to fully replace judgment 90.27% recall means roughly 1 in 10 actual leavers would currently be missed by the model — good enough to prioritize HR's attention, not yet reliable enough to be the sole signal for high-stakes retention decisions.

Recommendation: Deploy the model as a risk-ranking tool that flags employees for a manager conversation, not as an automated decision-maker — and revisit the classification threshold, since in this use case a missed at-risk employee (false negative) is more costly than a false alarm (false positive).
Owner: People Analytics / Data Science
Priority: 🟡 Medium — a deployment/rollout decision, not a data gap

## 🛠️ Tech Stack
Data manipulation — Python, Pandas, NumPy
Visualization — Matplotlib, Seaborn
Machine learning — Scikit-learn (Random Forest, train/test split, StandardScaler, LabelEncoder, GridSearchCV, cross-validation)
Environment — Jupyter Notebook


## 🚀 What This Project Demonstrates
Full ML lifecycle: cleaning, EDA, feature engineering, model training, evaluation, feature importance, and validation — not just one stage of it.
The right metric for the problem: prioritizing recall on the positive class over raw accuracy, because in an attrition context, missing an at-risk employee is the costlier error.
Business framing over pure technique: every EDA finding and every model output is tied back to a specific HR decision, not reported as an isolated statistic.
Honest self-review: documenting real issues in the modeling pipeline (encoding choices, an incomplete tuning step, a script bug) instead of only presenting the clean, final numbers — a stronger signal of technical maturity than a notebook with no visible rough edges.

## 📬 Contact

[Sakshi Dave] — sakshidave115@gmail.com

If you're a recruiter or hiring manager and want to talk through the design decisions behind this project, I'd love to chat.
