# Description for main_file.py

**Program:** TekWorks  
**Purpose:**  
This document explains the logic and workflow implemented in `filling_the_missing.ipynb`.

## Overview
The file handles missing data in the dataset using statistical checks and multi-feature imputation methods.

## Key Steps
1. Load the dataset.
2. Identify missing values across features.
3. Apply clustering-based imputation.
4. Save the cleaned output dataset.

## Notes
This description helps readers understand the structure without reading the full code.




# 📘 A Story About How Missing LoanAmount Was Filled Using Data-Driven Clusters

## 🌤️ The Beginning: A Dataset With Missing Values  
Inside a financial institution, thousands of applicants apply for loans.  
Each applicant brings details like their income, credit history, and where they live.

But one day, while analyzing the dataset, you notice something troubling:

> **Some rows are missing the LoanAmount.**

This is like reading a story where important chapters are missing.  
To properly analyze the data, these missing chapters must be completed.

But your professor gives a strict instruction:  
**"Use no machine learning. Use data logic."**

So you begin your journey.

---

## 🌱 Step 1: Understanding Income by Dividing It Into Chapters  
Applicant incomes vary. Some applicants earn very little, some moderate, some quite a lot.  
To analyze them properly, you divide all incomes into **four equal-sized groups** (quartiles).

```python
df['income_band'] = pd.qcut(
    df['ApplicantIncome'],
    4,
    labels=['low','mid_low','mid_high','high']
)

Each applicant now belongs to one of these bands:

    low → bottom 25%

    mid_low → next 25%

    mid_high → next 25%

    high → top 25%

These bands help us treat similar applicants together.
It’s like separating characters in a story by background or personality.
🔍 Step 2: Identifying the Incomplete Chapters

You locate all applicants whose LoanAmount is missing.

missing_rows = df[df['LoanAmount'].isna()].copy()

You keep them aside.
These are the "incomplete characters" whose stories you need to complete later.
🧠 Step 3: Forming Clusters Based on Real Applicant Behavior

You observe that LoanAmount is influenced by several features:

    Applicant's income level

    Credit history

    Property area (Urban, Rural, Semiurban)

So you create clusters using:

    income_band

    Credit_History

    Property_Area

These clusters represent groups of applicants who share similar financial traits.

Now you look inside each cluster and find the median LoanAmount for that group.
Then you assign this median to all missing values inside the group.

df['LoanAmount'] = df.groupby(
    ['income_band', 'Credit_History', 'Property_Area'],
    observed=False
)['LoanAmount'].transform(lambda x: x.fillna(x.median()))

This step is the heart of the process.

You are saying:

    "People who are similar should have similar loan amounts,
    so missing loan amounts will be filled using the median of their group."

This is smart, logical, and completely data-driven.
✨ Step 4: Revealing the Completed Chapters

You now display only the rows that originally had missing LoanAmount.

filled = df.loc[missing_rows.index, [
    'Loan_ID','ApplicantIncome','Credit_History','Property_Area','LoanAmount'
]]
filled

Every missing value has now been filled with a meaningful number.
Each one reflects the data pattern of the group the applicant belongs to.

