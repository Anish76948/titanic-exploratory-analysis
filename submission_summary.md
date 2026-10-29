# Titanic Exploratory Data Analysis (EDA) — Submission Summary
**InternCircle Learner Guide — Project 01: Data Science & AI**

---

## 1. Questions Upfront
Before performing data transformations, three core analytical questions were formulated:
1. **Gender:** Did female passengers have a significantly higher survival rate than male passengers?
2. **Ticket Class:** Did passenger ticket class (`pclass`) affect survival chances, and what was the gradient across classes?
3. **Age Categories:** How did survival rates vary across distinct life stages (specifically children vs. adults and seniors)?

---

## 2. Key Data Cleaning Decisions & Methodological Rationale

| Column | Observation | Action Taken | Analytical Rationale |
| :--- | :--- | :--- | :--- |
| **`deck`** | 688 missing values (77.22%) | **Dropped column** | With over three-quarters of data missing, imputing `deck` would introduce severe artificial bias and noise. Dropping prevents false precision. |
| **`age`** | 177 missing values (19.87%) | **Imputed with median grouped by `pclass` and `sex`** | Age distributions vary significantly across class and gender (e.g., 1st class passengers were older than 3rd class steerage). Median is robust against outliers and preserves subgroup variance. |
| **`embarked` / `embark_town`** | 2 missing values (0.22%) | **Imputed with mode (`'S'` / `'Southampton'`)** | With fewer than 0.25% missing records and Southampton representing >72% of all boardings, mode imputation is standard and statistically neutral. |
| **Duplicate Rows** | 107 duplicate demographic rows | **Preserved rows** | Unique identifiers (`PassengerId`, `Name`, `Ticket`) were excluded from the Seaborn dataset. Identical demographic combinations (e.g. 3rd class adult males embarking at Southampton with identical fares) represent distinct historical passengers, not data entry duplicates. |

---

## 3. Analysis & Grouping Findings (Exact Figures)

### A. Survival by Biological Sex
- **Female:** **74.20%** survival rate (233 survivors out of 314 passengers)
- **Male:** **18.89%** survival rate (109 survivors out of 577 passengers)

### B. Survival by Passenger Ticket Class (`pclass`)
- **1st Class:** **62.96%** survival rate (136 survivors out of 216 passengers)
- **2nd Class:** **47.28%** survival rate (87 survivors out of 184 passengers)
- **3rd Class:** **24.24%** survival rate (119 survivors out of 491 passengers)

### C. Survival by Age Band (`[0, 12, 18, 40, 60, 100]`)
- **Child (0–12 yrs):** **57.97%** survival rate (40 survivors out of 69 passengers)
- **Teen (12–18 yrs):** **42.86%** survival rate (30 survivors out of 70 passengers)
- **Middle-aged (40–60 yrs):** **39.06%** survival rate (50 survivors out of 128 passengers)
- **Adult (18–40 yrs):** **36.05%** survival rate (217 survivors out of 602 passengers)
- **Senior (60–100 yrs):** **22.73%** survival rate (5 survivors out of 22 passengers)

---

## 4. Answers to the 3 Research Questions (One Clear Sentence Each)

1. **Question 1 (Sex):**  
   **Female passengers survived at a rate of 74.20% (233 of 314), compared to only 18.89% (109 of 577) for male passengers.**

2. **Question 2 (Class):**  
   **First-class passengers achieved a survival rate of 62.96% (136 of 216), which was substantially higher than the 47.28% (87 of 184) for second-class and 24.24% (119 of 491) for third-class passengers.**

3. **Question 3 (Age Bands):**  
   **Children aged 12 and under had the highest survival rate across all age groups at 57.97% (40 of 69), while seniors aged 60 and older had the lowest survival rate at 22.73% (5 of 22).**

---

## 5. Critical Insight: Correlation vs. Causation
- **Correlation:** Higher passenger class and female gender exhibit strong statistical correlations with survival.
- **Causal Mechanism:** Buying a first-class ticket did not directly cause physiological survival; rather, survival was mediated by:
  1. **Evacuation Policy:** The maritime command *"women and children first"* prioritized females and children during lifeboat loading.
  2. **Physical Accessibility:** First-class staterooms were situated on higher decks in direct proximity to the lifeboat stations, whereas third-class passengers were located in deep steerage cabins with complex stairwells and locked gates delaying deck access.

---

## 6. Submission Checklist Compliance
- [x] Three clear questions formulated upfront.
- [x] Missing values detected, imputed/dropped with explicit domain justifications.
- [x] Groupings performed across sex, class, and age bands with sorted results.
- [x] Visualizations generated with Seaborn and Matplotlib including percentages, labels, and titles.
- [x] Exactly one clear sentence per question with real numerical values (no ambiguous adjectives).
- [x] Correlation distinguished from causation.
- [x] Notebook runs top-to-bottom without errors with all outputs and figures preserved.

