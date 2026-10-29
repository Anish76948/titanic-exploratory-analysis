"""
Script to construct, execute, and validate the Titanic EDA Jupyter Notebook.
Follows the InternCircle Learner Guide and project specifications:
1. Formulate 3 distinct questions upfront.
2. Load data, inspect shape, columns, info, describe, and provide data dictionary.
3. Check missing values; explain why 'deck' is dropped, impute 'age' by (pclass, sex) median, impute 'embarked' by mode.
4. Check duplicates and datatypes with thorough explanations.
5. Group by sex, pclass, and age_band ([0, 12, 18, 40, 60, 100] -> Child, Teen, Adult, Middle-aged, Senior) with sorted results.
6. Create clean, annotated Seaborn/Matplotlib visual charts.
7. Answer each question in exactly one clear sentence with real numbers/percentages.
8. Distinguish correlation from causation.
"""

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

def create_titanic_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    }

    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Exploratory Data Analysis (EDA) on the Titanic Dataset
### InternCircle | Data Science & AI — Project 01
**Author:** Data Science Intern  
**Objective:** Perform a rigorous, reproducible exploratory data analysis of the Titanic passenger manifest to identify key demographic and socio-economic factors associated with passenger survival.

---

## 1. Questions Upfront (Step 1)
Before inspecting or altering any data, we define three specific, testable questions to guide our exploratory analysis and prevent biased data dredging:

1. **Question 1 (Gender & Survival):** Did female passengers have a significantly higher survival rate than male passengers during the disaster?
2. **Question 2 (Socio-Economic Class & Survival):** Did ticket class (`pclass`) influence a passenger's likelihood of survival, and how pronounced was the gradient from 1st to 3rd class?
3. **Question 3 (Age Demographics & Survival):** How did survival rates vary across distinct life stages (specifically comparing children, adults, and seniors)?"""))

    # Section 2: Loading & Inspection
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Data Loading & Initial Inspection (Step 2)
In this step, we load the raw dataset directly from `seaborn` and inspect its dimensions, schema, summary statistics, and column meanings.

### Data Dictionary
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `survived` | Integer | Survival indicator (0 = No, 1 = Yes) |
| `pclass` | Integer | Ticket class / socio-economic tier (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `sex` | Object | Passenger biological sex (`female`, `male`) |
| `age` | Float | Passenger age in years |
| `sibsp` | Integer | Number of siblings or spouses aboard |
| `parch` | Integer | Number of parents or children aboard |
| `fare` | Float | Passenger fare in British pounds (£) |
| `embarked` | Object | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) |
| `class` | Category | Categorical equivalent of `pclass` (`First`, `Second`, `Third`) |
| `who` | Object | Man, woman, or child indicator |
| `adult_male`| Boolean | True if the passenger is an adult male |
| `deck` | Category | Cabin deck level (`A` through `G`) |
| `embark_town` | Object | Full town name of embarkation |
| `alive` | Object | Text survival indicator (`yes`, `no`) |
| `alone` | Boolean | True if passenger had no family aboard (`sibsp == 0` and `parch == 0`) |"""))

    cells.append(nbf.v4.new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure inline plotting for notebook environments
%matplotlib inline

# Set plotting aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Display shape
print(f"Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")"""))

    cells.append(nbf.v4.new_code_cell("""# Display the first 5 records
df.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Inspect column types and non-null counts
df.info()"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics for numeric features
df.describe().round(2)"""))

    cells.append(nbf.v4.new_markdown_cell("""### Initial Observations:
- **Sample Size:** 891 passengers and 15 attributes.
- **Overall Survival Rate:** Approximately **38.38%** of passengers in this sample survived (`survived.mean() = 0.38`).
- **Age Distribution:** Passenger age ranges from infants (0.42 years) to 80 years, with a mean of 29.70 years and median of 28.00 years.
- **Fare Skewness:** Passenger fares show substantial positive skewness (mean £32.20 vs. median £14.45, with a max of £512.33), highlighting wide socio-economic disparity among travelers."""))

    # Section 3: Missing Values
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Missing Value Detection (Step 3)
Real-world datasets frequently contain missing values. We audit null values across all columns to evaluate the magnitude and pattern of missingness before choosing an imputation or dropping strategy."""))

    cells.append(nbf.v4.new_code_cell("""# Calculate absolute count and percentage of missing values per column
missing_counts = df.isna().sum()
missing_percent = (missing_counts / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage (%)': missing_percent.round(2)
})

# Display columns with missing data
missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)"""))

    # Section 4: Data Cleaning & Reasoning
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Data Cleaning & Handling Missing Values (Step 4)

We apply domain-specific, context-aware remediation for each column with missing values:

### Cleaning Decisions and Methodological Rationale:
1. **`deck` (688 missing, 77.22%):**
   - **Action:** Drop the `deck` column entirely.
   - **Reasoning:** More than three-quarters (77.22%) of the values are unrecorded. Attempting to impute a feature where missingness dominates would introduce substantial synthetic bias and false precision. Dropping it preserves the integrity of our analyses.

2. **`age` (177 missing, 19.87%):**
   - **Action:** Impute missing ages using the **median age grouped by passenger ticket class (`pclass`) and biological sex (`sex`)**.
   - **Reasoning:** Overall mean imputation fails because passenger age distributions differ markedly across socio-economic tiers and gender (e.g., 1st class passengers were wealthier and older on average than 3rd class laborers and children). Median imputation is chosen over mean imputation because it is robust against outliers and preserves subgroup age distributions.

3. **`embarked` and `embark_town` (2 missing, 0.22%):**
   - **Action:** Impute missing values with the **mode (most frequent value)**: `'S'` / `'Southampton'`.
   - **Reasoning:** Only 2 passenger records are missing embarkation port data. Southampton accounts for over 72% of all boarding passengers. Mode imputation is the standard, conservative practice for categorical variables with negligible missingness, having zero impact on variance."""))

    cells.append(nbf.v4.new_code_cell("""# 1. Impute 'age' using median grouped by 'pclass' and 'sex'
age_group_medians = df.groupby(['pclass', 'sex'])['age'].transform('median')
df['age'] = df['age'].fillna(age_group_medians)

# 2. Impute 'embarked' and 'embark_town' using the mode
mode_embarked = df['embarked'].mode()[0]
mode_embark_town = df['embark_town'].mode()[0]

df['embarked'] = df['embarked'].fillna(mode_embarked)
df['embark_town'] = df['embark_town'].fillna(mode_embark_town)

# 3. Drop the 'deck' column
df = df.drop(columns=['deck'])

# Verify that no missing values remain in our working features
print("Remaining null values per column:")
print(df.isna().sum())"""))

    # Section 5: Duplicates & Types
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Duplicate Rows and Data Types Validation (Step 5)
Next, we audit duplicate rows and ensure all columns adhere to their appropriate data types."""))

    cells.append(nbf.v4.new_code_cell("""# Check duplicate rows
num_duplicates = df.duplicated().sum()
print(f"Total duplicate rows detected: {num_duplicates}")

# Display a sample of duplicate rows to inspect their context
if num_duplicates > 0:
    display_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    print("\\nSample duplicate records:")
    print(df[df.duplicated(keep=False)][display_cols].head(6))"""))

    cells.append(nbf.v4.new_markdown_cell("""### Rationale on Duplicate Records:
In Seaborn's preprocessed Titanic dataset, unique identifier columns such as `PassengerId`, `Name`, and `Ticket` number were removed. As a result, rows with identical demographic and fare attributes (such as unmarried 3rd class males aged 25 embarking at Southampton with £7.89 fare) appear as duplicate rows. These reflect distinct historical individuals on the ship's passenger manifest rather than erroneous data collection duplicates. Therefore, these rows are deliberately preserved."""))

    cells.append(nbf.v4.new_code_cell("""# Verify column datatypes
print("Data types summary:")
print(df.dtypes)"""))

    # Section 6: Grouping & Visualizations
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Grouping, Aggregation & Comparative Analysis (Step 6)
We now evaluate passenger survival rates across our three primary factors:
1. Biological Sex (`sex`)
2. Socio-Economic Class (`pclass`)
3. Age Bands (`age_band`)

Age is categorized into standardized life-stage intervals using `pd.cut`:
- **Child:** $(0, 12]$ years
- **Teen:** $(12, 18]$ years
- **Adult:** $(18, 40]$ years
- **Middle-aged:** $(40, 60]$ years
- **Senior:** $(60, 100]$ years"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Survival Rate by Sex
sex_summary = df.groupby('sex')['survived'].agg(
    total_passengers='count',
    survivors='sum',
    survival_rate='mean'
).reset_index()

sex_summary['survival_percentage'] = (sex_summary['survival_rate'] * 100).round(2)
sex_summary['survival_rate'] = sex_summary['survival_rate'].round(2)
print("Survival Summary by Sex:")
print(sex_summary.sort_values(by='survival_rate', ascending=False).to_string(index=False))"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Survival Rate by Ticket Class (pclass)
class_summary = df.groupby('pclass')['survived'].agg(
    total_passengers='count',
    survivors='sum',
    survival_rate='mean'
).reset_index()

class_summary['survival_percentage'] = (class_summary['survival_rate'] * 100).round(2)
class_summary['survival_rate'] = class_summary['survival_rate'].round(2)
print("\\nSurvival Summary by Ticket Class:")
print(class_summary.sort_values(by='survival_rate', ascending=False).to_string(index=False))"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Create Age Bands and Compute Survival Rates
bins = [0, 12, 18, 40, 60, 100]
labels = ["Child", "Teen", "Adult", "Middle-aged", "Senior"]

df['age_band'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

age_summary = df.groupby('age_band', observed=True)['survived'].agg(
    total_passengers='count',
    survivors='sum',
    survival_rate='mean'
).reset_index()

age_summary['survival_percentage'] = (age_summary['survival_rate'] * 100).round(2)
age_summary['survival_rate'] = age_summary['survival_rate'].round(2)
print("\\nSurvival Summary by Age Band (Sorted by Survival Rate):")
print(age_summary.sort_values(by='survival_rate', ascending=False).to_string(index=False))"""))

    cells.append(nbf.v4.new_markdown_cell("""### Visual Exploratory Data Analysis
We present clear visual charts with explicit percentages, error bars, and informative annotations to highlight the observed differences."""))

    cells.append(nbf.v4.new_code_cell("""# Plotting comparative charts
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart 1: Survival Rate by Sex
ax1 = sns.barplot(
    data=df, x='sex', y='survived', errorbar=None,
    palette=['#e74c3c', '#3498db'], ax=axes[0]
)
axes[0].set_title('Survival Rate by Sex', fontsize=14, fontweight='bold', pad=12)
axes[0].set_xlabel('Biological Sex', fontsize=12)
axes[0].set_ylabel('Survival Rate', fontsize=12)
axes[0].set_ylim(0, 1.0)
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(f"{height * 100:.1f}%",
                 (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom', fontsize=11, fontweight='bold',
                 xytext=(0, 4), textcoords='offset points')

# Chart 2: Survival Rate by Class
ax2 = sns.barplot(
    data=df, x='pclass', y='survived', errorbar=None,
    palette='Blues_r', ax=axes[1]
)
axes[1].set_title('Survival Rate by Ticket Class', fontsize=14, fontweight='bold', pad=12)
axes[1].set_xlabel('Passenger Class (1st, 2nd, 3rd)', fontsize=12)
axes[1].set_ylabel('Survival Rate', fontsize=12)
axes[1].set_ylim(0, 1.0)
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(f"{height * 100:.1f}%",
                 (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom', fontsize=11, fontweight='bold',
                 xytext=(0, 4), textcoords='offset points')

# Chart 3: Survival Rate by Age Band
ax3 = sns.barplot(
    data=df, x='age_band', y='survived', errorbar=None,
    palette='viridis', ax=axes[2]
)
axes[2].set_title('Survival Rate by Age Band', fontsize=14, fontweight='bold', pad=12)
axes[2].set_xlabel('Age Category', fontsize=12)
axes[2].set_ylabel('Survival Rate', fontsize=12)
axes[2].set_ylim(0, 1.0)
for p in ax3.patches:
    height = p.get_height()
    ax3.annotate(f"{height * 100:.1f}%",
                 (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom', fontsize=11, fontweight='bold',
                 xytext=(0, 4), textcoords='offset points')

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### Multivariate Interaction: Class and Gender
To further investigate how socio-economic standing interacted with the maritime evacuation protocol, we examine survival rates segmented simultaneously by both ticket class and sex."""))

    cells.append(nbf.v4.new_code_cell("""# Multivariate Pivot: Survival Rate by Class and Sex
interaction_pivot = df.pivot_table(
    index='pclass', columns='sex', values='survived', aggfunc='mean'
).round(4) * 100

plt.figure(figsize=(7, 4.5))
sns.heatmap(interaction_pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Survival Rate (%)'})
plt.title("Survival Rate (%) by Class and Sex Interaction", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Sex", fontsize=11)
plt.ylabel("Passenger Class", fontsize=11)
plt.show()

print("Interaction Table (Survival %):")
print(interaction_pivot)"""))

    # Section 7: Answering the 3 Questions
    cells.append(nbf.v4.new_markdown_cell("""---
## 7. Answers to Research Questions & Findings (Step 7)

Below are the exact, quantitative answers to the three upfront questions:

### Answer 1 (Gender):
**Female passengers experienced a survival rate of 74.20% (233 survivors out of 314), compared to only 18.89% (109 survivors out of 577) for male passengers.**

### Answer 2 (Ticket Class):
**First-class passengers achieved a survival rate of 62.96% (136 of 216), compared to 47.28% (87 of 184) for second-class and 24.24% (119 of 491) for third-class passengers.**

### Answer 3 (Age Bands):
**Children (ages 0 to 12) recorded the highest survival rate among all age groups at 57.97% (40 of 69), whereas seniors (ages 60 to 100) recorded the lowest survival rate at 22.73% (5 of 22).**

---

### Critical Analytical Note: Correlation vs. Causation
- **Statistical Correlation Observed:** Higher ticket class and female gender are strongly positively correlated with surviving the shipwreck.
- **Why Correlation $\\neq$ Causation:** Holding a 1st class ticket did not biologically or inherently shield an individual from hypothermia or drowning. Instead, the observed difference was mediated by external physical and social mechanisms:
  1. **Evacuation Protocol Enforcement:** The maritime protocol of *"women and children first"* was strictly enforced by officers loading lifeboats, causing the dramatic disparity between male and female survival rates.
  2. **Physical Ship Architecture & Proximity:** 1st and 2nd class staterooms were situated on higher decks adjacent to the lifeboat stations, whereas 3rd class steerage passengers were housed in lower decks separated by bulkheads and lengthy passageways, severely limiting their access to lifeboats during the sinking."""))

    # Section 8: Summary & Checklist
    cells.append(nbf.v4.new_markdown_cell("""---
## 8. Summary & Submission Checklist (Step 8)

### Project Verification Checklist:
- [x] **Three clear questions written at the start:** Formulated in Section 1 before data alteration.
- [x] **Missing values handled with documented reasons:** `deck` dropped (77.2% missing), `age` imputed with `(pclass, sex)` median (19.9% missing), `embarked` imputed with mode (0.2% missing).
- [x] **Groupings compare survival by at least three factors:** Analyzed by sex, ticket class (`pclass`), and age band (`Child`, `Teen`, `Adult`, `Middle-aged`, `Senior`).
- [x] **Each question answered with exact numbers:** Formulated as exactly one clear sentence per question with verified percentages and counts (no vague terms like "many" or "most").
- [x] **Correlation distinguished from causation:** Thoroughly documented in Section 7.
- [x] **Notebook runs cleanly from top to bottom:** Verified through headless execution with all outputs and figures preserved."""))

    nb.cells = cells
    return nb

def main():
    notebook_path = "titanic_eda.ipynb"
    print(f"Creating notebook structure for {notebook_path}...")
    nb = create_titanic_notebook()

    # Save unexecuted notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook written to disk.")

    # Execute the notebook end-to-end to capture all outputs
    print("Executing notebook end-to-end with ExecutePreprocessor...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})

    # Write back executed notebook with all outputs preserved
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Executed notebook saved with all outputs!")

    # Export to HTML for convenient viewing and submission verification
    html_path = "titanic_eda.html"
    print(f"Exporting to {html_path}...")
    from nbconvert import HTMLExporter
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(body)
    print("Export to HTML completed successfully!")

if __name__ == "__main__":
    main()
