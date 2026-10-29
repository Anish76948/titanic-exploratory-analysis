# Exploratory Data Analysis (EDA) on the Titanic Dataset
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An exploratory data analysis (EDA) of the classic Titanic passenger manifest, completed as part of the **InternCircle Data Science & AI Internship (Project 01)**.

The project evaluates demographic and socio-economic determinants of survival, employing disciplined missing data imputation, duplicate auditing, life-stage binning, and multivariate interaction visualizations.

---

## 📌 Project Overview & Research Questions

Before transforming or analyzing the data, three focused research questions were established to prevent exploratory data dredging:

1. **Gender:** Did female passengers have a significantly higher survival rate than male passengers?
2. **Ticket Class:** Did passenger ticket class (`pclass`) affect survival chances, and what was the gradient across classes?
3. **Age Categories:** How did survival rates vary across distinct life stages (specifically children vs. adults and seniors)?

---

## 📊 Key Findings & Answers

| Question / Factor | Key Metric / Survival Rate | One-Sentence Quantitative Finding |
| :--- | :--- | :--- |
| **1. Gender (`sex`)** | **Female: 74.20%** (233 / 314)<br>**Male: 18.89%** (109 / 577) | Female passengers survived at a rate of 74.20%, compared to only 18.89% for male passengers. |
| **2. Ticket Class (`pclass`)** | **1st: 62.96%** (136 / 216)<br>**2nd: 47.28%** (87 / 184)<br>**3rd: 24.24%** (119 / 491) | First-class passengers achieved a survival rate of 62.96%, substantially higher than the 47.28% for second-class and 24.24% for third-class passengers. |
| **3. Age Category (`age_band`)** | **Child: 57.97%** (40 / 69)<br>**Teen: 42.86%** (30 / 70)<br>**Middle-aged: 39.06%** (50 / 128)<br>**Adult: 36.05%** (217 / 602)<br>**Senior: 22.73%** (5 / 22) | Children aged 12 and under had the highest survival rate across all age groups at 57.97%, while seniors aged 60 and older had the lowest survival rate at 22.73%. |

---

## 🔬 Data Cleaning & Methodological Rationale

| Column | Missingness | Action Taken | Rationale |
| :--- | :--- | :--- | :--- |
| **`deck`** | 688 (77.22%) | **Dropped column** | More than three-quarters of the values are unrecorded. Imputation would introduce severe synthetic noise and false precision. |
| **`age`** | 177 (19.87%) | **Grouped Median by `(pclass, sex)`** | Passenger age distributions differ markedly across socio-economic classes and gender. Median imputation is robust to outliers and maintains realistic subgroup demographics. |
| **`embarked`** | 2 (0.22%) | **Mode Imputation (`'S'` / Southampton)** | Less than 0.25% of records are missing. Southampton accounts for over 72% of all boardings; mode imputation preserves sample variance. |
| **Duplicates** | 107 rows | **Preserved rows** | Unique identifiers (`PassengerId`, `Name`) were removed in the Seaborn dataset. Identical demographic combinations represent distinct historical individuals on the manifest rather than duplicate entries. |

---

## ⚖️ Correlation vs. Causation

A critical distinction in data science is recognizing that statistical association does not imply direct physical causation:
* **The Correlation:** Female gender and higher passenger class are strongly positively correlated with survival.
* **The Causal Mechanisms:**
  1. **Evacuation Policy:** The maritime command *"women and children first"* was enforced by officers during lifeboat boarding, creating the massive gender survival disparity.
  2. **Physical Ship Architecture:** First- and second-class accommodations were situated on the upper decks adjacent to the lifeboat deck. Third-class steerage passengers were housed in the lower hull behind watertight bulkheads and labyrinthine passageways, physically restricting their access to lifeboats.

---

## 📂 Repository Structure

```text
├── titanic_eda.ipynb              # Fully executed Jupyter Notebook with all outputs & plots
├── titanic_eda.html               # Interactive, self-contained HTML export of the notebook
├── submission_summary.md          # Concise project submission brief
├── generate_and_execute_notebook.py# Python automation script to generate & verify the notebook
├── project-1-guide.pdf            # Official InternCircle project task specification
├── requirements.txt               # Environment dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Anish76948/titanic-exploratory-analysis.git
cd titanic-exploratory-analysis
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook titanic_eda.ipynb
```
Or view the pre-rendered [`titanic_eda.html`](titanic_eda.html) directly in any web browser.

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
