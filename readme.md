# Predicting High-Risk Drug Interactions in FAERS Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Source: openFDA](https://img.shields.io/badge/Data-openFDA-orange.svg)](https://open.fda.gov/data/faers/)

##  Introduction
Adverse Drug Events (ADEs) are a leading cause of hospitalization and mortality worldwide. This project leverages the **FDA Adverse Event Reporting System (FAERS)** to monitor post-market drug safety. 

By analyzing quarterly data from 2025, this research aims to uncover hidden patterns in drug co-prescription that lead to severe clinical outcomes. The ultimate goal is to alert  to undocumented or high-risk **Drug-Drug Interactions (DDIs)** through data-driven insights.

---

##  Research Problems

1.  **Task A: Interaction Mining:** Identifying specific drug combinations (Drug A + Drug B) with strong statistical associations to life-threatening reactions using Association Rule Mining (comparing Apriori and FP-Growth algorithms).
2.  **Task B: Severity Prediction:** Developing a classification pipeline to predict the clinical severity level of a report based on patient demographics and medication history.

---

##  Data Description

* **Source:** [openFDA / FAERS API](https://open.fda.gov/apis/drug/event/)
* **Sample Size:** 108,000 records corresponding to 9 JSON files from 2025.
* **Key Features:** 
    * **Demographics:** Patient age and sex.
    * **Medication:** Medicinal product name, active substance.
    * **Clinical:** Reported reactions and seriousness indicator.

### Dataset Visualizations

#### 1. Seriousness and Demographics
This plot illustrates the distribution of adverse event outcomes and the gender breakdown of the reported cases.

<img src="plots/seriousness_and_sex.png" width="600" alt="Seriousness and Sex">

#### 2. Severity Indicators

<div style="display: flex; align-items: center; gap: 20px;">
    <div style="flex: 1;">A detailed breakdown of the specific seriousness criteria reported (Hospitalization, Death, etc.).</div>
    <div style="flex: 0 0 auto;">
        <img src="plots/severity_indicators_pie.png" width="400" alt="Severity Indicators">
    </div>
</div>

#### 3. Drug Frequency Analysis
Comparison between reported Brand Names and their corresponding Active Substances.

| Brand Names | Active Substances |
| :---: | :---: |
| <img src="plots/top_drug_names.png" width="400"> | <img src="plots/top_active_substances.png" width="400"> |

#### 4. Transactional Complexity
Distributions showing how many unique items are present per report, justifying the use of Association Rule Mining.

<img src="plots/dist_drug_names_per_report.png" width="500" alt="Drug Distribution">

---

## Technical Implementation

### 1. Data Fields Selection - Dimensionality Reduction

To ensure high performance and follow the principle of **Dimensionality Reduction**, we selected only the most relevant fields from the 69 available in the FAERS dataset. This reduces noise and improves the statistical significance of our models.

#### Task A: Interaction Mining (Association Rules)
This task focuses on finding relationships between drugs and adverse reactions.
* **`safetyreportid`**: Acts as the **Transaction ID**. It allows us to group multiple drugs and symptoms into a single "basket" or medical case.
* **`patient.drug.medicinalproduct`**: The commercial brand name. Used to identify associations between specific medications.
* **`patient.drug.activesubstance.activesubstancename`**: The generic active ingredient. This is used to handle redundancy (merging different brands with the same chemical component).
* **`patient.reaction.reactionmeddrapt`**: The standardized medical term for the reaction. This serves as the **Item Label** or "consequent" in our association rules.

#### Task B: Severity Prediction (Supervised Learning)
This task uses patient profiles to predict the clinical outcome of a report.
* **`patient.patientonsetage`**: A numerical feature used for risk assessment, as age often correlates with reaction severity.
* **`patient.patientsex`**: A categorical feature (1=Male, 2=Female) used to capture biological differences in drug responses.
* **`seriousnessindicators`**: Fields such as `seriousnessdeath` or `seriousnesshospitalization` are consolidated to create our **Target Label** (Serious vs. Non-Serious).
* **`patient.reaction.reactionoutcome`**: Used for multi-class classification to provide deeper insights into the patient's recovery status.

---


### 2. Data Processing
* Parsing semi-structured **JSON** payloads from the openFDA API.
* Flattening nested lists of drugs and reactions into a relational format for machine learning.
* Feature engineering on drug classes and patient age buckets.

### Modeling Pipeline
* **Classification:** Logistic Regression, Random Forest, and Support Vector Machines (SVM).
* **Association Rules:** Apriori and FP-Growth to determine Support, Confidence, and Lift.



---

##  Expected Outcomes

* **DDI Ranking:** A prioritized list of the most dangerous drug-drug interactions found in recent data.
* **Model Benchmarking:** A comparative analysis showing which predictive models best handle sparse medical event data.
* **Automated Pipeline:** A Python-based framework for converting raw medical JSON into actionable clinical insights.

