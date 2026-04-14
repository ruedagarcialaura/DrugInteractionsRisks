# Predicting High-Risk Drug Interactions in FAERS Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Source: openFDA](https://img.shields.io/badge/Data-openFDA-orange.svg)](https://open.fda.gov/data/faers/)

##  Introduction
Adverse Drug Events (ADEs) are a leading cause of hospitalization and mortality worldwide. This project leverages the **FDA Adverse Event Reporting System (FAERS)** to monitor post-market drug safety. 

By analyzing quarterly data from **2024–2025**, this research aims to uncover hidden patterns in drug co-prescription that lead to severe clinical outcomes. The ultimate goal is to alert healthcare providers to undocumented or high-risk **Drug-Drug Interactions (DDIs)** through data-driven insights.

---

##  Research Problems

1.  **Interaction Mining:** Identifying specific drug combinations (e.g., Drug A + Drug B) with strong statistical associations to life-threatening reactions using **Association Rule Mining** (comparing Apriori and FP-Growth algorithms).
2.  **Severity Prediction:** Developing a classification pipeline to predict the clinical severity level of a report based on patient demographics and medication history.

---

##  Data Description

* **Source:** [openFDA / FAERS API](https://open.fda.gov/apis/drug/event/)
* **Sample Size:** ~120,000 to 150,000 records.
* **Key Features:** * **Demographics:** Patient age, sex.
    * **Medication:** Medicinal product name, administration route.
    * **Clinical:** Reported reactions and outcomes.
* **Target Variable:** `Clinical Severity` (Derived from death, hospitalization, and disability indicators).

---

##  Technical Implementation

### Data Processing
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

