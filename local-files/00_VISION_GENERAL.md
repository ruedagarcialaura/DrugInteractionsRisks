# VISIÓN GENERAL DEL PROYECTO
## Drug Interactions Risk — FAERS Data Mining

---

## ¿Qué es este proyecto?

Este proyecto es una investigación de **Pharmacovigilance** (vigilancia farmacéutica post-mercado). Se toman datos reales del sistema de reportes de eventos adversos de la FDA americana (FAERS - FDA Adverse Event Reporting System) y se aplican técnicas de Machine Learning para:

1. **Task A** → Descubrir combinaciones de fármacos que estadísticamente producen reacciones adversas graves (Association Rule Mining)
2. **Task B** → Predecir si un nuevo caso de un paciente va a ser severo o no (Binary Classification)

**Los datos son reales**: ~108,000 reportes de pacientes reales de 9 archivos JSON trimestrales de 2025.

---

## Estado actual del proyecto (a 30 Abril 2026)

```
Task A — COMPLETO ✅
Task B — INCOMPLETO ❌ (falta el script de modeling: 3B_modeling.py)
```

### Lo que está hecho:

| Script | Qué hace | Estado |
|--------|----------|--------|
| `dataExploration.py` | Visualizaciones exploratorias del dataset | ✅ Hecho |
| `taskA/1A_dataIngestion.py` | Lee JSONs → consolidated_data.parquet | ✅ Hecho |
| `taskA/2A_preprocessing.py` | Extrae drug/reaction baskets → parquet | ✅ Hecho |
| `taskA/3A_Encoding.py` | One-hot encoding → matrices binarias | ✅ Hecho |
| `taskA/4A_APriori.py` | Algoritmo Apriori → reglas CSV | ✅ Hecho |
| `taskA/5A_FPGrowth.py` | Algoritmo FP-Growth → reglas CSV | ✅ Hecho |
| `taskA/6A_FilterTrueInteractions.py` | Filtra ruido → reglas reales | ✅ Hecho |
| `taskA/7A_AnalysisPlots.py` | Grafos de red NetworkX → PNGs | ✅ Hecho |
| `taskA/8A_GenerateMetricsTable.py` | Tabla de métricas comparativa | ✅ Hecho |
| `taskB/1B_dataIngestion.py` | Lee ZIP → consolidated_data.parquet | ✅ Hecho |
| `taskB/2B_preprocessing.py` | Feature engineering → task_b_features.parquet | ✅ Hecho |
| `taskB/3B_modeling.py` | **CLASIFICADORES ML + EVALUACIÓN** | ❌ **FALTA** |

### Lo que falta para terminar:

1. **CREAR** `taskB/3B_modeling.py` con los 3 clasificadores (Logistic Regression, Random Forest, SVM)
2. **EJECUTAR** todo el pipeline en el servidor remoto (NVIDIA 3060)
3. **OBTENER** las métricas finales y plots de Task B

---

## Estructura de carpetas

```
DrugInteractionsRisks/
├── dataExploration.py          # EDA plots
├── requirements.txt
├── readme.md                   # README principal del proyecto
├── fields.pdf                  # Documentación de campos FAERS
├── plots/                      # Plots del EDA
│   ├── seriousness_and_sex.png
│   ├── severity_indicators_pie.png
│   ├── top_active_substances.png
│   ├── top_drug_names.png
│   ├── dist_drug_names_per_report.png
│   └── dist_active_substances_per_report.png
├── taskA/
│   ├── 1A_dataIngestion.py
│   ├── 2A_preprocessing.py
│   ├── 3A_Encoding.py
│   ├── 4A_APriori.py
│   ├── 5A_FPGrowth.py
│   ├── 6A_FilterTrueInteractions.py
│   ├── 7A_AnalysisPlots.py
│   ├── 8A_GenerateMetricsTable.py
│   ├── active_substances_encoded.parquet    ← ya generado ✅
│   ├── drug_names_encoded.parquet           ← ya generado ✅
│   ├── TaskAPlots/                          ← plots ya generados ✅
│   ├── association_rules/                   ← generado al correr 4A/5A (git-ignored)
│   ├── filtered_association_rules/          ← generado al correr 6A (git-ignored)
│   └── evaluation/                         ← generado al correr 8A (git-ignored)
├── taskB/
│   ├── 1B_dataIngestion.py
│   ├── 1B_preprocessing.py    ← skeleton vacío, ignorar
│   ├── 2B_preprocessing.py
│   └── 3B_modeling.py         ← ❌ ESTE ES EL QUE FALTA
└── local-files/               ← esta carpeta, documentación local
```

---

## El dataset FAERS explicado en 2 minutos

Cuando alguien en EEUU tiene una reacción adversa a un medicamento, el médico (o el paciente) puede reportarlo a la FDA. Cada reporte contiene:
- Quién es el paciente (edad, sexo)
- Qué medicamentos tomaba
- Qué reacciones tuvo
- Si fue grave (murió, fue hospitalizado, etc.)

De ~69 campos disponibles, este proyecto usa SOLO 4:
1. `safetyreportid` → ID del reporte (como el ID de la transacción)
2. `patient.drug.activesubstance.activesubstancename` → nombre genérico del fármaco
3. `patient.reaction.reactionmeddrapt` → reacción MedDRA estandarizada
4. `patient.patientonsetage` / `patient.patientsex` → demografía del paciente

---

## Por qué importa este proyecto

Las interacciones fármaco-fármaco (DDIs - Drug-Drug Interactions) son una causa importante de hospitalizaciones y muertes prevenibles. Con este pipeline automatizado se pueden detectar señales de seguridad nuevas directamente de datos del mundo real, sin necesidad de ensayos clínicos costosos.
