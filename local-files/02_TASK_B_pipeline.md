# TASK B: Severity Prediction Pipeline
## Clasificación Binaria de Severidad de Eventos Adversos

---

## ¿Qué hace Task B?

Task B entrena modelos de clasificación que, dado un nuevo reporte FAERS (paciente + medicamentos), predice si el resultado será **severo** o **no severo**.

```
INPUT:  [edad=67, sexo=mujer, toma_methotrexate=1, toma_prednisone=1, 
         num_farmacos=4, has_drug_reaction_rule=1]
         
OUTPUT: is_severe_outcome = 1  (SEVERO — probabilidad 78%)
```

**Target binario:**
- `1 = Severo` → el reporte incluye muerte, hospitalización, o evento life-threatening (26.6%)
- `0 = No severo` → otros outcomes (73.4%)

---

## Feature Matrix: Las 25 columnas del dataset de Task B

Producida por `2B_preprocessing.py` → `taskB/task_b_features.parquet` (108,000 filas × 25 columnas)

### Grupo A: Demographics (4 columnas)
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `patientonsetage` | float | Edad del paciente en años. ~40% NaN (preservado intencionalmente) |
| `is_male` | int8 | 1 si es hombre |
| `is_female` | int8 | 1 si es mujer |
| `sex_unknown` | int8 | 1 si no se reportó sexo |

**Por qué one-hot encoding para sexo:** FAERS usa códigos 1/2/0 que son ordinales artificiales. One-hot elimina esa relación falsa.

**Por qué preservar NaN en edad:** Aplicar imputation ANTES del train/test split sería **data leakage** (filtrar información del test set al entrenamiento). La imputation va DENTRO del pipeline de cada modelo.

### Grupo B: Polypharmacy Index (1 columna)
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `num_drugs_taken` | int | Número de active substances únicas en el reporte |

**Por qué importa:** La polifarmacia es el principal factor de riesgo de DDI. A más fármacos, mayor probabilidad de interacción no deseada.

### Grupo C: Top-15 Active Substance Flags (15 columnas)
Variables binarias 0/1 para los 15 fármacos más frecuentes del dataset:
- `takes_tirzepatide` (Ozempic-like, diabetes tipo 2)
- `takes_dupilumab` (eczema, asma)
- `takes_prednisone` (corticoide, antiinflamatorio)
- `takes_methotrexate` (artritis reumatoide, psoriasis)
- `takes_omalizumab` (asma alérgica)
- `takes_adalimumab` (HUMIRA, artritis)
- `takes_acetaminophen` (paracetamol, dolor)
- `takes_infliximab_dyyb` (biosimilar de REMICADE)
- `takes_infliximab` (REMICADE, artritis)
- `takes_rituximab` (linfoma, artritis)
- `takes_leuprolide_acetate` (cáncer próstata/mama)
- `takes_tocilizumab` (artritis reumatoide)
- `takes_vedolizumab` (enfermedad inflamatoria intestinal)
- `takes_amlodipine_besylate` (hipertensión)
- `takes_cetirizine_hydrochloride` (antihistamínico)

**Por qué estos 15:** Son los más frecuentes → mayor exposición poblacional → más señal estadística.

### Grupo D: Interaction Risk Features (4 columnas) — venidas de Task A
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `has_drug_drug_interaction` | int8 | 1 si el reporte contiene una combinación que matchea una regla drug-drug con lift≥2.0 |
| `has_drug_reaction_rule` | int8 | 1 si los fármacos del reporte matchean una regla cuyo consecuente es una reacción MedDRA |
| `num_matching_rules` | int | Número de reglas de Task A que se activan para este reporte |
| `max_interaction_lift` | float | Mayor lift entre todas las reglas activadas (0.0 si ninguna) |

**Esto es el puente entre Task A y Task B.** Task A descubrió las reglas; Task B las usa como features.

`has_drug_reaction_rule` es la feature más fuerte porque conecta directamente combinaciones de fármacos con reacciones MedDRA adversas documentadas.

### Grupo E: Target Variable (1 columna)
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `is_severe_outcome` | int8 | 1 = severo (muerte/hospitalización/life-threatening), 0 = no severo |

**Class balance:**
- Severo (1): 28,683 reportes (26.6%)
- No severo (0): 79,317 reportes (73.4%)
- Desbalance moderado → usar `class_weight='balanced'` y evaluar con AUC-ROC y F1

---

## Los 3 clasificadores de Task B

### 1. Logistic Regression
**Qué es:** Modelo lineal. Aprende pesos para cada feature y calcula una probabilidad con la función sigmoide.
**Requiere:** Imputation + StandardScaler (sensible a la escala de las variables)
**Ventaja:** Interpretable (los coeficientes indican qué features importan más)
**Desventaja:** Solo captura relaciones lineales

Pipeline:
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(class_weight='balanced', max_iter=1000))
])
```

### 2. Random Forest
**Qué es:** Ensemble de árboles de decisión. Cada árbol vota y la mayoría gana.
**Requiere:** Solo imputation (no necesita scaling — los árboles son invariantes a la escala)
**Ventaja:** Maneja no-linealidades, feature importance, robusto a outliers
**Desventaja:** Menos interpretable que LR, más costoso computacionalmente

Pipeline:
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model',   RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])
```

### 3. SVM (Support Vector Machine)
**Qué es:** Encuentra el hiperplano óptimo que separa las clases con máximo margen.
**Requiere:** Imputation + StandardScaler (el más sensible a la escala de todos)
**Ventaja:** Muy potente en espacios de alta dimensión, kernel RBF captura no-linealidades
**Desventaja:** Muy lento en datasets grandes (108k × 25), no produce probabilidades por defecto

Pipeline:
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   SVC(kernel='rbf', class_weight='balanced', probability=True))
])
```

⚠️ **SVM en 108k registros puede tardar horas.** Considerar `LinearSVC` o usar subset para probar.

---

## Métricas de evaluación

**Por qué NO usar accuracy:**
- Con 73.4% de clase 0, un modelo que siempre predice "no severo" tiene 73.4% accuracy
- Accuracy es misleading con clases desbalanceadas

**Métricas correctas:**

### AUC-ROC (Area Under the ROC Curve)
- Mide la capacidad discriminativa del modelo a todos los thresholds
- 0.5 = aleatorio, 1.0 = perfecto
- **Objetivo realista para este problema: AUC ≥ 0.75**

### F1-Score (weighted)
- Harmonic mean de Precision y Recall
- Weighted average pondera por el número de muestras de cada clase
- **Objetivo realista: F1 ≥ 0.70**

### Classification Report
- Precision, Recall, F1 por clase
- Muestra el comportamiento diferenciado en clase 0 y clase 1

### Confusion Matrix
- Visualiza True Positives, False Positives, True Negatives, False Negatives
- Los False Negatives (severo predicho como no-severo) son los más peligrosos médicamente

---

## Scripts de Task B

### `1B_dataIngestion.py` — COMPLETO ✅
Lee el ZIP con los 9 JSONs sin extraerlos a disco → `consolidated_data.parquet`

### `2B_preprocessing.py` — COMPLETO ✅
Feature engineering completo → `taskB/task_b_features.parquet`

### `1B_preprocessing.py` — IGNORAR
Es un skeleton vacío que sirve de documentación del handoff. Supersedido por `2B_preprocessing.py`.

### `3B_modeling.py` — ❌ FALTA (HAY QUE CREARLO Y EJECUTARLO)
Ver fichero `03_LO_QUE_FALTA.md` para el plan de acción.

---

## Cómo correr Task B

```bash
# Paso 1: Ingestion (si no existe consolidated_data.parquet)
python taskB/1B_dataIngestion.py "taskB/9 json files - no tocar.zip"

# Paso 2: Feature Engineering (si no existe task_b_features.parquet)
python taskB/2B_preprocessing.py

# Paso 3: Modeling (NECESITA SER CREADO PRIMERO)
python taskB/3B_modeling.py
```

---

## Notas arquitectónicas importantes

### Por qué no hay scaling global
Aplicar `StandardScaler` al dataset completo antes de split = **data leakage**. Los parámetros del scaler (media y std) estarían calculados con datos del test set, lo que infla artificialmente las métricas. La solución es encapsular `StandardScaler` DENTRO de `sklearn.Pipeline`, donde se ajusta SOLO con X_train.

### NaN en edad y por qué está bien
40% de los reportes FAERS no incluyen edad. FAERS no la exige. Opciones:
- Árboles (Random Forest): pueden manejar NaN si se usa `enable_iterative_imputer` o `SimpleImputer` dentro del pipeline
- Modelos lineales/SVM: requieren imputation explícita con la mediana

No se imputa globalmente porque la mediana global incluiría info del test set.
