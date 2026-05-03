# TASK B — Historial de Iteraciones y Resultados Finales
## De cero modelos a AUC 0.805: todo lo que se hizo, por qué, y qué salió

---

## Punto de partida (antes de esta sesión)

Al comenzar esta sesión, el estado era:

```
Task A — COMPLETO ✅  (8 scripts + plots generados)
Task B — INCOMPLETO ❌
  ✅ 1B_dataIngestion.py    → consolidated_data.parquet
  ✅ 2B_preprocessing.py    → task_b_features.parquet (25 columnas iniciales)
  ❌ 3B_modeling.py         → NO EXISTÍA
```

La feature matrix inicial tenía **25 columnas**: demographics (4) + polypharmacy index (1) + top-15 drug flags (15) + interaction features de Task A (4) + target (1).

El objetivo: crear el pipeline de clasificación completo, ejecutarlo en el servidor remoto (NVIDIA RTX 3060), y obtener resultados.

---

## Iteración 0 — Crear 3B_modeling.py v1 (LR + RF + LinearSVM)

### Qué se hizo
Se creó `taskB/3B_modeling.py` con tres clasificadores baseline:
- **Logistic Regression**: lineal, requiere imputer + StandardScaler
- **Random Forest**: 100 árboles, class_weight='balanced'
- **Linear SVM**: CalibratedClassifierCV sobre LinearSVC

### El problema: LinearSVM se colapsó
Al ejecutar en el servidor, el SVM producía `F1_severe = 0.133` — prácticamente peor que azar para la clase severa. Investigación del root cause:

> CalibratedClassifierCV con cv=3 + LinearSVC con class_weight='balanced' en 108k muestras genera artifacts de calibración. La probabilidad calibrada colapsa a valores extremos, lo que hace que el threshold tuning no pueda encontrar nada útil.

**Resultado v1:**
| Model | AUC-ROC |
|---|---|
| Random Forest | ~0.75 |
| Logistic Regression | ~0.72 |
| Linear SVM | **COLAPSADO** (F1-sev=0.133) |

---

## Iteración 1 — 3B_modeling.py v2: Reemplazar SVM con HGBC

### Qué se cambió
LinearSVM → **HistGradientBoostingClassifier** (HGBC de sklearn).

**Por qué HGBC:**
- Maneja NaN de forma nativa (no necesita imputer)
- Gradient boosting sobre histogramas = más rápido que GBM clásico
- `class_weight` via `sample_weight` — no colapsa como el SVM calibrado
- Ya disponible en sklearn, sin dependencias extra

### Resultado v2:
| Model | AUC-ROC | F1-sev |
|---|---|---|
| HGBC | 0.784 | ~0.55 |
| Random Forest | 0.775 | ~0.54 |
| Logistic Regression | 0.743 | ~0.52 |

Mejora real, pero el AUC seguía siendo modesto. La feature matrix de 25 columnas tenía poca señal clínica.

---

## Iteración 2 — Feature Engineering: Reaction Features (MedDRA top-50)

### Diagnóstico
La feature matrix original solo tenía qué fármacos tomaba el paciente, pero **no qué reacciones había reportado**. Las reacciones MedDRA son una señal directísima de severidad.

### Qué se añadió en 2B_preprocessing.py
```
TOP_N_REACTIONS = 50   # top-50 términos MedDRA más frecuentes
```
Por cada reacción del top-50: una columna binaria `rxn_<nombre>` (0/1).

**Problema inmediato:** Leakage de datos detectado en feature importance.

> `rxn_hospitalisation` aparecía como la feature #2 más importante (10.7% de gain en XGBoost). Pero `HOSPITALISATION` es un término MedDRA que **codifica directamente** `seriousnesshospitalization=1`, que ES PARTE DEL TARGET. Esto es leakage puro.

### Términos excluidos del feature set (EXCLUDE_REACTIONS)
```python
# Death-related — codifican seriousnessdeath=1 (target leakage)
"DEATH", "COMPLETED SUICIDE", "SUDDEN DEATH", "HOMICIDE",
"ACCIDENTAL DEATH", "APPARENT DEATH",

# Hospitalization — codifica seriousnesshospitalization=1 (target leakage)
"HOSPITALISATION", "HOSPITALISATION EMERGENCY",

# Admin/pharmacovigilance — no son reacciones clínicas, son ruido operacional
"PRODUCT DOSE OMISSION ISSUE", "INCORRECT DOSE ADMINISTERED",
"OFF LABEL USE", "DRUG INEFFECTIVE", ...

# Non-specific
"NO ADVERSE EVENT", "ADVERSE DRUG REACTION", "DRUG INTERACTION", ...
```

**Evidencia del leakage:** AUC bajó de 0.805 → 0.796 al excluir `rxn_hospitalisation`. Eso confirma que la métrica anterior estaba inflada por datos filtrados del target.

La feature matrix pasó de 25 → ~70 columnas.

---

## Iteración 3 — GPU: XGBoost + CatBoost + Voting Ensemble

### Qué se cambió en 3B_modeling.py (v3)

Se reemplazó el pipeline completo:

| Antes (v2) | Después (v3) |
|---|---|
| Logistic Regression | Logistic Regression (sin cambios) |
| Random Forest | Random Forest (sin cambios) |
| HGBC (sklearn CPU) | **XGBoost GPU** (`device='cuda'`, `tree_method='hist'`) |
| — | **CatBoost GPU** (`task_type='GPU'`) |
| — | **Voting Ensemble** (soft voting, sin re-entrenamiento) |

### XGBoost GPU — parámetros baseline
```python
XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=neg/pos,   # = 2.77 en este dataset
    device='cuda',
    tree_method='hist',
)
```

### CatBoost GPU — parámetros baseline
```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    auto_class_weights='Balanced',
    task_type='GPU',
    devices='0',
)
```

### Voting Ensemble
Promedia `predict_proba` de los modelos ya entrenados (RF + XGB + CatBoost). Sin re-entrenamiento. El threshold se tuña en val set.

### Tiempo de ejecución GPU vs CPU
| Modelo | CPU (estimado) | GPU (real en RTX 3060) |
|---|---|---|
| Random Forest (300 trees) | ~4 min | 4.3s |
| XGBoost | ~20 min | **2.6s** |
| CatBoost | ~20 min | **2.5s** |

---

## Iteración 4 — Features adicionales: Reporter + Drug Characterization

### Reporter Qualification (`primarysource.qualification`)
FAERS codifica quién reportó el evento:
- `1` = Physician
- `2` = Pharmacist
- `3` = Other HCP
- `5` = Consumer

Se añaden 4 columnas binarias: `reporter_physician`, `reporter_pharmacist`, `reporter_other_hcp`, `reporter_consumer`.

**Por qué importa:** Los reportes de médicos tienden a ser más completos y a documentar más eventos life-threatening. Los reportes de consumers tienen más bias de autopercepción.

### Drug Characterization (`drugcharacterization`)
- `1` = Suspect drug (el sospechoso principal)
- `2` = Concomitant (tomado al mismo tiempo, no necesariamente causante)

Se añaden: `num_suspect_drugs`, `num_concomitant_drugs`.

**Por qué importa:** A más fármacos suspect, más compleja la imagen de evento adverso y mayor probabilidad de interacción polisustancia.

### Feature matrix final: 71 columnas

| Grupo | Columnas | Descripción |
|---|---|---|
| Demographics | 4 | age, is_male, is_female, sex_unknown |
| Polypharmacy | 1 | num_drugs_taken |
| Drug flags | 15 | takes_\<drug\> (top-15 active substances) |
| Drug characterization | 2 | num_suspect_drugs, num_concomitant_drugs |
| Reaction flags | 50 | rxn_\<meddrapt\> (top-50, excluyendo leakage) |
| Reporter | 4 | reporter_\<type\> |
| Interaction (Task A) | 4 | has_drug_drug_interaction, has_drug_reaction_rule, num_matching_rules, max_interaction_lift |
| **Target** | 1 | is_severe_outcome |

### Resultados v3 (clean, sin leakage):

| Model | AUC-ROC | Avg Precision | F1 (severe) | Recall (sev) | Prec. (sev) | Fit time |
|---|---|---|---|---|---|---|
| **Voting Ensemble** | **0.7962** | 0.6032 | 0.5750 | 0.6477 | 0.5170 | — |
| XGBoost (GPU) | 0.7949 | 0.5985 | 0.5737 | 0.6847 | 0.4937 | 2.6s |
| CatBoost (GPU) | 0.7936 | 0.5958 | 0.5739 | 0.6411 | 0.5195 | 2.5s |
| Random Forest | 0.7916 | 0.5962 | 0.5697 | 0.6442 | 0.5106 | 4.3s |
| Logistic Regression | 0.7519 | 0.5208 | 0.5332 | 0.6409 | 0.4564 | 1.3s |

**Progresión de AUC desde el inicio:**
```
v1 (LR+RF+SVM)            →  ~0.750  (SVM colapsado)
v2 (HGBC)                 →  ~0.784
v3 clean (XGB+CB+Ensemble) →  0.796
```

---

## Iteración 5 — Optuna Hyperparameter Tuning (v4)

### Motivación
Los hiperparámetros de XGBoost y CatBoost en v3 eran hardcoded (defaults razonables, no optimizados). Con GPU a 2-3s por trial, explorar el espacio de hiperparámetros es factible.

### Estrategia: early stopping ceiling
En lugar de incluir `n_estimators` en el search space (lo que forzaría re-entrenamiento completo en cada trial), se usa:
- `n_estimators = 3000` como techo fijo
- `early_stopping_rounds = 50` para parar automáticamente cuando el val AUC no mejora

Esto permite que cada trial encuentre el número óptimo de árboles sin desperdicio computacional.

### Script: `taskB/4B_optuna_tuning.py`

#### XGBoost — 100 trials, TPE sampler, GPU

| Parámetro | Rango | Distribución |
|---|---|---|
| `learning_rate` | [0.005, 0.3] | log-uniform |
| `max_depth` | [3, 10] | int |
| `subsample` | [0.5, 1.0] | uniform |
| `colsample_bytree` | [0.3, 1.0] | uniform |
| `min_child_weight` | [1, 20] | int |
| `gamma` | [0.0, 5.0] | uniform |
| `reg_alpha` | [1e-8, 10] | log-uniform (L1, nuevo) |
| `reg_lambda` | [1e-8, 10] | log-uniform (L2, nuevo) |

Fijo: `scale_pos_weight=2.77`, `device='cuda'`, `tree_method='hist'`

#### CatBoost — 50 trials, TPE sampler, GPU

| Parámetro | Rango | Distribución |
|---|---|---|
| `learning_rate` | [0.01, 0.3] | log-uniform |
| `depth` | [4, 10] | int |
| `l2_leaf_reg` | [1, 10] | uniform |
| `bagging_temperature` | [0, 10] | uniform |
| `random_strength` | [0, 10] | uniform |
| `min_data_in_leaf` | [1, 50] | int |

Fijo: `auto_class_weights='Balanced'`, `task_type='GPU'`, `iterations=2000`

### Mejores parámetros encontrados

**XGBoost** (trial #76, best val AUC = 0.8039):
```json
{
  "learning_rate": 0.02394,
  "max_depth": 10,
  "subsample": 0.6022,
  "colsample_bytree": 0.9535,
  "min_child_weight": 1,
  "gamma": 1.703,
  "reg_alpha": 6.43e-4,
  "reg_lambda": 7.14e-3
}
```

**CatBoost** (best val AUC = 0.8033):
```json
{
  "learning_rate": 0.03156,
  "depth": 9,
  "l2_leaf_reg": 9.439,
  "bagging_temperature": 0.773,
  "random_strength": 5.424,
  "min_data_in_leaf": 8
}
```

### Tiempo de ejecución del tuning
| Fase | Tiempo real en RTX 3060 |
|---|---|
| XGBoost study (100 trials) | ~5 min |
| CatBoost study (50 trials) | ~12 min |
| Re-entrenamiento + evaluación | ~1 min |
| **Total** | **~18 min** |

### Resultados finales tuned:

| Model | AUC-ROC | Avg Precision | F1 (severe) | Recall (sev) | Prec. (sev) |
|---|---|---|---|---|---|
| **Tuned Ensemble** | **0.8054** | 0.6255 | 0.5860 | 0.6512 | 0.5327 |
| XGBoost Tuned (GPU) | 0.8036 | 0.6249 | 0.5843 | 0.6751 | 0.5150 |
| CatBoost Tuned (GPU) | 0.8035 | 0.6186 | 0.5835 | 0.6618 | 0.5217 |

### Comparación baseline vs tuned:

| Modelo | Baseline AUC | Tuned AUC | Delta |
|---|---|---|---|
| XGBoost (GPU) | 0.7949 | 0.8036 | **+0.0087** |
| CatBoost (GPU) | 0.7936 | 0.8035 | **+0.0099** |
| Ensemble | 0.7962 | **0.8054** | **+0.0092** |

---

## Progresión total de AUC a lo largo del proyecto

```
Inicio (LR+RF, features básicas)     →  ~0.750
+ HGBC (sustituto SVM)               →  ~0.784
+ XGBoost + CatBoost GPU             →  ~0.805  (leakage)
  → Detectar y corregir leakage      →   0.796  (limpio)
+ Reaction features (MedDRA top-50)  →  0.796   (base clean)
+ Reporter + drug characterization   →  0.796   (marginal)
+ Optuna tuning (100+50 trials GPU)  →  0.805   ← ESTADO ACTUAL
```

> El salto más grande fue detectar el leakage de `rxn_hospitalisation` — el modelo parecía en 0.805 pero en realidad estaba haciendo trampa. El 0.805 final es legítimo.

---

## Ficheros generados en esta sesión

### Scripts nuevos
| Fichero | Qué hace |
|---|---|
| `taskB/3B_modeling.py` | Pipeline completo: LR + RF + XGBoost GPU + CatBoost GPU + Voting Ensemble |
| `taskB/4B_optuna_tuning.py` | Optuna TPE search: XGBoost 100 trials + CatBoost 50 trials, GPU |

### Data files (git-ignored, existen en servidor y localmente)
| Fichero | Qué contiene |
|---|---|
| `taskB/task_b_features.parquet` | Feature matrix final (108000 × 71) |
| `taskB/task_b_evaluation.csv` | Métricas baseline de los 5 modelos |
| `taskB/task_b_tuned_evaluation.csv` | Métricas de los 3 modelos tuned |
| `taskB/optuna_best_params.json` | Mejores hiperparámetros (commiteado a git) |
| `taskB/results_history.csv` | Log histórico de todas las ejecuciones con timestamp |

### Plots generados (en `taskB/TaskBPlots/`)
| Plot | Qué muestra |
|---|---|
| `confusion_matrices.png` | Matrices de confusión normalizadas de los 5 modelos |
| `roc_curves.png` | Curvas ROC baseline de los 5 modelos |
| `roc_curves_tuned.png` | Curvas ROC de los modelos tuned |
| `pr_curves.png` | Curvas Precision-Recall (informativas con clases desbalanceadas) |
| `feature_importance_rf.png` | Top-25 features según Random Forest |
| `feature_importance_xgb.png` | Top-25 features según XGBoost baseline |
| `feature_importance_xgb_tuned.png` | Top-25 features según XGBoost tuned |
| `feature_importance_catboost.png` | Top-25 features según CatBoost |
| `lr_coefficients.png` | Coeficientes de Logistic Regression (rojo=sube riesgo) |
| `eda_distributions.png` | Distribuciones EDA de la feature matrix |
| `optuna_history_xgb.png` | Convergencia Optuna — XGBoost (100 trials) |
| `optuna_history_catboost.png` | Convergencia Optuna — CatBoost (50 trials) |

### Notebook
| Fichero | Estado |
|---|---|
| `TaskB_notebook.ipynb` | Notebook completo end-to-end con 41 celdas |
| `TaskB_notebook_EXECUTED.ipynb` | Versión ejecutada (en servidor, con outputs embebidos) |

**FASEs del notebook:**
```
FASE 0: Dependency check
FASE 1: Generar filtered rules de Task A (puente Task A→B)
FASE 2: Data ingestion desde ZIP
FASE 3: Feature engineering (llama a 2B_preprocessing.py)
FASE 4: EDA de la feature matrix
FASE 5: Train/Val/Test split (70/10/20, stratificado)
FASE 6: Entrenamiento de 5 modelos con threshold tuning en val set
FASE 7: Evaluation plots (ROC, PR, confusion, feature importance)
FASE 8: Tabla comparativa con highlight del mejor
FASE 8.5: Optuna tuning (llama a 4B_optuna_tuning.py) ← NUEVO
FASE 9: Conclusiones
```

---

## Decisiones de diseño importantes

### Train / Val / Test: 70 / 10 / 20
El split de 3 partes es crítico:
- **Train (70%)**: para ajustar los parámetros del modelo
- **Val (10%)**: para el threshold tuning (encontrar el umbral de clasificación óptimo sin tocar el test set)
- **Test (20%)**: métricas finales, nunca visto durante entrenamiento NI threshold tuning

Si se usara solo train/test y se afinase el threshold en test, las métricas estarían infladas.

### Threshold tuning en val set
En lugar de usar threshold=0.5 por defecto, se hace un grid search en [0.10, 0.70] (61 puntos) sobre el val set, maximizando F1 de la clase severa (pos_label=1). El threshold óptimo varía por modelo (~0.51–0.56).

### Scale_pos_weight para XGBoost
```python
spw = neg_count / pos_count   # = 79317 / 28683 ≈ 2.77
```
Equivalente a `class_weight='balanced'` pero específico de XGBoost. Hace que cada error en un caso severo cuente 2.77× más en la función de pérdida.

### No scaling global
El `StandardScaler` va DENTRO del Pipeline de sklearn, ajustado solo con `X_train`. Si se aplica globalmente antes del split, hay data leakage porque los parámetros del scaler (media/std) están calculados con datos del test set.

### Imputer pre-Optuna
Para el tuning de Optuna, se pre-imputa con `SimpleImputer(strategy='median')` fuera del pipeline (fit en train, transform en val/test). Esto es equivalente a tenerlo dentro del pipeline pero evita re-imputar en cada trial, reduciendo el overhead.

---

## Features más importantes (según XGBoost Tuned)

Las features con mayor ganancia de información (aprox.):

1. `has_drug_reaction_rule` — el puente Task A→B más fuerte
2. `num_reactions` — número de reacciones reportadas
3. `patientonsetage` — edad del paciente
4. `num_drugs_taken` — polifarmacia
5. `num_suspect_drugs` — número de fármacos sospechosos
6. `reporter_physician` — reportado por médico
7. `rxn_nausea`, `rxn_pyrexia`, `rxn_fatigue` — reacciones frecuentes
8. `takes_prednisone`, `takes_methotrexate` — fármacos de alto riesgo

`rxn_hospitalisation` fue eliminada porque era leakage (codificaba directamente el target).

---

## Estado final del proyecto

```
Task A — COMPLETO ✅ (8/8 scripts + plots)
Task B — COMPLETO ✅ (4/4 scripts + plots + Optuna tuning)

Mejor modelo: Tuned Ensemble (XGBoost + CatBoost, Optuna-tuned)
AUC-ROC:      0.8054  (vs 0.5 baseline aleatorio)
F1-severe:    0.586
Recall-sev:   0.651
```

### Scripts de Task B en orden de pipeline:
| Script | Estado | Qué genera |
|---|---|---|
| `taskB/1B_dataIngestion.py` | ✅ | consolidated_data.parquet |
| `taskB/2B_preprocessing.py` | ✅ | task_b_features.parquet (108k × 71) |
| `taskB/3B_modeling.py` | ✅ | 5 modelos + 8 plots + task_b_evaluation.csv |
| `taskB/4B_optuna_tuning.py` | ✅ | 3 modelos tuned + 4 plots + optuna_best_params.json |

---

## Para la presentación — puntos clave de Task B

### El número más importante
**AUC-ROC = 0.805** → el modelo es capaz de discriminar casos severos con un 80.5% de capacidad discriminativa (vs 50% de un clasificador aleatorio).

### Por qué AUC y no accuracy
Con 73.4% de clase 0, un modelo que siempre prediga "no severo" tiene 73.4% accuracy. AUC mide la capacidad de ranking: a cualquier threshold, ¿puede el modelo separar severos de no-severos?

### El puente Task A → Task B
`has_drug_reaction_rule` (feature de Task A) es la feature individual más predictiva. Las reglas de asociación de Task A no solo sirven para describir interacciones — sirven como señal predictiva en el clasificador de Task B. Esto demuestra que el pipeline KDD es coherente end-to-end.

### Recall vs Precision en contexto médico
En un contexto clínico real, **Recall (severo) = 0.65** es más importante que Precision. Un False Negative (predecir "no severo" cuando sí lo es) puede significar no intervenir a tiempo. Un False Positive (sobreestimar severidad) es conservador pero menos peligroso. El threshold de ~0.53 refleja esta priorización.

### Optuna como argumento metodológico
Si alguien pregunta "¿no habría que hacer GridSearch?": "Lo hicimos — pero con Optuna (Bayesian optimization). En lugar de explorar exhaustivamente una grid de parámetros (ineficiente), TPE aprende de los trials anteriores para dirigir la búsqueda hacia regiones prometedoras. 150 trials totales en ~18 minutos en GPU."
