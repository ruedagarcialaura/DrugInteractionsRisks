# CONCEPTOS CLAVE PARA LA PRESENTACIÓN
## Glosario técnico explicado en español

---

## Conceptos de Association Rule Mining (Task A)

### FAERS (FDA Adverse Event Reporting System)
Base de datos de la FDA americana donde médicos, pacientes y fabricantes reportan voluntariamente eventos adversos post-mercado. No es exhaustiva (subregistro estimado del 90%), pero es el mayor dataset de pharmacovigilance del mundo.

### KDD (Knowledge Discovery in Databases)
El proceso completo desde datos crudos hasta conocimiento accionable:
`Raw Data → Preprocessing → Transformation → Data Mining → Interpretation`
Este proyecto implementa un KDD pipeline completo.

### Transactional Format (Formato Transaccional)
Representación de los datos donde cada "transacción" (reporte FAERS) es una "cesta de la compra" (basket) con ítems (fármacos + reacciones). El mismo formato que usa Amazon para "quien compró A también compró B".

```
Transaction/Basket:
  Report #12345: {METHOTREXATE, PREDNISONE, NAUSEA, FATIGUE, ANEMIA}
```

### Association Rule (Regla de Asociación)
Una regla de la forma `{antecedentes} → {consecuente}` con métricas estadísticas.
Ejemplo médico: `{DOXORUBICIN, CYCLOPHOSPHAMIDE} → {FEBRILE NEUTROPENIA}`

### Support (Soporte)
Frecuencia de la regla en el dataset. Si support = 0.003, la combinación aparece en el 0.3% de los 108k reportes (~324 casos).

### Confidence (Confianza)
Dado que el paciente toma los fármacos del antecedente, ¿qué probabilidad hay de la reacción? Es un P(reacción | fármacos).

### Lift (Elevación) — el más importante para DDI
Mide si la asociación es "sorprendente" o "esperada por azar".
- Lift = 1 → la combinación no añade riesgo (independientes)
- Lift > 1 → la combinación es peligrosa (dependencia positiva)
- Lift = 223 → la combinación produce esa reacción 223× más de lo esperado por azar

### Indication Bias (Sesgo de Indicación)
Problema donde el algoritmo detecta la relación fármaco→enfermedad porque se usa para tratarla, no porque la cause:
- `{METHOTREXATE} → {RHEUMATOID ARTHRITIS}` → bias, es su indicación
- `{METHOTREXATE, ADALIMUMAB} → {HEPATOTOXICITY}` → señal real de DDI

Se filtra en el paso 6A.

### Apriori Algorithm
Genera frecuent itemsets candidatos de forma iterativa (nivel a nivel). Limitación: genera todos los subconjuntos posibles → costoso en memoria con muchos ítems o threshold bajo.

### FP-Growth (Frequent Pattern Growth)
Alternativa más eficiente al Apriori. Comprime la base de datos en un FP-Tree (árbol de patrones frecuentes) y mina el árbol directamente. Mismos resultados matemáticos, menor uso de memoria.

---

## Conceptos de Classification (Task B)

### Binary Classification (Clasificación Binaria)
Predecir entre 2 clases: severo (1) o no severo (0). El modelo aprende de casos históricos etiquetados y generaliza a nuevos casos.

### Feature Engineering (Ingeniería de Features)
El proceso de transformar datos raw en variables numéricas útiles para los modelos ML. En Task B:
- `patient.patientsex` → 3 columnas binarias (one-hot encoding)
- `patient.drug` (lista) → `num_drugs_taken` (conteo)
- Reglas de Task A → 4 features de riesgo de interacción

### Data Leakage (Fuga de Datos)
Error crítico donde información del test set "contamina" el entrenamiento, inflando artificialmente las métricas. En Task B se evita poniendo `StandardScaler` y `SimpleImputer` DENTRO del `sklearn.Pipeline`.

### Class Imbalance (Desbalance de Clases)
El dataset tiene más casos "no severos" (73.4%) que "severos" (26.6%). Si se ignora, el modelo aprende a predecir siempre "no severo" y tiene alta accuracy pero pésimo recall en la clase positiva.

Solución: `class_weight='balanced'` → el algoritmo pondera más los errores en la clase minoritaria.

### AUC-ROC (Area Under the ROC Curve)
Métrica robusta ante desbalance de clases. La curva ROC grafica Sensitivity (True Positive Rate) vs 1-Specificity (False Positive Rate) a todos los thresholds. El área bajo esa curva (AUC) mide la capacidad discriminativa general.
- 0.5 = aleatorio (baseline)
- 0.7-0.8 = bueno
- >0.8 = excelente

### F1-Score
`F1 = 2 * (Precision * Recall) / (Precision + Recall)`
Combina Precision y Recall en una sola métrica. Más informativo que accuracy cuando hay desbalance.

### sklearn.Pipeline
Encadena pasos de preprocesado + modelo en un solo objeto. Garantiza que el fitting se hace solo sobre training data, previniendo data leakage automáticamente.

```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression()),
])
pipe.fit(X_train, y_train)  # imputer y scaler se ajustan SOLO con train
pipe.predict(X_test)        # aplica transforms ajustados sobre test
```

### Polypharmacy (Polifarmacia)
Uso simultáneo de múltiples medicamentos. En FAERS, `num_drugs_taken` captura esto. La polifarmacia es el principal factor de riesgo para DDIs: más fármacos = más oportunidades de interacción.

### MedDRA (Medical Dictionary for Regulatory Activities)
Vocabulario médico internacional estandarizado que usa la FDA. Las reacciones adversas en FAERS están codificadas como términos MedDRA (ej: "FEBRILE NEUTROPENIA", "HEPATOTOXICITY"). Esto permite comparar señales entre países y sistemas.

---

## Conceptos de los algoritmos ML en detalle

### Logistic Regression
**Cómo funciona:** Aprende pesos (coeficientes) para cada feature y calcula `P(severo) = sigmoid(w₀ + w₁*edad + w₂*num_farmacos + ...)`. El límite de decisión es lineal.

**Interpretabilidad:** Los coeficientes dicen cuánto contribuye cada feature. Un coeficiente positivo alto en `has_drug_reaction_rule` diría "esta feature aumenta mucho la probabilidad de evento severo".

**Limitaciones:** No captura interacciones no lineales entre features (ej: "el riesgo de edad alta + muchos fármacos no es simplemente la suma de ambos").

### Random Forest
**Cómo funciona:** Entrena N árboles de decisión (N=100 por defecto), cada uno con un subset aleatorio de datos y features (bagging + feature randomness). La predicción final es el voto mayoritario de todos los árboles.

**Por qué es robusto:** Los errores individuales de cada árbol se cancelan entre sí (ensemble). El ruido en los datos afecta a cada árbol de forma diferente.

**Feature Importance:** Mide cuánto disminuye la impureza de los nodos (Gini o entropy) cuando se usa cada feature. Permite entender qué variables el modelo considera más útiles.

### SVM (Support Vector Machine)
**Cómo funciona:** Encuentra el hiperplano que maximiza el margen entre las dos clases. Con kernel RBF, transforma implícitamente los datos a un espacio de mayor dimensión donde sí son linealmente separables.

**Support vectors:** Los puntos más cercanos al hiperplano de decisión. Solo estos importan para definir el límite.

**Problema con 108k muestras:** La complejidad de entrenamiento es O(n²) a O(n³). Para escalabilidad se usa `LinearSVC` (kernel lineal, mucho más rápido) o `SGDClassifier`.

---

## Por qué Task A alimenta a Task B

Esta es la innovación conceptual del proyecto:

```
Task A descubre:
  {METHOTREXATE, ADALIMUMAB} → {HEPATOTOXICITY} (lift=45)

Task B recibe para un paciente:
  drugs = {METHOTREXATE, ADALIMUMAB, PREDNISONE}
  
2B_preprocessing.py computa:
  has_drug_reaction_rule = 1  (la regla de Task A se activa)
  max_interaction_lift = 45.0
  
3B_modeling.py puede aprender:
  "Cuando has_drug_reaction_rule=1 y max_interaction_lift es alto,
   la probabilidad de outcome severo aumenta"
```

Sin Task A, Task B solo tendría información demográfica y polifarmacia. Con Task A, tiene conocimiento farmacológico destilado de 108k casos históricos.
