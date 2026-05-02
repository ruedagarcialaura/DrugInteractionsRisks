# TASK A: Association Rule Mining Pipeline
## Descubrimiento de Drug-Drug Interactions

---

## ¿Qué hace Task A?

Task A convierte los reportes FAERS (listas de fármacos + reacciones) en **reglas de asociación estadísticas**:

```
{DOXORUBICIN, CYCLOPHOSPHAMIDE} → {NAUSEA}
  antecedente (2 fármacos)         consecuente (reacción)
  support=0.003, confidence=0.85, lift=45.2
```

Esto significa: "Cuando un paciente toma DOXORUBICIN + CYCLOPHOSPHAMIDE juntos, hay un 85% de probabilidad de que tenga NAUSEA, y esa combinación produce náuseas 45x más de lo esperado por azar".

---

## Las 3 métricas clave (memorizarlas para la presentación)

### Support (Soporte)
> "¿Con qué frecuencia aparece esta regla en el dataset?"

```
support({A,B} → {C}) = P(A ∩ B ∩ C) = casos con A+B+C / total reportes
```
- Si support = 0.0025 → aparece en el 0.25% de los 108k reportes = ~270 casos
- **Muy bajo** → podría ser ruido estadístico
- **Demasiado alto** → los fármacos son muy comunes, la regla es obvia

### Confidence (Confianza)
> "Dado que tiene estos fármacos, ¿qué probabilidad hay de la reacción?"

```
confidence({A,B} → {C}) = P(C | A,B) = support(A,B,C) / support(A,B)
```
- Mide la fiabilidad predictiva de la regla

### Lift (Elevación) — el más importante para DDI
> "¿La combinación causa más reacción de lo que causarían los fármacos por separado?"

```
lift({A,B} → {C}) = confidence({A,B} → {C}) / support(C)
```
- `lift = 1.0` → la combinación no añade riesgo extra (independientes)
- `lift > 1.0` → la combinación SÍ añade riesgo (interacción real)
- `lift = 45` → esta combinación produce esa reacción 45x más que por azar

---

## El pipeline paso a paso

### Paso 1: Data Ingestion (`1A_dataIngestion.py`)
**Input:** Carpeta con 9 archivos JSON (datos FAERS trimestrales)
**Output:** `consolidated_data.parquet` (~108,000 filas, 69 columnas)

Qué hace:
- Lee cada JSON con `json.load()`
- Extrae el array `results` de cada archivo
- Aplana la estructura nested con `pd.json_normalize()`
- Guarda en Parquet (más eficiente que CSV para datos nested)

Comprobación: Verifica que hay ≥100,000 registros.

---

### Paso 2: Preprocessing (`2A_preprocessing.py`)
**Input:** `consolidated_data.parquet`
**Output:** `task_a_transactions_drug_names.parquet` + `task_a_transactions_activesubstance.parquet`

Qué hace:
- `df.explode('patient.drug')` → convierte cada fármaco de una lista a una fila separada
- Extrae `medicinalproduct` (nombre comercial) y `activesubstancename` (nombre genérico)
- `df.explode('patient.reaction')` → mismo proceso con reacciones
- `groupby('safetyreportid')['drug_name'].apply(list)` → reagrupa: un reporte = una fila con lista de fármacos

**Formato transaccional resultante:**
```
safetyreportid | all_items
12345678       | ['METHOTREXATE', 'PREDNISONE', 'NAUSEA', 'FATIGUE']
12345679       | ['ADALIMUMAB', 'RHEUMATOID ARTHRITIS', 'INJECTION SITE PAIN']
```

Por qué 2 versiones (drug names vs active substances):
- Drug names: nombres comerciales (HUMIRA, ENBREL, etc.) → más fragmentados
- Active substances: genéricos (ADALIMUMAB) → consolida redundancias de marca

**CONCLUSIÓN del experimento:** Active substances dan mejores reglas (mismo lift, menos ruido).

---

### Paso 3: Encoding (`3A_Encoding.py`)
**Input:** Parquets transaccionales
**Output:** `active_substances_encoded.parquet` + `drug_names_encoded.parquet`

Qué hace:
- `TransactionEncoder` de mlxtend convierte las listas en matrices binarias one-hot:

```
         | METHOTREXATE | PREDNISONE | NAUSEA | ADALIMUMAB | ...
12345678 |      1       |     1      |   1    |     0      | ...
12345679 |      0       |     0      |   0    |     1      | ...
```

- Filas = reportes, Columnas = fármacos/reacciones únicos
- La matriz resultante tiene ~108k filas × miles de columnas (muy sparse)

Estos ficheros YA EXISTEN en el repo. ✅

---

### Paso 4a: Apriori (`4A_APriori.py`)
**Input:** `active_substances_encoded.parquet`
**Output:** `association_rules/association_rules_APRIORI_active_substances_0_0025.csv`

Parámetros clave:
- `--min_support 0.0025` → valor empíricamente validado
- `min_threshold=1.0` para lift → solo reglas con lift ≥ 1 (asociación positiva)

Limitación del Apriori: genera todos los itemsets frecuentes candidatos (costoso en memoria). A support=0.001 → MemoryError en casi todos los casos.

---

### Paso 4b: FP-Growth (`5A_FPGrowth.py`)
**Input:** `active_substances_encoded.parquet`
**Output:** `association_rules/association_rules_FP_GROWTH_active_substances_0_0025.csv`

Ventaja vs Apriori:
- Usa un árbol FP-Tree comprimido → mucho más eficiente en memoria
- Mismos resultados matemáticos que Apriori al mismo threshold

Resultado experimental: A support=0.0025, **ambos algoritmos producen exactamente las mismas 13 reglas**. Esto valida la correctitud del pipeline.

---

### Paso 5: Filtrado (`6A_FilterTrueInteractions.py`)
**Input:** CSV de reglas raw
**Output:** `filtered_association_rules/filtered_*.csv`

Problema que resuelve: Las reglas incluyen "ruido de indicación" (indication bias):
- Regla: `{METHOTREXATE} → {RHEUMATOID ARTHRITIS}` → esto NO es un efecto adverso, es la indicación terapéutica
- Regla: `{HUMIRA} → {RHEUMATOID ARTHRITIS}` → igual, sesgo de indicación

Filtros aplicados:
1. Elimina consecuentes que son términos de "indicación" (RHEUMATOID ARTHRITIS, PSORIASIS, DIABETES...)
2. Solo mantiene reglas con ≥2 ítems en el antecedente (combinaciones de fármacos)
3. Ordena por lift descendente

Resultado: De ~miles de reglas raw → 13 reglas genuinas de DDI.

---

### Paso 6: Visualización (`7A_AnalysisPlots.py`)
**Output:** Grafos de red en `TaskAPlots/*.png`

Diseño visual:
- Nodos **teal/cyan** = fármacos o combinaciones
- Nodos **coral/rojo** = reacciones adversas
- Grosor de arista = fuerza de asociación (lift)
- Layout spring con k=2 para separar nodos

---

### Paso 7: Métricas (`8A_GenerateMetricsTable.py`)
**Output:** `evaluation/final_metrics_table.csv` + tabla Markdown

Métricas calculadas por configuración:
- Total Rules Found
- Max Lift
- Avg Confidence
- Unique Reactions

---

## Resultados de Task A (ya obtenidos)

| Model Configuration | Total Rules | Max Lift | Avg. Confidence | Unique Reactions |
|---------------------|-------------|----------|-----------------|------------------|
| Active Substances (Apriori) - 0.0025 | 13 | 223.45 | 0.6367 | 11 |
| Active Substances (FP-Growth) - 0.0025 | 13 | 223.45 | 0.6367 | 11 |
| Drug Names (Apriori) - 0.0025 | 13 | 224.13 | 0.6367 | 11 |
| Drug Names (Apriori) - 0.001 | 352,484 | 851.08 | 0.8300 | 6,878 |

**Conclusión clave:** Apriori = FP-Growth en términos de resultados. Active substances = mejor feature space. Support 0.0025 = punto óptimo.

---

## Cómo correr Task A (orden exacto)

```bash
# Desde la raíz del proyecto
python taskA/1A_dataIngestion.py <ruta/carpeta/jsons>
python taskA/2A_preprocessing.py
python taskA/3A_Encoding.py
python taskA/4A_APriori.py --file_path taskA/active_substances_encoded.parquet --min_support 0.0025
python taskA/5A_FPGrowth.py --file_path taskA/active_substances_encoded.parquet --min_support 0.0025
python taskA/6A_FilterTrueInteractions.py --input taskA/association_rules/association_rules_APRIORI_active_substances_0_0025.csv
python taskA/7A_AnalysisPlots.py --file taskA/filtered_association_rules/filtered_association_rules_APRIORI_active_substances_0_0025.csv
python taskA/8A_GenerateMetricsTable.py
```

NOTA: Los pasos 1-3 ya están ejecutados (los `.parquet` encoded existen). Solo hace falta correr 4-8 si los CSVs de reglas no existen.
