# PLAN FINAL DE EJECUCIÓN
## Lo que hay, lo que falta, y el orden exacto para completar Task B

---

## REVISIÓN CRÍTICA DEL CÓDIGO — Bugs encontrados y arreglados

### Bug 1 (ARREGLADO): `filtered_association_rules/` no existe
`2B_preprocessing.py` necesita `taskA/filtered_*.csv` para construir las features de interacción (`has_drug_reaction_rule`, etc.). Sin ellas, esas 4 columnas serán todas **cero** — el puente Task A→Task B queda roto.

**Solución:** El notebook (`TaskB_notebook.ipynb`) regenera automáticamente las reglas filtradas usando los encoded parquets que **sí están en el repo**, corriendo `4A_APriori.py` + `6A_FilterTrueInteractions.py` antes de llamar a `2B_preprocessing.py`.

### Bug 2 (ARREGLADO): `quick_mode` sampling en `3B_modeling.py`
El código original usaba `.groupby().apply()` con MultiIndex — deprecado en pandas 2.x y con comportamiento inconsistente. Reemplazado por `StratifiedShuffleSplit`.

### Bug 3 (ARREGLADO): Import no usado en `3B_modeling.py`
`import matplotlib.gridspec as gridspec` estaba importado pero no se usaba. Eliminado.

---

## Estado actual de los ficheros DATA (no están en git — git-ignored)

| Fichero | ¿Existe? | Cómo generarlo |
|---------|----------|----------------|
| `taskA/active_substances_encoded.parquet` | ✅ SÍ (en repo) | Ya está |
| `taskA/drug_names_encoded.parquet` | ✅ SÍ (en repo) | Ya está |
| `taskA/association_rules/*.csv` | ❌ NO | 4A_APriori.py (arranca de encoded parquet) |
| `taskA/filtered_association_rules/*.csv` | ❌ NO | 6A_FilterTrueInteractions.py |
| `consolidated_data.parquet` | ❌ NO | 1B_dataIngestion.py (necesita el ZIP) |
| `taskB/task_b_features.parquet` | ❌ NO | 2B_preprocessing.py (necesita los 2 de arriba) |

---

## FICHEROS CREADOS/ARREGLADOS en esta sesión

| Fichero | Estado |
|---------|--------|
| `taskB/3B_modeling.py` | ✅ Creado y bugs arreglados |
| `TaskB_notebook.ipynb` | ✅ Creado — notebook completo end-to-end |
| `setup_server.sh` | ✅ Creado — setup de venv en el servidor |

---

## EL PLAN COMPLETO — Orden exacto de ejecución en el servidor

### PASO 0: Conectar y hacer setup (solo la primera vez)

```bash
# 1. Conectar al servidor
ssh usuario@IP_SERVIDOR

# 2. Clonar el repo (o copiar con scp/rsync)
git clone https://github.com/ruedagarcialaura/DrugInteractionsRisks.git
cd DrugInteractionsRisks

# 3. Poner el ZIP (tú lo tienes localmente — súbelo desde tu máquina)
#    Desde Windows Git Bash, en OTRA terminal:
scp "/c/Users/sergi/LocalSpring2025/DrugInteractionsRisks/taskB/9 json files - no tocar.zip" \
    usuario@IP:~/DrugInteractionsRisks/taskB/

# 4. Correr el setup (crea venv + instala todo)
bash setup_server.sh
```

### PASO 1: Activar el entorno

```bash
source faers_env/bin/activate
# Confirmar:
which python    # debe apuntar a faers_env/bin/python
```

### PASO 2: Ejecutar el notebook (modo no-interactivo — recomendado para el servidor)

```bash
# Este comando corre el notebook completo y guarda los outputs (plots incluidos)
# Timeout de 2 horas por si el SVM tarda
nohup jupyter nbconvert \
  --to notebook \
  --execute TaskB_notebook.ipynb \
  --output TaskB_notebook_EXECUTED.ipynb \
  --ExecutePreprocessor.timeout=7200 \
  > notebook_run.log 2>&1 &

echo "PID: $!"
echo "Monitorear con: tail -f notebook_run.log"
```

### PASO 3 (Alternativa interactiva): Jupyter Lab con port forwarding

Si prefieres ver el notebook ejecutándose en tiempo real desde tu browser:

```bash
# En el SERVIDOR:
jupyter lab --no-browser --port=8888

# En tu máquina LOCAL (otra terminal):
ssh -L 8888:localhost:8888 usuario@IP

# En tu browser:
# http://localhost:8888
# Abre TaskB_notebook.ipynb y corre celda a celda
```

### PASO 4: Recoger los resultados

```bash
# Desde tu máquina local, cuando el notebook haya terminado:
scp usuario@IP:~/DrugInteractionsRisks/TaskB_notebook_EXECUTED.ipynb ./
scp -r usuario@IP:~/DrugInteractionsRisks/taskB/TaskBPlots/ ./taskB/
scp usuario@IP:~/DrugInteractionsRisks/taskB/task_b_evaluation.csv ./taskB/
```

Abre `TaskB_notebook_EXECUTED.ipynb` localmente — tiene todos los plots embebidos.

---

## Qué hace el notebook exactamente

El notebook `TaskB_notebook.ipynb` hace todo de forma inteligente:

```
Celda 1: Setup de imports + verifica que estamos en el directorio correcto
Celda 2: Configura paths y crea carpeta de outputs

FASE 0: Dependency check — muestra qué existe y qué falta
FASE 1: Genera filtered rules si no existen (corre 4A + 6A usando encoded parquets)
FASE 2: Data ingestion si no existe consolidated_data.parquet (corre 1B desde ZIP)
FASE 3: Feature engineering si no existe task_b_features.parquet (corre 2B)
FASE 4: EDA — class balance, distribuciones, estadísticas de interaction features
FASE 5: Train/test split estratificado
FASE 6: Entrena los 3 modelos (Logistic Regression, Random Forest, Linear SVM)
FASE 7: Plots inline — ROC curves, confusion matrices, feature importance, LR coefficients
FASE 8: Tabla comparativa con highlight del mejor modelo
FASE 9: Conclusiones
```

---

## Tiempo estimado de ejecución (servidor con CPU decente)

| Fase | Tiempo estimado |
|------|----------------|
| FASE 1: Generar reglas filtradas (4A + 6A) | 5-15 min |
| FASE 2: Data ingestion (1B) | 2-5 min |
| FASE 3: Feature engineering (2B) | 5-10 min |
| FASE 6a: Logistic Regression | 2-5 min |
| FASE 6b: Random Forest (100 trees, n_jobs=-1) | 5-15 min |
| FASE 6c: Linear SVM (CalibratedCV cv=3) | 10-20 min |
| TOTAL | ~30-70 min |

---

## CHECKLIST FINAL

- [x] `taskB/3B_modeling.py` — creado y bugs arreglados
- [x] `TaskB_notebook.ipynb` — creado, end-to-end, incluyendo generación de filtered rules
- [x] `setup_server.sh` — setup de venv completo
- [ ] Poner el ZIP en el servidor en: `taskB/9 json files - no tocar.zip`
- [ ] Correr `bash setup_server.sh` en el servidor
- [ ] Correr el notebook (modo nbconvert o Jupyter Lab)
- [ ] Descargar `TaskB_notebook_EXECUTED.ipynb` y los plots
- [ ] Añadir resultados al `readme.md`
