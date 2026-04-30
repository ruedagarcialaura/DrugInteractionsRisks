# GUÍA: EJECUTAR EN EL SERVIDOR REMOTO (NVIDIA 3060)
## Step-by-step para completar Task B

---

## Contexto

El servidor tiene una NVIDIA GeForce RTX 3060. Para Task B (sklearn classifiers) la GPU no se usa directamente — sklearn corre en CPU. Sin embargo, el servidor tiene más RAM y CPU que tu máquina local, lo que es clave para el SVM en 108k muestras.

Si quisieras usar GPU en el futuro: instalar XGBoost con soporte CUDA o cuML (RAPIDS) para Random Forest en GPU. Por ahora, el pipeline sklearn estándar es suficiente.

---

## Checklist pre-ejecución

Antes de conectar al servidor, asegúrate de tener:
- [ ] IP/hostname del servidor
- [ ] Credenciales SSH (usuario + contraseña o clave .pem)
- [ ] Ruta donde están (o van a estar) los datos en el servidor
- [ ] El ZIP de los 9 JSONs de FAERS en el servidor (o transferirlo)

---

## Flujo completo

### 1. Conectar al servidor

```bash
# Opción A: con contraseña
ssh usuario@IP_DEL_SERVIDOR

# Opción B: con clave SSH
ssh -i /ruta/a/clave.pem usuario@IP_DEL_SERVIDOR

# Opción C: si hay un túnel o VPN primero
# (pregúntale a quien tiene el servidor cómo conectarse)
```

### 2. Preparar el entorno en el servidor

```bash
# Ver si Python está instalado
python3 --version   # debería ser 3.8+

# Ver si conda/venv está disponible
conda --version
# o
python3 -m venv --version

# Crear entorno virtual (recomendado)
python3 -m venv faers_env
source faers_env/bin/activate   # Linux/Mac
# faers_env\Scripts\activate    # Windows

# Instalar dependencias
pip install numpy pandas scikit-learn matplotlib seaborn mlxtend fastparquet pyarrow tqdm tabulate
```

### 3. Transferir el proyecto al servidor

**Opción A: Clonar desde GitHub (si el repo es público o tienes acceso)**
```bash
# En el servidor
git clone https://github.com/ruedagarcialaura/DrugInteractionsRisks.git
cd DrugInteractionsRisks
```

**Opción B: SCP desde tu máquina local**
```bash
# Desde tu Windows (Git Bash o PowerShell)
scp -r /c/Users/sergi/LocalSpring2025/DrugInteractionsRisks usuario@IP:/home/usuario/
```

**Opción C: rsync (más eficiente, puede reanudar transferencias)**
```bash
rsync -avz --exclude='.git' \
  /c/Users/sergi/LocalSpring2025/DrugInteractionsRisks/ \
  usuario@IP:/home/usuario/DrugInteractionsRisks/
```

### 4. Transferir el ZIP de datos (si no está en el servidor)

```bash
# Desde tu máquina local — esto puede tardar según el tamaño del ZIP
scp "taskB/9 json files - no tocar.zip" usuario@IP:/home/usuario/DrugInteractionsRisks/taskB/
```

### 5. Ejecutar el pipeline completo

```bash
# En el servidor, dentro del directorio del proyecto
cd /home/usuario/DrugInteractionsRisks/

# Activar entorno si lo creaste
source faers_env/bin/activate

# PASO 1: Ingestion (si no existe consolidated_data.parquet)
python taskB/1B_dataIngestion.py "taskB/9 json files - no tocar.zip"
# Debería imprimir "Total Consolidated Records: 108,000" aprox.

# PASO 2: Feature engineering (si no existe taskB/task_b_features.parquet)
python taskB/2B_preprocessing.py
# Tarda unos minutos en procesar 108k reportes

# PASO 3: MODELADO — el script principal de Task B
python taskB/3B_modeling.py
# Tiempo estimado:
#   Logistic Regression: 2-5 minutos
#   Random Forest:       5-15 minutos  
#   SVM (LinearSVC):     5-10 minutos
#   TOTAL: ~30 minutos
```

### 6. Monitorear el progreso

Para dejar corriendo en background y poder cerrar SSH:
```bash
# Opción A: nohup (el proceso sigue aunque cierres la sesión SSH)
nohup python taskB/3B_modeling.py > output.log 2>&1 &
echo $!  # muestra el PID del proceso

# Monitorear el log en tiempo real:
tail -f output.log

# Ver si sigue corriendo:
ps aux | grep 3B_modeling

# Opción B: screen (más cómodo)
screen -S modeling
python taskB/3B_modeling.py
# Para desconectarte sin matar el proceso: Ctrl+A, luego D
# Para reconectar: screen -r modeling
```

### 7. Recuperar los resultados

```bash
# Desde tu máquina local, después de que termine
scp -r usuario@IP:/home/usuario/DrugInteractionsRisks/taskB/TaskBPlots/ ./taskB/
scp usuario@IP:/home/usuario/DrugInteractionsRisks/taskB/task_b_evaluation.csv ./taskB/
```

---

## Si hay problemas de memoria con el SVM

El script `3B_modeling.py` usa `LinearSVC` por defecto (más rápido). Si aun así hay problemas:

```bash
# Editar 3B_modeling.py para reducir dataset de entrenamiento:
# Cambiar el parámetro --sample-size al correr el script
python taskB/3B_modeling.py --quick-mode
```

El flag `--quick-mode` entrena con 20k muestras para ver resultados rápido.

---

## Verificación rápida: ¿qué ficheros deberían existir después de correr todo?

```
taskB/
├── task_b_features.parquet        ← después de 2B_preprocessing.py
├── TaskBPlots/
│   ├── confusion_matrices.png     ← después de 3B_modeling.py
│   ├── roc_curves.png             ← después de 3B_modeling.py
│   └── feature_importance_rf.png  ← después de 3B_modeling.py
└── task_b_evaluation.csv          ← tabla comparativa de los 3 modelos
```

---

## Troubleshooting común

### Error: "No module named mlxtend"
```bash
pip install mlxtend
```

### Error: "consolidated_data.parquet not found"
```bash
# Asegúrate de correr desde la raíz del proyecto
cd /home/usuario/DrugInteractionsRisks/
python taskB/1B_dataIngestion.py "taskB/9 json files - no tocar.zip"
```

### Error: "task_b_features.parquet not found"
```bash
python taskB/2B_preprocessing.py
```

### Error de memoria durante SVM
El script ya usa LinearSVC que escala mejor. Si sigue fallando, reducir con `--quick-mode`.

### SVM tarda demasiado
LinearSVC debería terminar en <10 minutos. Si tarda más, cancela (Ctrl+C) y comenta el bloque de SVM en el script. Logistic Regression + Random Forest ya son suficiente para la presentación.
