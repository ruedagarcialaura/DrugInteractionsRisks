#!/usr/bin/env bash
# ============================================================================
# setup_server.sh — Setup completo del entorno en el servidor remoto
# Ejecutar desde la raíz del proyecto: bash setup_server.sh
# ============================================================================
set -e   # exit on any error

echo "========================================"
echo " FAERS Drug Interactions — Server Setup"
echo "========================================"

# ── 1. Python version check ──────────────────────────────────────────────────
echo ""
echo "[1/5] Checking Python version..."
python3 --version
PYTHON_OK=$(python3 -c "import sys; print('ok' if sys.version_info >= (3,8) else 'fail')")
if [ "$PYTHON_OK" != "ok" ]; then
    echo "ERROR: Python 3.8+ required."
    exit 1
fi
echo "       Python OK ✓"

# ── 2. Create virtual environment ────────────────────────────────────────────
echo ""
echo "[2/5] Creating virtual environment (faers_env)..."
if [ -d "faers_env" ]; then
    echo "       faers_env already exists, skipping creation."
else
    python3 -m venv faers_env
    echo "       faers_env created ✓"
fi

# Activate
source faers_env/bin/activate
echo "       Activated: $(which python)"

# ── 3. Install dependencies ──────────────────────────────────────────────────
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip --quiet
pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    mlxtend \
    pyarrow \
    fastparquet \
    tqdm \
    tabulate \
    notebook \
    jupyterlab \
    nbconvert \
    ipykernel \
    --quiet

# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name faers_env --display-name "Python 3 (faers_env)"

echo "       Dependencies installed ✓"

# ── 4. Verify key imports ────────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying imports..."
python -c "
import pandas, numpy, sklearn, mlxtend, matplotlib, seaborn, pyarrow
from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
print('  All imports OK ✓')
print(f'  pandas   {pandas.__version__}')
print(f'  sklearn  {sklearn.__version__}')
print(f'  mlxtend  {mlxtend.__version__}')
"

# ── 5. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Checking project files..."
[ -f "taskA/active_substances_encoded.parquet" ] && echo "  ✓ active_substances_encoded.parquet" || echo "  ✗ active_substances_encoded.parquet MISSING"
[ -f "taskA/drug_names_encoded.parquet" ]         && echo "  ✓ drug_names_encoded.parquet"         || echo "  ✗ drug_names_encoded.parquet MISSING"
[ -f "TaskB_notebook.ipynb" ]                     && echo "  ✓ TaskB_notebook.ipynb"               || echo "  ✗ TaskB_notebook.ipynb MISSING"
[ -f "taskB/3B_modeling.py" ]                     && echo "  ✓ taskB/3B_modeling.py"               || echo "  ✗ taskB/3B_modeling.py MISSING"

ZIP_NAME="taskB/9 json files - no tocar.zip"
[ -f "$ZIP_NAME" ] && echo "  ✓ FAERS ZIP file" || echo "  ✗ FAERS ZIP — PUT IT AT: '$ZIP_NAME'"

echo ""
echo "========================================"
echo " Setup complete! Next steps:"
echo ""
echo " 1. Place ZIP at: 'taskB/9 json files - no tocar.zip'"
echo ""
echo " 2. Run the notebook (non-interactive):"
echo "    source faers_env/bin/activate"
echo "    jupyter nbconvert --to notebook --execute TaskB_notebook.ipynb \\"
echo "      --output TaskB_notebook_EXECUTED.ipynb \\"
echo "      --ExecutePreprocessor.timeout=7200"
echo ""
echo " 3. OR open Jupyter Lab in browser via port forwarding:"
echo "    [server]  jupyter lab --no-browser --port=8888"
echo "    [local]   ssh -L 8888:localhost:8888 usuario@IP"
echo "    [browser] http://localhost:8888"
echo ""
echo " 4. After execution, download results:"
echo "    scp -r usuario@IP:/ruta/proyecto/taskB/TaskBPlots/ ./taskB/"
echo "    scp usuario@IP:/ruta/proyecto/TaskB_notebook_EXECUTED.ipynb ./"
echo "========================================"
