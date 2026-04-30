# ÍNDICE — local-files/
## Documentación completa del proyecto

Leer en este orden si es la primera vez:

| Fichero | Qué contiene | Urgencia |
|---------|--------------|----------|
| `00_VISION_GENERAL.md` | Qué es el proyecto, qué está hecho, qué falta | LEER PRIMERO |
| `01_TASK_A_pipeline.md` | Pipeline de Association Rules explicado a fondo | Task A |
| `02_TASK_B_pipeline.md` | Pipeline de Classification explicado a fondo | Task B |
| `03_LO_QUE_FALTA.md` | Checklist de tareas pendientes + plan de acción | CRÍTICO |
| `04_CONCEPTOS_CLAVE.md` | Glosario técnico: KDD, Lift, AUC-ROC, etc. | Referencia |
| `05_SERVIDOR_REMOTO.md` | Guía SSH paso a paso para ejecutar en la 3060 | Para ejecutar |
| `06_RESUMEN_PRESENTACION.md` | Narrativa, preguntas difíciles, datos impactantes | Para presentar |

---

## Estado del proyecto en una línea

```
Task A: 100% COMPLETO ✅ (8/8 scripts + plots generados)
Task B: 66% COMPLETO ⚠️  (2/3 scripts — falta ejecutar 3B_modeling.py)
```

## El fichero más importante que hay que ejecutar

```bash
# En el servidor remoto, desde la raíz del proyecto:
python taskB/3B_modeling.py
```

Este script está en `taskB/3B_modeling.py` y genera:
- Tabla comparativa AUC-ROC / F1 de los 3 clasificadores
- Confusion matrices plot
- ROC curves plot
- Feature importance del Random Forest
