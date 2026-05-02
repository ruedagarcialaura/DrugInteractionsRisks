# ÍNDICE — local-files/
## Documentación completa del proyecto

Leer en este orden si es la primera vez:

| Fichero | Qué contiene | Urgencia |
|---------|--------------|----------|
| `00_VISION_GENERAL.md` | Qué es el proyecto, qué está hecho, qué falta | LEER PRIMERO |
| `01_TASK_A_pipeline.md` | Pipeline de Association Rules explicado a fondo | Task A |
| `02_TASK_B_pipeline.md` | Pipeline de Classification explicado a fondo | Task B |
| `03_LO_QUE_FALTA.md` | Checklist de tareas (⚠️ DESACTUALIZADO — ver 07) | Histórico |
| `04_CONCEPTOS_CLAVE.md` | Glosario técnico: KDD, Lift, AUC-ROC, etc. | Referencia |
| `05_SERVIDOR_REMOTO.md` | Guía SSH (⚠️ DESACTUALIZADO — refleja modelos viejos) | Histórico |
| `06_RESUMEN_PRESENTACION.md` | Narrativa, preguntas difíciles, datos impactantes | Para presentar |
| **`07_ITERACIONES_TASK_B.md`** | **Historial completo de iteraciones + resultados finales** | **ESTADO ACTUAL** |

---

## Estado del proyecto en una línea

```
Task A: 100% COMPLETO ✅ (8/8 scripts + plots generados)
Task B: 100% COMPLETO ✅ (4/4 scripts + Optuna tuning + plots + AUC 0.805)
```

## Resultado final Task B

```
Mejor modelo:  Tuned Ensemble (XGBoost + CatBoost, Optuna-tuned)
AUC-ROC:       0.8054
F1-severe:     0.5860
Recall-sev:    0.6512
```

Ver `07_ITERACIONES_TASK_B.md` para el historial completo de cómo se llegó aquí.
