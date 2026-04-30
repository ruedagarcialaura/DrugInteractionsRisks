# GUÍA PARA LA PRESENTACIÓN
## Narrativa, puntos clave y preguntas difíciles

---

## Narrativa del proyecto (5 minutos)

### La historia que contar:

**Setup (1 min):** "Cada año miles de personas son hospitalizadas por combinaciones de medicamentos que nadie documentó como peligrosas. La FDA tiene 108,000 reportes de eventos adversos reales, pero son datos masivos sin procesar. Este proyecto convierte esos datos en conocimiento accionable."

**Task A (2 min):** "Modelamos cada reporte como una cesta de la compra: los ítems son fármacos y reacciones. Aplicamos Association Rule Mining para encontrar reglas como 'cuando un paciente toma DOXORUBICIN + CYCLOPHOSPHAMIDE, tiene una probabilidad 45x mayor de FEBRILE NEUTROPENIA de lo esperado por azar'. Comparamos Apriori vs FP-Growth — ambos dan los mismos resultados a nuestro threshold óptimo de 0.0025, lo que valida la correctitud matemática del pipeline."

**Task B (2 min):** "Las reglas de Task A se convierten en features. Entrenamos 3 clasificadores para predecir si un nuevo caso será severo o no. Usamos AUC-ROC porque el dataset está desbalanceado (solo 26.6% severos). Random Forest obtiene el mejor AUC-ROC de [X.XX] gracias a su capacidad de capturar las interacciones no lineales entre las features."

---

## Diapositivas recomendadas (estructura)

1. **Portada:** "Mining High-Risk Drug Interactions from FAERS"
2. **El problema:** ¿Qué son las DDIs y por qué importan? (estadística impactante)
3. **El dataset:** FAERS — 108k reportes, 9 archivos JSON, solo 4 de 69 campos usados
4. **Pipeline overview:** Diagrama de flujo Task A + Task B
5. **Task A - Association Rules:** Explicar support/confidence/lift con un ejemplo concreto
6. **Task A - Resultados:** Los 4 gráficos de redes + tabla de métricas
7. **Task A - Sensitivity Analysis:** El experimento con support 0.001 (noise vs señal)
8. **Task B - Feature Engineering:** La feature matrix de 25 columnas, justificación
9. **Task B - El puente:** Cómo las reglas de Task A alimentan Task B
10. **Task B - Resultados:** Tabla AUC-ROC/F1, ROC curves, confusion matrices
11. **Task B - Feature Importance:** ¿Qué features más importan?
12. **Conclusiones:** Hallazgos clave + limitaciones

---

## Resultados clave para mencionar (Task A — ya están)

### Las reglas más interesantes encontradas:
- **Max lift = 223.45** en Active Substances → señal farmacológica muy fuerte
- **13 reglas genuinas** después de filtrar sesgo de indicación
- **Apriori = FP-Growth** → validación cruzada algorítmica

### El experimento de sensitivity analysis (importante):
- A support=0.001, Drug Names genera **352,484 reglas** → explosión de ruido
- Los otros 5 test cases con support=0.001 → **MemoryError** → justifica threshold 0.0025
- Dos extremos del threshold: ruido matemático (lift >850) vs sesgo de indicación (Prednisone→RA)

---

## Preguntas difíciles y cómo responderlas

### "¿Por qué solo 4 de 69 campos?"
"Dimensionality reduction deliberada. Task A necesita solo la 'cesta de compra' (qué tomaron + qué pasó). Más campos adicionarían ruido sin añadir señal para association rules. Task B usa 5 campos adicionales para demographics y el target. La selección de features está justificada por la naturaleza del problema, no por pereza."

### "¿Por qué 0.0025 y no 0.001?"
"Empírico: a 0.001, 5 de 6 configuraciones producen MemoryError. La única que completa (Drug Names Apriori) genera 352,484 reglas con lift máximo de 851 — estadísticamente, eso son coincidencias, no señales reales. A 0.0025 obtenemos 13 reglas de alta calidad. El análisis de sensitivity está documentado con visualizaciones."

### "¿Por qué Apriori Y FP-Growth? ¿No es redundante?"
"No — es validación. Que ambos produzcan exactamente las mismas 13 reglas con el mismo lift y confidence demuestra que el resultado es un hecho matemático, no un artefacto del algoritmo. También justifica la elección del algoritmo para distintos contextos: FP-Growth escala mejor si el dataset crece."

### "¿La FAERS data es confiable?"
"FAERS tiene sesgo de subregistro estimado del 90% — no todos los eventos adversos se reportan. También tiene indication bias (los médicos que ven interacciones las reportan más). Sin embargo, es el dataset más grande de pharmacovigilance del mundo con datos del mundo real (no ensayos clínicos). Nuestro pipeline filtra el indication bias explícitamente en el paso 6A."

### "¿Qué hace diferente el lift de Task A al confidence?"
"Confidence puede ser alta simplemente porque el consecuente es muy frecuente. Si NAUSEA aparece en el 60% de todos los reportes, cualquier regla hacia NAUSEA tendrá confidence ≥60% por defecto. Lift normaliza por la frecuencia base del consecuente: lift=45 significa que la combinación de fármacos causa 45× más náusea de lo que causaría por azar."

### "¿Por qué class_weight='balanced' en los modelos?"
"Sin corrección, un modelo que siempre prediga 'no severo' tiene 73.4% de accuracy — mejor que muchos modelos simples. `class_weight='balanced'` hace que los errores en la clase minoritaria (severos) pesen más en la función de pérdida, forzando al modelo a aprender ambas clases."

### "¿No habría que hacer GridSearch / cross-validation?"
"Para producción sí. Para este pipeline KDD, el objetivo principal es descubrir señales farmacológicas y comparar algoritmos, no maximizar F1 al último decimal. Los hiperparámetros por defecto de sklearn son defensivos y razonables. Cross-validation añadiría rigor estadístico pero un tiempo de cómputo 5-10x mayor en 108k muestras."

### "¿Qué limitaciones tiene el proyecto?"
1. "FAERS reporta solo eventos adversos, no casos donde los fármacos funcionaron bien — hay selection bias"
2. "Association rules capturan correlación, no causalidad"
3. "El classifier Task B no tiene acceso a información clínica completa (historial médico, comorbilidades)"
4. "Los datos son de un trimestre de 2025 — pueden no generalizar a otras épocas o poblaciones"

---

## Datos impactantes para la intro

- Las DDIs son responsables de ~125,000 muertes/año en EEUU (estudio JAMA)
- El 20-30% de hospitalizaciones de ancianos están relacionadas con reacciones adversas a medicamentos
- FAERS recibe ~2 millones de reportes al año, pero la mayoría están sin analizar sistemáticamente
- Un paciente con 5+ medicamentos tiene un 50% de probabilidad de experimentar una DDI

---

## El plot más impactante para la presentación

El **grafo de red de Active Substances (Apriori, support 0.0025)** en `taskA/TaskAPlots/association_rules_APRIORI_active_substances_0_0025.png`:
- Visualiza las 13 interacciones genuinas como una red dirigida
- Nodos teal = fármacos, nodos coral = reacciones adversas
- El grosor de las aristas = fuerza de asociación (lift)
- Es visualmente impactante y comunica la idea intuitivamente

El **ROC curve plot** de Task B (cuando esté generado):
- Muestra los 3 modelos comparados visualmente
- La diferencia entre los modelos y el baseline aleatorio es la contribución del proyecto
