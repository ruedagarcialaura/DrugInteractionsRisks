# QuercusHealth AI — Guión Final
**Formato:** rSpanglish — entiendes en español, lo dices en inglés  
**Tiempo total:** ~10 minutos  
**Regla:** Cada bloque tiene → lo que VAS A DECIR en inglés (en cursiva)

---

## SLIDE 1 — TITLE [0:00 – 0:30]

Saluda, di tu nombre, pon contexto rápido.

> *"Good morning. My name is Sergio Illescas, and this is QuercusHealth AI —
> a deep learning pipeline to detect diseased oak trees in Spanish satellite imagery,
> automatically, at scale, with no field visits needed."*

---

## SLIDE 2 — THE PROBLEM [0:30 – 1:30]

Explica qué es la Dehesa y qué es La Seca. El hook emocional.
Señala la imagen — verás los bounding boxes morados (las encinas anotadas con síntomas).

> *"This is the Spanish Dehesa — five million hectares of oak savanna, a UNESCO World Heritage landscape.
> It's being silently killed by a pathogen called Phytophthora cinnamomi, known locally as La Seca.
> La Seca rots the roots underground. By the time a forest ranger sees a dying tree from the ground,
> it's already too late to treat it.*
>
> *Manual monitoring of five million hectares is physically impossible.
> Our goal: detect which trees show early La Seca symptoms from satellite images —
> automatically, before the tree is lost."*

---

## SLIDE 3 — THE FULL PIPELINE [1:30 – 2:30]

Describe el pipeline de un vistazo. No entres en detalle — ya lo harás en cada slide.
Apunta a cada caja de izquierda a derecha.

> *"Here's the complete system we built across five phases.
> We start with satellite tile capture, run a zero-shot baseline to measure the problem,
> build our ground truth dataset, fine-tune the detector,
> add a two-stage classifier to separate Healthy from Seca —
> and finally evaluate the full end-to-end pipeline on raw tiles.*
>
> *Each phase answered a specific technical question. Let me walk you through them."*

---

## SLIDE 4 — DATA STRATEGY [2:30 – 3:30]

Explica cómo conseguiste el ground truth SIN ir al campo.
Señala las dos imágenes: izquierda = 2019 con boxes morados, derecha = 2024.

> *"Every AI model needs ground truth labels. The challenge here is: how do you label
> a sick tree from satellite imagery without physically visiting five million hectares?
>
> *Our solution was multi-temporal extraction. We scraped 324 tiles from Google Earth —
> the same GPS coordinates in Summer 2019 and Winter 2024.
> Trees that appeared suspicious in 2019 — those with sparse, radiating branches —
> and had disappeared five years later, are confirmed La Seca cases.
> Zero field visits.*
>
> *This gave us 7,341 ground truth boxes: 6,449 Healthy, 892 Seca.
> We also have 400 new tiles staged for the next annotation phase."*

---

## SLIDE 5 — DOMAIN SHIFT [3:30 – 4:15]

Señala la gráfica. Explica por qué el modelo original falla.
La clave: D=0.86 es estadísticamente devastador.

> *"Before building anything, we had to prove there was actually a problem.
> We took DeepForest — a RetinaNet pre-trained on millions of US NEON forest images —
> and ran it zero-shot on our Spanish tiles.*
>
> *F1 dropped from 0.68 to 0.32. But we didn't just observe this — we proved it statistically.
> A Kolmogorov-Smirnov test on the confidence score distributions gave D = 0.86, p less than 0.0001.
> These two environments are statistically different visual worlds to the model.*
>
> *Recall was 0.23 — missing three out of every four trees. And threshold tuning only gets us to 0.53.
> The model needs to learn what the Dehesa looks like. That's Domain Adaptation."*

---

## SLIDE 6 — FINE-TUNING STRATEGY [4:15 – 5:00]

Explica la arquitectura y POR QUÉ el learning rate bajo es crítico. Brevemente.

> *"We fine-tuned DeepForest — which uses a RetinaNet backbone with ResNet-50 —
> on our annotated Dehesa tiles, adding a two-class head: Healthy and Seca.*
>
> *The critical hyperparameter is the learning rate: 1e-4.
> Why so low? A standard learning rate would overwrite the backbone's weights
> before the head learns anything — catastrophic forgetting.
> Low LR preserves the model's understanding of tree crown shapes
> while teaching it the ochre Spanish soil and Seca morphology.*
>
> *We trained for 30 epochs on an RTX 3060, with flip, rotation, and color jitter augmentation."*

---

## SLIDE 7 — ABLATION STUDY [5:00 – 5:45]

Señala el bar chart. Este slide se explica casi solo.
El modelo con LR alto hizo CERO predicciones. Dramático.

> *"We didn't just choose a low learning rate arbitrarily — we proved it empirically.
> We trained two runs: one with LR = 1e-2, one with 1e-4.*
>
> *With the high learning rate, the model made zero predictions across the entire validation set
> after five epochs. The backbone was erased before the head could learn anything.*
>
> *With LR = 1e-4, F1 jumped to 0.669 — a 34.9 point improvement over the zero-shot baseline.
> This is the ablation study: catastrophic forgetting, confirmed experimentally."*

---

## SLIDE 8 — RESULTS: METRICS [5:45 – 6:15]

La tabla. Cuidado: F1-Seca = 0.000 aquí. No es un error del modelo, es el class imbalance.
Explícalo bien.

> *"Fine-tuning dramatically improved overall F1 from 0.32 to 0.669,
> and recall jumped from 0.23 to 0.742 — the model now finds most trees.*
>
> *But F1-Seca is still zero. This is not a fine-tuning failure —
> it's a class imbalance problem. 87.9% of training labels are Healthy.
> The gradient signal from 892 Seca examples is not strong enough
> to activate Seca predictions when the detector sees a full 800-pixel tile.
> Focal Loss helps, but it's not enough alone.*
>
> *This is exactly what Phase 4 solves."*

---

## SLIDE 9 — TRAINING DYNAMICS [6:15 – 6:40]

Señala la loss curve. Es muy visual, úsala.

> *"The loss curves confirm clean convergence — no overfitting.
> The sharp drop at epochs 14 to 16 is when the backbone adapts to Dehesa soil color.
> Validation loss tracks training loss closely throughout.*
>
> *The model learns where trees are — Recall 0.742 — but still cannot separate
> Healthy from Seca. For that, we need the two-stage pipeline."*

---

## SLIDE 10 — VISUAL COMPARISON [6:40 – 7:05]

Señala las dos imágenes. Muestra el progreso visual.
Cada imagen tiene 3 paneles: GT | Baseline | Fine-Tuned.

> *"Visually, the improvement is clear. Each image shows ground truth on the left,
> the zero-shot baseline in the middle, and our fine-tuned model on the right.*
>
> *The baseline detects very few trees. Fine-tuning recovers localisation —
> but you'll notice all boxes are green. No Seca detections yet.
> The full-tile context still overwhelms the minority class signal."*

---

## SLIDE 11 — FAILURE CASES [7:05 – 7:30]

Análisis crítico. Esto da puntos. Muestra que entiendes el modelo.

> *"A good model analysis requires understanding where it still fails — and why.
> We identified three systematic failure modes:*
>
> *First: bright sandy background — sparse oaks in ochre soil have low contrast,
> confusing the anchor grid. 42% of boxes missed.*
>
> *Second: high-density crown overlap — when crowns merge into a single connected region,
> single-tree anchor assumptions break. 56% missed.*
>
> *Third: mixed shrubs and shadows — the model confuses dense low vegetation with tree crowns,
> and shadows shift confidence scores down. 54% missed.*
>
> *These are systematic, not random — each has a targeted fix:
> contrast augmentation, NMS tuning, or adding a near-infrared channel."*

---

## SLIDE 11b — PHASE 4: TWO-STAGE [7:30 – 8:15]

Este es el resultado estrella. F1-Seca 0 → 0.354. Hazlo sonar importante.
Señala el pipeline steps y luego la confusion matrix.

> *"Phase 4 breaks the Seca detection barrier.*
>
> *Instead of one detector trying to do everything,
> we decouple the problem into two stages:
> Stage 1 — the fine-tuned DeepForest detector finds all trees and outputs bounding boxes.
> Stage 2 — each box is cropped to 96 pixels and fed to a ResNet-18 classifier
> that decides: Healthy or Seca.*
>
> *Why does this work? The classifier sees only the tree crop —
> no background noise, no ochre soil. With WeightedRandomSampler and class-weighted loss,
> 892 Seca crops are enough for ResNet-18 to learn the minority class.*
>
> *F1-Seca: 0.000 to 0.354. Plus 35.4 percentage points.
> F1-Overall: 0.669 to 0.712."*

---

## SLIDE 11c — PHASE 5: END-TO-END [8:15 – 8:50]

Explica que Phase 4 hizo trampa (usaba GT crops). Phase 5 es el sistema real.
Señala los tres tiles abajo — visual proof.

> *"Phase 4 had one limitation: Stage 2 was evaluated on perfect ground truth crops —
> the detector was bypassed. That's not a real system.*
>
> *Phase 5 closes the loop. Stage 1 runs on raw tiles, predicts boxes,
> those boxes are cropped and classified by Stage 2,
> and we match predictions to ground truth by IoU of at least 0.4.*
>
> *The result: F1-Seca drops from 0.354 to 0.346 — just 0.8 points.
> This is the localisation tax — the real cost of Stage 1 occasionally missing or misplacing trees.
> The classifier generalises well under real detection noise.*
>
> *These three tiles show the full pipeline output: green is Healthy, red is Seca,
> white dashed are the ground truth boxes."*

---

## SLIDE 12 — CONCLUSION [8:50 – 9:45]

Cierra fuerte. Primero los logros, luego las limitaciones, luego la visión.
La última línea tiene que sonar bien dicha.

> *"Let me summarise what we built and proved.*
>
> *We proved the domain gap is statistically real — KS D = 0.86.
> We closed 34.9 percentage points of F1 through fine-tuning.
> We broke the Seca detection barrier with a two-stage pipeline — F1-Seca from 0 to 0.354.
> And we validated the complete system end-to-end at F1-Seca 0.346 on raw satellite tiles.*
>
> *Our limitations are clear: we need more Seca annotations to push above 0.60,
> and a near-infrared channel would dramatically improve recall in shadow and shrub cases.
> Both are in Phase 6.*
>
> *The deployment vision is simple: a regional government uploads last summer's Google Earth tiles,
> the model returns a ranked list of GPS coordinates where La Seca is likely,
> and inspectors go exactly where the disease is — before the tree is gone.*
>
> *At two seconds per tile on GPU, a full scan of five million hectares takes under 45 hours, once a year.*
>
> *The tooling is real. The pipeline works. The forest is waiting.*
>
> *Thank you."*

---

## Q&A — Respuestas preparadas

**"Why not train from scratch?"**
> *"DeepForest's backbone already knows what a tree crown looks like from above —
> that's millions of NEON images of prior knowledge. Training from scratch with 14 annotated tiles
> would overfit immediately. Transfer learning plus low LR is the right tool for small domain datasets."*

**"Why IoU 0.4 and not 0.5 for Phase 5?"**
> *"Stage 1 predicts slightly imprecise boxes on raw tiles — the crops are not perfect rectangles.
> IoU 0.4 is a fair match threshold that accounts for this imprecision
> without being so loose it counts wrong detections as correct."*

**"What's next?"**
> *"Phase 6: annotate the 400 staged tiles, add NIR channel data from Sentinel-2,
> and target F1-Seca above 0.60. The infrastructure is already in place."*

**"Why did you use RetinaNet and not YOLO?"**
> *"DeepForest wraps a RetinaNet backbone specifically trained for aerial tree detection.
> It's the state of the art for this task. YOLO would require training from scratch
> on a dataset too small to generalise."*
