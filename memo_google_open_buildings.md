# Research Memo: Google Open Buildings and Its Role as Training Labels

**Date:** 2026-03-08
**Project:** Africa-Econ-Satellite (Microspatial GDP Mapping)

---

## 1. The Paper: Sirko et al. (2021)

### Citation

Sirko, W., Kashubin, S., Ritter, M., Annkah, A., Bouchareb, Y.S.E., Dauphin, Y., Keysers, D., Neumann, M., Cisse, M., and Quinn, J. (2021). "Continental-Scale Building Detection from High Resolution Satellite Imagery." arXiv preprint arXiv:2107.12283. https://doi.org/10.48550/arXiv.2107.12283

### Methodology

The paper describes a pipeline for detecting individual buildings across the African continent from 50cm commercial satellite imagery (Maxar Technologies). The core approach is pixel-wise semantic segmentation: each pixel is classified as building or non-building, then connected components are grouped into individual building instances.

### Model Architecture

The model is a **U-Net variant** -- an encoder-decoder architecture with skip connections. Key design choices:

- **Distance weighting:** Instead of standard binary masks, the loss function uses Gaussian convolution of building edges to emphasize boundary detection. This prevents neighboring buildings from merging. Removing this degrades mAP by 0.33 -- the single most important design choice.
- **Mixup regularization:** Random training images are blended via weighted averaging, improving generalization across diverse terrain types (savannah, desert, forest, informal settlements). Impact: +0.12 mAP.
- **ImageNet pre-training:** Transfer learning from ImageNet provides a warm start. Impact: +0.07 mAP.
- **Self-training (Noisy Student):** A trained teacher model generates pseudo-labels on 8.7 million unlabeled satellite images. A student model is then trained on augmented versions of this expanded dataset with soft KL divergence loss, reducing false positives. Impact: +0.06 mAP.

The architecture is deliberately compact to allow continental-scale inference without prohibitive compute costs.

### Training Data

- **100,000 satellite images** across Africa
- **1.75 million manually labeled building instances** (hand-annotated polygons)
- Additional unlabeled datasets for self-training (8.7M images)
- Satellite source: **Maxar Technologies** commercial imagery at **50cm resolution**

### Coverage and Output

- **516 million buildings** detected in Africa (V1)
- **1.8 billion buildings** globally in V3 (Africa + South Asia + Southeast Asia + Latin America + Caribbean)
- Coverage: **58 million km2** of inference area
- Africa: 19.4 million km2 (64% of the continent)
- 8.6 billion image tiles processed

### Accuracy Metrics

Each detected building carries a confidence score (0.65 to 1.0). The dataset provides per-tile suggested thresholds to achieve target precision levels:

- **80% precision threshold** (higher recall)
- **85% precision threshold**
- **90% precision threshold** (lower recall)

Performance varies by building density category:

- **Urban areas:** Higher precision but challenges with contiguous buildings that lack clear boundaries; tendency to split large buildings into multiple instances
- **Medium density:** Generally best performance
- **Rural areas:** Lower recall -- small buildings may be only a few pixels wide even at 50cm; label noise also higher in sparse regions
- **Arid/desert regions:** Lower precision due to natural materials blending with building materials

Regarding the approximate figures of ~0.95 precision and ~0.70 recall for Africa at the default confidence threshold (0.65): these represent rough continent-wide averages. The key point is the **asymmetry** -- precision is substantially higher than recall, meaning the dataset misses more buildings than it hallucinates.

---

## 2. Key Insight: Our Pipeline vs. Theirs

This is the crucial architectural distinction that must be stated clearly in any paper.

### Their Pipeline (Google Open Buildings)
```
Input:  Maxar commercial imagery (50cm, RGB)
Model:  Modified U-Net (encoder-decoder)
Output: Individual building polygons with confidence scores
Cost:   Commercial imagery ($$$), massive compute
```

### Our Pipeline (Africa-Econ-Satellite)
```
Input:  Sentinel-2 (10m, 12 spectral bands) -- FREE
Labels: Google Open Buildings polygons (rasterized to 10m grid)
Model:  U-Net variant (our own training)
Output: Building presence/density probability per 10m pixel
Cost:   Free imagery, moderate compute
```

### What We Are Actually Doing

We are performing a form of **cross-resolution knowledge transfer**. Google's model, trained on commercial 50cm imagery with human-annotated labels, produced building polygons. We take those polygons as ground truth and train a new model to predict building presence from freely available 10m Sentinel-2 imagery.

This is emphatically **not** a replication of Google's pipeline. We do not have access to 50cm imagery. We are not trying to detect individual buildings. We are learning a mapping from medium-resolution multispectral features to building density -- using Google's output as a noisy supervisory signal.

The analogy is: Google built a high-resolution "oracle" using expensive inputs. We train a cheaper, coarser model to approximate that oracle's outputs using free inputs. The ML literature calls this **knowledge distillation** or **teacher-student learning**, though our case is unusual because:

1. The teacher and student operate at different spatial resolutions
2. The teacher's outputs (polygons) are aggregated/rasterized before serving as labels
3. We never access the teacher model itself -- only its outputs
4. The student uses different input features (multispectral vs. RGB)

---

## 3. Implications of Using Open Buildings as Labels

### 3.1 Label Noise: Precision-Recall Asymmetry

Google Open Buildings achieves high precision (~0.95 at recommended thresholds) but moderate recall (~0.70), particularly in rural Africa. This creates **asymmetric label noise**:

- **False negatives (missed buildings, ~30% of true buildings):** Our model is trained to predict "no building" where buildings actually exist. This biases our model toward under-prediction of building presence, especially in rural areas with small or informal structures.
- **False positives (hallucinated buildings, ~5%):** Relatively rare. Our model occasionally learns to predict buildings where none exist, but this is a minor issue.

**Practical consequence:** Our model will systematically underestimate building density in areas where Open Buildings has low recall. Since recall is lowest in rural areas with small buildings, this creates a **spatial bias** -- our model is most accurate in denser urban/peri-urban areas and least reliable in sparse rural settings.

**Mitigation strategies:**
- Use higher-confidence polygons (>0.75) to maximize precision, accepting even lower recall
- Treat the problem as semi-supervised: pixels without building labels are "unknown," not "negative"
- Apply label smoothing to avoid training on hard zeros in potentially-missed-building areas

### 3.2 Resolution Mismatch: 50cm to 10m

A single 10m Sentinel-2 pixel covers a 10m x 10m area = 100 m2. At 50cm resolution, this same area contains 20 x 20 = **400 pixels** in Maxar imagery.

Implications:

- **Building fraction, not building presence:** At 10m, the natural label is not binary (building/no-building) but fractional: what proportion of the 10m pixel is covered by buildings? A 10m pixel might contain 30% building footprint. Training with hard binary labels (any overlap = 1) vs. soft fractional labels will produce different models.
- **Mixed pixels dominate:** In all but the densest urban cores, most 10m pixels will be mixed -- containing some building and some non-building area. The spectral signature of these pixels is a blend, making the classification problem fundamentally harder than at 50cm.
- **Small buildings vanish:** A 3m x 3m building covers 9% of a 10m pixel. Its spectral contribution is swamped by the surrounding land cover. Google's model can detect this building at 50cm; our model likely cannot at 10m. This is a fundamental resolution limit, not a model limitation.
- **Building counting is impossible:** At 10m, we cannot count individual buildings. We can estimate building density or built-up fraction. This is actually fine for our research question (economic density), but it means our model answers a different question than Google's.

**Best practice for label construction:**
- Compute building **area fraction** per 10m pixel from the Open Buildings polygons
- Use this fraction as a continuous label (regression target) rather than a binary one
- This better represents the ground truth at 10m resolution and is directly interpretable as building density

### 3.3 Knowledge Distillation and Noisy Label Theory

Our approach fits within the broader ML literature on learning from imperfect supervision:

**Knowledge distillation (Hinton et al., 2015):** Originally proposed for model compression, where a small "student" network is trained to match the soft outputs of a large "teacher" network. The key insight is that the teacher's probability distribution over classes (soft labels) carries more information than hard labels alone. Our case differs: we use the teacher's *thresholded* output polygons rather than soft probabilities, and we operate across resolution scales.

> Hinton, G., Vinyals, O., and Dean, J. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.

**Learning from noisy labels:** A rich literature establishes that deep networks can learn effectively from noisy labels under certain conditions:

- *Natarajan et al. (2013)* showed that if the noise rates (false positive and false negative rates) are known, the optimal classifier under clean labels can be recovered by cost-sensitive learning on noisy labels. Applied to our setting: if we know Open Buildings' precision (~0.95) and recall (~0.70), we can theoretically correct for the label noise.

  > Natarajan, N., Dhillon, I.S., Ravikumar, P., and Tewari, A. (2013). "Learning with Noisy Labels." Advances in Neural Information Processing Systems 26 (NeurIPS).

- *Arpit et al. (2017)* demonstrated that DNNs first learn "simple patterns" from the data before memorizing noisy labels in later epochs. This suggests that early stopping or regularization can help our model learn the real building signal without overfitting to label noise.

  > Arpit, D. et al. (2017). "A Closer Look at Memorization in Deep Networks." Proceedings of the 34th International Conference on Machine Learning (ICML).

- *Li et al. (2017)* specifically studied knowledge distillation as a mechanism for handling noisy labels, showing that the distillation process itself acts as a regularizer that suppresses the influence of mislabeled examples.

  > Li, Y., Yang, J., Song, Y., Cao, L., Luo, J., and Li, L.-J. (2017). "Learning from Noisy Labels with Distillation." Proceedings of the IEEE International Conference on Computer Vision (ICCV).

**Practical takeaway:** The literature suggests our approach is fundamentally sound. Training on Open Buildings labels -- even with ~30% false negative rate -- will produce a useful model, especially with:
- Appropriate loss functions (focal loss / focal Tversky, which we already use)
- Regularization (early stopping, dropout, augmentation)
- Treating confidence scores as label quality indicators

### 3.4 Coverage Gaps and Spatial Bias

Open Buildings coverage is not uniform:

- **Temporal gaps:** The underlying Maxar imagery was captured at various times. Some areas may have older imagery that misses recent construction.
- **Cloud/quality gaps:** Some areas lack sufficient high-quality cloud-free imagery for reliable detection.
- **Rural bias:** Rural areas have lower recall due to smaller buildings, lower image contrast, and sparser training data in Google's own pipeline.
- **Regional variation:** The paper documents varying performance across regions -- e.g., Sierra Leone and Mozambique had sparser labeling; Cairo showed building-splitting artifacts.

**For our model:** These coverage gaps translate directly into label quality variation across our study area. Our model will be more reliable in areas where Open Buildings performs well (dense settlements, cloud-free imagery) and less reliable where it performs poorly (sparse rural areas, recently developed areas).

---

## 4. Why NOT Replicate Google's Pipeline

| Dimension | Google Open Buildings | Our Approach |
|---|---|---|
| **Imagery** | Maxar 50cm (commercial, $$) | Sentinel-2 10m (free, open) |
| **Resolution** | Individual building footprints | Building density per 10m pixel |
| **Temporal coverage** | Snapshot (varies by tile) | Regular 5-day revisits since 2017 |
| **Spectral bands** | RGB (3 bands) | 12 bands including NIR, SWIR, Red Edge |
| **Research question** | Where are buildings? | Where is economic activity? |
| **Scalability** | Requires commercial imagery purchases | Fully reproducible with open data |
| **Temporal analysis** | Limited by imagery availability | Can track changes over time |
| **Cost** | Imagery + massive compute | Moderate compute only |

### Key Reasons

1. **Cost prohibition:** Maxar imagery costs thousands of dollars per tile. Continental-scale coverage is only feasible for well-funded organizations (Google, Microsoft). Sentinel-2 is free and globally available.

2. **Different research question:** We do not need individual building footprints. We need a *proxy for economic density* at moderate resolution. Building presence probability at 10m is actually better suited to this task than individual polygons, because it naturally aggregates to the district level where we validate against economic data.

3. **Temporal dimension:** Sentinel-2 provides repeat observations every 5 days. This enables change detection and temporal compositing, which is impossible with one-time commercial imagery snapshots. Our model can potentially track urbanization dynamics.

4. **Spectral richness:** Sentinel-2's 12 bands (including near-infrared, red edge, and shortwave infrared) provide information about land cover that RGB alone cannot. Vegetation indices (NDVI), built-up indices (NDBI), and soil brightness can all help distinguish built-up areas from bare soil -- a key confusion source at 10m resolution.

5. **Reproducibility:** Any researcher can replicate our pipeline using only open data. This is a core scientific value.

---

## 5. Literature Connections

### 5.1 Building Detection from Medium-Resolution Satellite Imagery

**Sirko, W., Asiedu Brempong, E., Marcos, J.T.C., et al. (2023). "High-Resolution Building and Road Detection from Sentinel-2." arXiv:2310.11622.**

This is the most directly relevant paper -- also from Google Research. It explicitly implements the teacher-student framework we use implicitly:
- Teacher model: trained on 50cm imagery, generates building/road segmentation masks
- Student model: trained on Sentinel-2 10m imagery to reproduce teacher's predictions
- Key finding: Using stacks of up to 32 temporally offset Sentinel-2 frames, the student achieves **78.3% mIoU** for building segmentation (vs. 85.3% for the teacher on high-res imagery)
- Achieves performance comparable to models running on single 4m-resolution images
- Building counting R2: 0.91 (student) vs. 0.95 (teacher on high-res)

This paper validates the core idea behind our approach: Sentinel-2 contains enough information to predict building presence, even though individual buildings are sub-pixel.

**Corbane, C., Syrris, V., Sabo, F., et al. (2021). "Convolutional neural networks for global human settlements mapping from Sentinel-2 satellite imagery." Neural Computing and Applications, 33, 6697-6717.**

Developed GHS-S2Net for the European Commission's Global Human Settlement Layer (GHSL):
- Lightweight CNN (1.4M parameters, 4 conv layers) for pixel-wise classification
- Global coverage using Sentinel-2 composites at 10m
- Validated against 40 million building polygons across 277 areas of interest
- Uses UTM-zone-specific transfer learning to handle geographic variation
- Produces the GHS-BUILT-S2 layer used in many development studies

**Qiu, C., Schmitt, M., Geiss, C., Chen, T.-H.K., and Zhu, X.X. (2020). "A framework for large-scale mapping of human settlement extent from Sentinel-2 images via fully convolutional neural networks." ISPRS Journal of Photogrammetry and Remote Sensing, 163, 152-170.**

- Fully convolutional architecture (Sen2HSE) for settlement extent mapping
- Validated against OpenStreetMap building layer and manual labels
- Addresses the specific challenges of mixed pixels at 10m resolution

### 5.2 Using ML Model Outputs as Training Labels

**Hinton, G., Vinyals, O., and Dean, J. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.**

The foundational knowledge distillation paper. Key insight: soft probability outputs from a teacher model contain "dark knowledge" about inter-class similarities that hard labels do not. While our case uses hard(ish) polygon labels rather than soft probabilities, the principle applies: Google's model has encoded information about building detection that we transfer to our student model.

**Li, Y., Yang, J., Song, Y., Cao, L., Luo, J., and Li, L.-J. (2017). "Learning from Noisy Labels with Distillation." IEEE International Conference on Computer Vision (ICCV).**

Directly studies how knowledge distillation interacts with label noise. Shows that distillation acts as a regularizer against noisy labels, and that student models can sometimes outperform teachers when label noise is present.

### 5.3 Sentinel-2 for Urban/Built-Up Area Mapping

**Pesaresi, M., Corbane, C., Julea, A., et al. (2016). "Assessment of the Added-Value of Sentinel-2 for Detecting Built-up Areas." Remote Sensing, 8(4), 299.**

Early assessment of Sentinel-2's capabilities for built-up area detection, establishing that the 10m bands plus red-edge and SWIR bands provide meaningful improvements over Landsat for urban mapping.

### 5.4 Connecting Building Detection to Economic Outcomes

**Jean, N., Burke, M., Xie, M., Davis, W.M., Lobell, D.B., and Ermon, S. (2016). "Combining satellite imagery and machine learning to predict poverty." Science, 353(6301), 790-794.**

The landmark paper establishing that CNN features extracted from satellite imagery can explain up to 75% of variation in local economic outcomes across five African countries. Uses a transfer learning approach (nightlights as intermediate target) to overcome limited survey data. Demonstrates the fundamental link between visible satellite features and economic well-being.

**Yeh, C., Perez, A., Driscoll, A., et al. (2020). "Using publicly available satellite imagery and deep learning to understand economic well-being in Africa." Nature Communications, 11, 2583.**

Extends Jean et al. using publicly available multispectral imagery (Landsat, Sentinel) to predict asset wealth across ~20,000 African villages. Key results:
- Explains 70% of variation in village wealth in held-out countries
- Can explain up to 50% of variation in *changes* in wealth over time
- Daytime imagery outperforms nightlights for predicting wealth changes
- Demonstrates that freely available medium-resolution imagery is sufficient for economic prediction

This paper is our closest methodological cousin: it uses free Sentinel-type imagery to predict economic outcomes, though it predicts wealth directly rather than using building detection as an intermediate step.

**Goldblatt, R., You, W., Hanson, G., and Khandelwal, A.K. (2016). "Detecting the Boundaries of Urban Areas in India: A Dataset for Pixel-Based Image Classification in Google Earth Engine." Remote Sensing, 8(8), 634.**

Demonstrates pixel-level urban classification using freely available imagery in a developing-country context, connecting urban extent to economic activity.

---

## 6. Summary: Position Statement for Our Paper

The recommended framing for our methodology section:

> We use building footprint polygons from Google Open Buildings V3 (Sirko et al., 2021) as training labels for a U-Net segmentation model operating on freely available Sentinel-2 imagery at 10m resolution. This approach constitutes a form of cross-resolution knowledge transfer: the Open Buildings dataset, derived from commercial 50cm satellite imagery using a separate deep learning pipeline, serves as a noisy but high-quality supervisory signal. Our model learns to predict building density -- not individual footprints -- from the multispectral features available in Sentinel-2, effectively distilling the building detection capability of a high-resolution commercial system into a model that runs entirely on open data.
>
> This design choice entails specific label noise characteristics. Open Buildings achieves high precision (~0.95) but moderate recall (~0.70) across Africa, with recall lowest in rural areas with small or informal structures. Our training labels therefore exhibit asymmetric noise: false negatives (missed buildings) substantially outnumber false positives. We address this through [focal Tversky loss / label smoothing / confidence thresholding] and note that our downstream validation target -- district-level economic intensity -- is robust to pixel-level label noise because aggregation over thousands of pixels attenuates random errors.

---

## Sources

- [Sirko et al. 2021 - arXiv](https://arxiv.org/abs/2107.12283)
- [Open Buildings V3 - Earth Engine Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings_v3_polygons)
- [Google Research Blog - Mapping Africa's Buildings](https://research.google/blog/mapping-africas-buildings-with-satellite-imagery/)
- [Open Buildings Website](https://sites.research.google/gr/open-buildings/)
- [Sirko et al. 2023 - High-Resolution Building Detection from Sentinel-2](https://arxiv.org/abs/2310.11622)
- [Corbane et al. 2021 - GHS-S2Net](https://link.springer.com/article/10.1007/s00521-020-05449-7)
- [Qiu et al. 2020 - Sen2HSE](https://www.sciencedirect.com/science/article/pii/S0924271620300344)
- [Jean et al. 2016 - Predicting Poverty](https://www.science.org/doi/10.1126/science.aaf7894)
- [Yeh et al. 2020 - Economic Well-Being in Africa](https://www.nature.com/articles/s41467-020-16185-w)
- [Hinton et al. 2015 - Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [Li et al. 2017 - Noisy Labels with Distillation](https://arxiv.org/abs/1703.02391)
- [Natarajan et al. 2013 - Learning with Noisy Labels](https://papers.nips.cc/paper/5073-learning-with-noisy-labels)
- [Arpit et al. 2017 - Memorization in Deep Networks](https://openreview.net/pdf?id=H12GRgcxg)
