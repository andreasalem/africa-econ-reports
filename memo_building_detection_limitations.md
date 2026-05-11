# Analytical Memo: Building Detection as a Proxy for Economic Activity
## Methodological Quality, Limitations, and Paths Forward

*Prepared: 2026-03-08*
*For: Africa-Econ-Satellite Project (GDP Mapping, Project A)*
*Target audience: Top-5 economics journal (AER/QJE) framing*

---

## 1. The Semantic Bottleneck: Why Building Count != Economic Intensity

The core methodological concern for any referee at AER/QJE is straightforward: a U-Net trained on Google Open Buildings labels learns to detect *physical structures*, not *economic activity*. Building presence is a necessary but insufficient condition for economic output. The gap between pixel-level detection performance (AUC = 0.976) and district-level economic correlation (Pearson r = 0.38-0.43) is not a model failure -- it is a *conceptual ceiling* imposed by the proxy itself.

### 1.1 Specific Sources of Semantic Mismatch

**Heterogeneous economic intensity per unit footprint.** A warehouse employing 500 workers and generating millions in revenue occupies roughly the same 2D footprint as a small retail shop employing 2 people. At 10m Sentinel-2 resolution, both register as identical building-present pixels. Building count treats them as interchangeable, but their economic contributions differ by orders of magnitude. This is the intensive-margin problem: satellite-derived building counts capture the *extensive* margin of economic activity (whether activity exists) but not the *intensive* margin (how much activity occurs) (Baragwanath Vogel et al., 2021, Journal of Urban Economics; Goldblatt et al., 2020, World Bank Economic Review).

**Informal economy invisibility.** In Sub-Saharan Africa, the informal sector accounts for 25-65% of GDP and an even larger share of employment (ILO estimates). Market stalls, street vendors, mobile traders, and home-based enterprises leave minimal or zero building footprint. Recent work by researchers using satellite imagery to map rural markets in Ethiopia found that market activity has a "unique temporal and visual signature" but requires specialized detection methods distinct from building footprint analysis (arxiv:2407.12953). Standard building detection systematically undercounts economic activity in precisely the places where measurement is most needed.

**Building age, quality, and functional status.** A derelict structure and an active factory may present near-identical spectral signatures at 10m resolution. Sentinel-2 cannot reliably distinguish between occupied commercial buildings, abandoned structures, residential dwellings, and public/government buildings. The Google Open Buildings confidence score reflects detection confidence, not building function or economic use. This functional ambiguity degrades the building-to-economic-activity mapping.

**Vertical density is invisible.** Multi-story buildings are ubiquitous in African commercial districts (Kigali, Lagos, Nairobi). A 10-story commercial building generates far more economic output per footprint pixel than a single-story residential structure. At 10m resolution, both are one building. Google's Open Buildings 2.5D Temporal dataset now provides building *height* estimates from Sentinel-2 time series (2016-2023), which could partially address this, but the current pipeline uses only presence/absence.

**Land use heterogeneity within building class.** Religious buildings, schools, government offices, and commercial establishments all register as "buildings" but contribute very differently to establishment census counts. The semantic gap between "is a building" and "is an economic establishment" cannot be closed by improving pixel-level detection accuracy.

### 1.2 What the Literature Says

Donaldson and Storeygard (2016, JEP) provide the canonical framework for evaluating satellite-derived economic proxies. They emphasize that satellite data measures *physical correlates* of economic activity, not economic activity itself, and that the strength of the correlation depends on how tightly the physical signal tracks the economic concept of interest. Building presence is a weaker correlate of economic intensity than, say, nightlight luminance is of electrified economic activity -- because nightlights at least vary in *intensity* proportional to activity level, while building presence is binary.

Goldblatt et al. (2020, WBER) directly address this limitation: "one would need additional information, such as the density and height of structures, to improve the prediction of economic activity based on daytime satellite imagery, because the landcover layers only record whether or not a man-made impervious structure is present." Their Vietnam study found that Landsat spectral bands outperform nightlights at predicting enterprise counts at the commune level, but the landcover-to-economics mapping remains noisy.

The proxying-economic-activity literature (Beyer et al., 2023, PNAS Nexus) finds that daytime satellite imagery can proxy economic activity, but notes that "while satellite imagery may reveal the spatial extent of markets, it may not reveal the intensity of economic activity."

---

## 2. What COULD Improve District-Level Correlation Beyond Pixel AUC?

The project's key insight is correct: pixel AUC is near its ceiling (~0.97), and the bottleneck is the district-level correlation. Improving r from 0.38-0.43 requires changing *what* the model predicts, not *how accurately* it predicts buildings.

### 2.1 Building Morphology Features (Not Just Count)

Instead of counting buildings, aggregate *building characteristics*:
- **Total building area** per district (sum of footprint areas). OpenStreetMap-based studies using building area achieve r-squared of 0.58-0.66 for wealth prediction at fine spatial scales (Lee and Braithwaite, 2020).
- **Building size distribution** (Gini of footprint areas, share of large vs. small buildings). Large buildings correlate with commercial/industrial use; small buildings with residential.
- **Building density gradients** (central vs. peripheral concentration within districts).
- **Morphological typology** (regular grid patterns suggest planned commercial areas; irregular patterns suggest organic residential growth).

These features require either Google Open Buildings polygons (which include area) or a model that outputs continuous building density/area rather than binary presence.

### 2.2 Spectral Signatures of Economic Activity

Economic activity generates ancillary physical signals beyond buildings:
- **Impervious surface fraction** (roads, parking lots, paved areas): strongly correlated with commercial intensity. Landsat's spectral bands can detect these (Goldblatt et al., 2020).
- **Roof material signatures**: corrugated iron (informal/residential) vs. concrete/commercial roofing have distinct NIR/SWIR signatures. This is already partially captured by the model's input bands but not explicitly leveraged.
- **Vegetation patterns**: commercial areas have less vegetation (lower NDVI). NDVI as an additional input band (Experiment 1 in the Alpha Earth plan) could help separate commercial from residential areas.
- **Road network density**: a strong proxy for market access and economic connectivity. Available from OSM and the Global Human Settlement Layer.

### 2.3 Nighttime Lights as Complementary Signal

Henderson, Storeygard, and Weil (2012, AER) established that nighttime light intensity has an elasticity of ~1.55 with respect to GDP in developing countries. VIIRS nighttime lights provide an *intensity* measure that complements the *extent* measure from building detection. However:
- Nightlights saturate in cities (top-coding problem).
- Nightlights are zero in many rural areas with real economic activity.
- Resolution (~500m for VIIRS) is far coarser than Sentinel-2 (10m).
- Nightlights are better for cross-sectional level estimation than for growth (R-squared ~ 0.10 for changes over time; World Bank, 2015).

A combined model using building density (extent) + nightlight intensity (intensity) could capture both margins. Yeh et al. (2020, Nature Communications) found that CNNs trained on nightlights alone and daytime imagery alone perform "similarly to each other and almost as well as the combined model," suggesting these signals contain overlapping but not identical information.

### 2.4 Alpha Earth Embeddings

Google's Satellite Embedding V1 (Alpha Earth) compresses multi-sensor time series (Sentinel-2, Sentinel-1, Landsat) into 64 learned features per pixel at 10m resolution. These embeddings are trained via self-supervised learning on diverse geospatial tasks and encode spatial and temporal semantics that raw spectral bands may not explicitly represent.

**Theoretical argument for AE improving district r:**
- AE embeddings capture *contextual* information: a building pixel surrounded by commercial infrastructure may have a different embedding than one surrounded by agricultural land, even if both have identical spectral signatures.
- AE is trained on temporal patterns (annual composites from multiple years), so it may encode *dynamism* (actively used areas vs. static structures).
- The AlphaEarth model integrates Sentinel-1 SAR (sensitive to structure/texture) alongside optical data, potentially capturing building material, height proxies, and urban morphology that S2 alone misses.
- In downstream benchmarks, AE embeddings fused with socioeconomic features improved prediction of FEMA's National Risk Index by an average of 11% in R-squared across 20 hazards (Google Research, 2025).

**Theoretical argument against:**
- AE embeddings are optimized for general geospatial representation, not economic measurement specifically. They may encode ecological and geological features that are irrelevant or confounding for economic prediction.
- 64 bands of learned features with no physical interpretation create a black-box concern for economics reviewers.
- If AE's training data overlaps heavily with Google Open Buildings (both use Sentinel-2 as input), the embeddings may not contain much information beyond what building detection already captures.

### 2.5 Temporal Change Detection

New construction is a strong signal of economic growth. A model trained on temporal pairs (year t vs. year t+k) could detect building *change* rather than building *level*. Google's Open Buildings 2.5D Temporal dataset now provides yearly building presence and height estimates from 2016-2023. This enables:
- Measuring construction rates per district as a growth proxy.
- Identifying areas of rapid densification (commercial development) vs. stable (residential).
- Correlating construction change with census change (2020 vs. 2023 establishment censuses).

### 2.6 Land Use Classification

Rather than binary building detection, a multi-class land use model could distinguish: commercial, residential, industrial, agricultural, institutional. Each class has different economic intensity. This transforms the problem from "how many buildings?" to "what kind of land use?" -- directly addressing the semantic bottleneck. Dynamic World (Google, available on GEE) provides 10m land use classification that could serve as either an additional input or a training target.

---

## 3. Benchmarks from the Literature

### 3.1 Key Performance Numbers

| Paper | Outcome Variable | Unit of Analysis | Metric | Value | Notes |
|-------|-----------------|-----------------|--------|-------|-------|
| Jean et al. (2016, Science) | Consumption expenditure & asset wealth | Village/cluster (~5km) | R-squared | **0.75** | 5 African countries; CNN transfer from nightlights |
| Yeh et al. (2020, Nat. Comms) | Asset wealth index | Village (~5km) | R-squared | **0.50-0.70** | 0.70 cross-sectional; 0.50 for temporal changes; publicly available Landsat+NTL |
| Henderson et al. (2012, AER) | GDP | Country-level | Elasticity | **1.55** | NTL-GDP elasticity in developing countries |
| Goldblatt et al. (2020, WBER) | Enterprise count, employment, expenditure | Commune (~district) | R-squared | **0.30-0.50** | Medium-resolution Landsat; Vietnam |
| Beyer et al. (2023, PNAS Nexus) | Regional economic activity | Subnational | Correlation | **0.60-0.80** | Daytime surface groups as activity proxy |
| Lee & Braithwaite (2020) | Asset wealth | Fine grid (115m cell) | R-squared | **0.58-0.66** | OSM building features + random forest |
| Asher, Novosad et al. (various) | Poverty, economic performance | District/subdistrict | Various | **0.30-0.60** | India; SHRUG platform; nightlights as proxy |
| **This project** | **Establishment count** | **District (n=30)** | **Pearson r** | **0.38-0.43** | **Rwanda; building count from U-Net** |

### 3.2 Interpretation of r = 0.38-0.43

An r of 0.38-0.43 corresponds to R-squared of 0.14-0.18 -- meaning building counts explain only 14-18% of the variance in establishment counts across Rwanda's 30 districts. This is *below* the typical range in the literature (R-squared 0.30-0.70 depending on method, resolution, and outcome variable).

However, critical context:
1. **n = 30 is very small.** With only 30 districts, the confidence interval on r is wide (roughly +/- 0.30 at 95% CI). The true population correlation could plausibly be anywhere from 0.10 to 0.65. Most papers in the literature use hundreds or thousands of spatial units.
2. **Establishment count is a harder target** than asset wealth or consumption expenditure. Asset wealth indices aggregate durable goods (roof material, water source) that have direct spectral signatures. Establishment counts measure firm presence, which has a weaker physical correlate.
3. **Binary count vs. continuous features.** The current approach counts building pixels per district. Papers achieving R-squared > 0.50 typically use CNN-extracted feature vectors (hundreds of learned features), not simple building counts. The comparison is between a hand-crafted single-variable predictor and a learned multi-feature predictor.
4. **The label source matters.** Google Open Buildings has known miss rates (60% recall in rural Rwanda). Systematic label errors propagate to district-level aggregates, attenuating the correlation.

### 3.3 What Constitutes "Good" Performance?

For a top-5 economics paper, the absolute R-squared matters less than:
- **Demonstrating improvement** over existing measures (nightlights, GHSL, existing poverty maps).
- **Documenting what the measure captures and what it misses** -- honest characterization of measurement error is valued.
- **Showing the measure is useful for answering an economic question** (causal identification, policy evaluation) -- not just prediction accuracy.

The measurement-paper template for top-5 journals (following Donaldson & Storeygard, 2016; Henderson et al., 2012) emphasizes: (a) construct the measure, (b) validate it against ground truth, (c) show it reveals something new that existing measures cannot capture, (d) demonstrate an application. An r of 0.40 is publishable if the measure fills a gap (10m resolution vs. 1km+ from nightlights) and enables analysis at spatial scales previously impossible.

---

## 4. Alpha Earth Embeddings: Detailed Assessment

### 4.1 What AE Is

Alpha Earth / Satellite Embedding V1 is a foundation model from Google DeepMind that processes multi-year time series of Sentinel-2, Sentinel-1, and Landsat imagery and outputs 64 learned features per 10m pixel. It is trained via self-supervised learning (likely masked image modeling or contrastive learning) on global coverage. The embeddings "compress vast Earth observation information into 64 features representing meaningful spatial and temporal semantics" (Google Earth Engine documentation).

The model is described as a "virtual satellite" that integrates diverse observation sources into a unified representation. Critically, it uses SAR (Sentinel-1) data alongside optical data, meaning it encodes structural information (building height, surface roughness, urban texture) that pure optical sensors miss.

### 4.2 Theoretical Case for Economic-Relevant Features

Raw Sentinel-2 bands measure *spectral reflectance* -- the physical interaction between sunlight and surfaces. AE embeddings encode *learned semantic features* that capture higher-order spatial patterns. The distinction matters for economic measurement:

- **Urban morphology encoding.** AE may learn to distinguish planned commercial districts (grid streets, large rectangular footprints, parking areas) from organic residential settlements (irregular, small footprints, vegetation interspersed). This morphological information is latent in raw spectral bands but requires complex spatial reasoning to extract -- exactly what a foundation model is trained to do.
- **Temporal activity signatures.** Annual AE composites may encode seasonal variation patterns that differ between active commercial areas (stable year-round) and agricultural areas (seasonal). This temporal signature is a proxy for economic structure.
- **Cross-sensor fusion.** By integrating SAR data, AE embeddings may capture building volume and surface texture features invisible to optical sensors. Multi-story buildings produce stronger SAR returns than single-story structures. Industrial facilities have distinct SAR signatures from residential areas.

### 4.3 Interpreting Experimental Outcomes

**Scenario 1: AE-only AUC ~ baseline, but district r improves.**
This would be the most informative outcome. It means AE embeddings detect buildings roughly as well as raw S2 bands (not surprising, since AE is trained on S2 as input), but the *way* they represent building-adjacent information captures economic-relevant variation. The building detection task is a bottleneck that both representations hit equally, but AE embeddings carry additional economic signal in their non-building features (land use context, infrastructure patterns, temporal dynamics). This would strongly support using AE as input for economic measurement.

**Scenario 2: AE-only AUC < baseline, but district r improves.**
This would be the most *surprising* and theoretically provocative outcome. It would mean AE embeddings are *worse* at pixel-level building detection (perhaps because their 64 dimensions are allocated to encode broader geospatial semantics rather than fine-grained building boundaries) but *better* at capturing the economic signal we actually care about. This would demonstrate that the building-detection task is the wrong intermediate objective -- that economic prediction benefits from a richer representation even at the cost of building detection accuracy. For a paper, this finding would be gold: it directly demonstrates the semantic bottleneck and motivates moving beyond building detection to learned representations.

**Scenario 3: AE-only AUC > baseline, but district r unchanged.**
AE is better at detecting buildings (plausible given richer training data) but the additional building pixels don't help at the district level. This confirms the semantic bottleneck: the problem is not detection accuracy but the building-to-economics mapping.

**Scenario 4: Dual-branch (S2+AE) improves both AUC and district r.**
The expected outcome if AE provides complementary information. The magnitude of the district-r improvement relative to the AUC improvement is the key diagnostic: if district r increases proportionally more than AUC, AE is contributing economic signal beyond building detection.

### 4.4 Concerns for Economics Reviewers

- **Black-box features.** AE's 64 bands have no physical interpretation. Reviewers may ask: "What exactly are these features measuring?" Unlike spectral bands (NIR = vegetation vigor, SWIR = moisture content), AE features are latent dimensions. Mitigation: report feature importance analysis, SHAP values, or PCA/t-SNE visualizations showing that AE features cluster by land-use type.
- **Reproducibility and stability.** AE is a Google product. If Google updates the model, embeddings change. This is a concern for reproducibility of economic research. Mitigation: specify exact version (V1 ANNUAL), document GEE asset ID, note that the dataset is CC-BY 4.0 and archived.
- **Circular validation risk.** If AE was trained partly on Google Open Buildings (which uses Sentinel-2 as input), and we use Open Buildings as labels, the AE features might trivially encode "building presence" rather than providing independent economic information. This needs careful discussion.

---

## 5. Recommendations for Improving the Pipeline

### 5.1 Continuous Output Instead of Binary Detection

The single highest-impact change: replace binary building presence with **continuous building density or building area fraction** per pixel. This converts the output from {0, 1} to [0, 1], where the value represents the fraction of the pixel covered by building footprint. Benefits:
- Captures intensive margin (a pixel 90% covered by a large warehouse vs. 10% covered by a small hut).
- Naturally aggregates to district-level building area rather than building count.
- Building *area* is a much better economic proxy than building *count* (Newhouse, 2024, JIDS).

Implementation: change the loss function from binary cross-entropy / focal loss to MSE or Huber loss; use Google Open Buildings polygon areas rasterized as continuous labels rather than binary.

### 5.2 Multi-Task Learning

Train the U-Net with a dual objective:
1. **Primary task:** building segmentation (pixel-level, using Open Buildings labels).
2. **Auxiliary task:** district-level establishment count regression (weak supervision).

The auxiliary task forces the learned representation to encode information relevant to economic outcomes, not just building geometry. This is architecturally similar to the approach in the multi-task poverty prediction literature (AAAI, 2018) which jointly predicted developmental statistics (roof material, lighting source, water source) and used them to estimate poverty.

Implementation: add a global average pooling + regression head after the U-Net encoder. During training, alternate between pixel-level segmentation loss and district-level regression loss (with appropriate loss weighting). The district-level signal is weak (n=30) but even a small gradient signal pointing the representation toward economic relevance could help.

### 5.3 Feature Aggregation Beyond Count

Even without retraining the model, the district-level correlation could improve by aggregating richer statistics from the existing predictions:
- **Predicted building probability distribution** per district (mean, std, skewness, kurtosis).
- **Spatial concentration index** (Moran's I of predicted building density within each district).
- **Urban-rural gradient** (ratio of high-density to low-density predicted pixels).
- **Predicted building area** (using Open Buildings polygon areas to weight pixels).

Use these as features in a simple regression (OLS or random forest) predicting establishment counts. This extracts more signal from the existing model without retraining.

### 5.4 Incorporate Height Information

Google Open Buildings 2.5D Temporal now provides estimated building height from Sentinel-2 time series. Building *volume* (footprint * height) is a far better economic proxy than footprint alone. Districts with tall commercial buildings would register as higher economic intensity. This is the most direct way to address the multi-story limitation.

### 5.5 Recommendations for the Economics Paper

**Framing.** Position the paper as a *measurement contribution* in the tradition of Henderson et al. (2012) and Donaldson and Storeygard (2016). The value proposition is not prediction accuracy per se, but *spatial resolution* (10m vs. 1km+) and *global scalability* (free Sentinel-2 data, trainable with existing census data).

**Honest limitations section.** Explicitly acknowledge:
- Building count is a noisy proxy for economic activity (cite r = 0.38-0.43).
- The informal economy is systematically undercounted.
- The measure captures the extensive margin (where activity exists) better than the intensive margin (how much activity occurs).
- Performance likely varies by urbanization level, economic structure, and region.

**Validation strategy.** Benchmark against:
1. Nighttime lights (VIIRS) at district level -- your 10m measure should outperform at fine spatial scales even if it underperforms at coarse scales.
2. GHSL built-up area fraction -- a simpler building-based proxy; show your model adds value.
3. Dynamic World land use -- test whether land use classification captures economic variation your building detector misses.
4. If Peru data arrives: validate at finer spatial scales where n >> 30 and confidence intervals are tighter.

**Application.** The measure is most convincing if used to answer an economic question: inequality within districts, market access effects, spatial agglomeration patterns. Show that 10m resolution reveals variation that 1km measures miss.

---

## 6. Summary of Key Takeaways

1. **The r = 0.38-0.43 gap is not a model failure but a proxy limitation.** Building count is a crude extensive-margin measure. The literature achieves R-squared 0.50-0.75 using learned CNN features (not simple counts) or continuous wealth indices (not establishment counts).

2. **The path to higher district correlation runs through richer representations, not better building detection.** Pixel AUC is near its ceiling. Improvements should focus on: continuous output (building area/density), multi-feature aggregation, height information, and learned embeddings (Alpha Earth).

3. **Alpha Earth embeddings are the most promising single intervention.** They encode multi-sensor, temporal, contextual information that raw S2 bands lack. The experiment plan is well-designed. The interpretability challenge is real but manageable.

4. **For a top-5 paper, the correlation coefficient matters less than the narrative.** Position as a measurement innovation (10m global resolution), honestly document limitations, benchmark against alternatives, and demonstrate an economic application.

5. **n = 30 districts is the binding statistical constraint.** With Peru data (finer spatial units, larger n), the same model could achieve both tighter confidence intervals and higher correlations. Rwanda validates the concept; Peru validates the measurement.

---

## References

- Baragwanath Vogel, K., Goldblatt, R., Hanson, G., Khandelwal, A. (2021). Detecting urban markets with satellite imagery: An application to India. *Journal of Urban Economics*, 125.
- Beyer, R., Hu, Y., Yao, J. (2023). Proxying economic activity with daytime satellite imagery. *PNAS Nexus*, 2(4).
- Burke, M., Driscoll, A., Lobell, D., Ermon, S. (2021). Using satellite imagery to understand and promote sustainable development. *Science*, 371(6535).
- Donaldson, D., Storeygard, A. (2016). The view from above: Applications of satellite data in economics. *Journal of Economic Perspectives*, 30(4), 171-198.
- Goldblatt, R., Stuber, G., Glickman, R., You, S., Hanson, G., Khandelwal, A. (2020). Can medium-resolution satellite imagery measure economic activity at small geographies? Evidence from Landsat in Vietnam. *World Bank Economic Review*, 34(3), 635-653.
- Google DeepMind (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data.
- Henderson, J.V., Storeygard, A., Weil, D. (2012). Measuring economic growth from outer space. *American Economic Review*, 102(2), 994-1028.
- Jean, N., Burke, M., Xie, M., Davis, W.M., Lobell, D., Ermon, S. (2016). Combining satellite imagery and machine learning to predict poverty. *Science*, 353(6301), 790-794.
- Lee, K., Braithwaite, J. (2020). High-resolution poverty maps in Sub-Saharan Africa. *arXiv:2009.00544*.
- Newhouse, D. (2024). Small area estimation of poverty and wealth using geospatial data: What have we learned so far? *Journal of International Development and Cooperation*, 30(1).
- Yeh, C., Perez, A., Driscoll, A., Azzari, G., Tang, Z., Lobell, D., Ermon, S., Burke, M. (2020). Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. *Nature Communications*, 11, 2583.
