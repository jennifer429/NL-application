# Patient Exam Relevance API

HTTP API that decides, for each previous radiology examination of a patient, whether it should be surfaced to the radiologist reading the current examination.

- Challenge: `relevant-priors-v1`
- Endpoint: `POST /predict`
- Health: `GET /health`
- Stack: Node 20 + TypeScript + Express + Zod
- Public-split accuracy: **94.06%** on 27,614 labeled priors across 996 cases
- Latency: full public split processed in **~270 ms** (well under the 360 s evaluator cutoff)

## How it works

The evaluator sends, per case, a `current_study` and a list of `prior_studies`. Each study carries only a `study_description` (free text, e.g. `"MRI BRAIN STROKE LIMITED WITHOUT CONTRAST"`) and a `study_date`. No reports, no structured codes, no demographics.

That constraint shapes the approach: a heuristic that **parses each `study_description` into structured fields** and then compares the parsed pairs against a small set of clinically motivated rules outperforms a logistic regression trained on the same features, and dodges every timeout/cost/availability risk that comes with an LLM in the hot path.

### Pipeline

1. **Parser** (`src/parse.ts`, `src/vocabulary.ts`). For each description, extract:
   - **Modality**: `CT`, `MRI`, `XR`, `US`, `MAMMO`, `NM`, `PET`, `ECHO`, `DXA`, `VAS`, `ANGIO`, `FLUORO`. CT/MRI angiography fall into the CT/MRI buckets rather than a separate `ANGIO` bucket. Plain-film descriptions with no modality token (e.g. `"LUMBAR SPINE, LATERAL VIEW ONLY"`, `"CHEST 1 VIEW"`) fall back to `XR` via a "views / frontal / AP-LAT" heuristic.
   - **Region set**: `BRAIN`, `HEAD`, `CHEST`, `ABDOMEN`, `PELVIS`, `BREAST`, `CSPINE`/`TSPINE`/`LSPINE`, `SHOULDER`/`ELBOW`/`WRIST`/`HAND`, `HIP`/`KNEE`/`ANKLE`/`FOOT`, `THYROID`/`CAROTID`/`KIDNEY`/`BLADDER`, `BONE` (whole-body scans), `HEART`, and ~30 others.
   - **Laterality** (`LT`, `RT`, `BI`) and **contrast** (`true`, `false`, `null`).

2. **Rules** (`src/rules.ts`).
   - **Exact normalized description match** ⇒ relevant.
   - Otherwise, require **modality compatibility**: same modality, or a whitelisted cross-modality pair (`CT↔MRI`, `CT↔XR`, `MRI↔XR`, `CT↔NM`, `MAMMO↔US`, `MAMMO↔NM` for sentinel lymphoscint, `ECHO↔CT/MRI/NM`, etc.).
   - **Region overlap** via a small equivalence map (`BRAIN↔HEAD`, `LIVER↔ABDOMEN`, `UTERUS↔PELVIS`, `PULMONARY↔CHEST`, …). Broader equivalences (e.g. `TSPINE↔CHEST`, `HEART↔CHEST`) were tested and *removed* because they cost more FPs than they recovered in TPs on the public split.
   - **Laterality mismatch** blocks a match unless one side is bilateral.
   - **Bone Scan wildcard**: `NM(BONE)` overlaps any region that a whole-body bone scan covers.
   - **Mammography implication**: any description parsed as `MAMMO` modality is also tagged with `BREAST` region, which recovers a long tail of `MAM SCREEN 3D`, `SEED LOC`, `COMBOHD`-style descriptions that don't spell out "breast".

3. **Request handling** (`src/server.ts`, `src/predict.ts`).
   - Zod-validated request, returns 400 on malformed input.
   - Deterministic, pure-function pipeline with an in-process LRU cache keyed on `(current_description, prior_description, current_date, prior_date)` so retries and duplicate pairs are instant.
   - Structured JSON logs with a per-request UUID, case count, prior count, latency, and cache hit-rate. No PHI-ish description content is logged.
   - Assertion: `predictions.length === Σ prior_studies.length`. Every prior gets exactly one prediction; skipping one counts as wrong under the accuracy rule, so there is no skipping path.

## Results

### Baseline ladder (public split, 27,614 labeled priors across 996 cases)

| Strategy | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Predict all true | 23.78% | 23.78% | 100.00% | 0.384 | 6567 | 0 | 21047 | 0 |
| Predict all false (base rate) | 76.22% | — | 0.00% | 0.000 | 0 | 21047 | 0 | 6567 |
| Exact normalized description match | 79.37% | 99.89% | 13.26% | 0.234 | 871 | 21046 | 1 | 5696 |
| Modality only (no region check) | 86.90% | 89.00% | 51.24% | 0.650 | 3365 | 20631 | 416 | 3202 |
| Modality + region | 87.51% | 97.33% | 48.82% | 0.650 | 3206 | 20959 | 88 | 3361 |
| **+ cross-modality pairs (shipped)** | **94.06%** | **93.27%** | **80.86%** | **0.866** | **5310** | **20664** | **383** | **1257** |
| + region adjacency graph | 91.74% | 80.33% | 86.43% | 0.833 | 5676 | 19657 | 1390 | 891 |
| Logistic regression, same features | 92.84% | — | — | — | 5285 | 20353 | 694 | 1282 |

### Slice analysis (shipped config)

**By modality pair (current → prior)** — samples with ≥500 pairs
| Pair | n | Accuracy |
|---|---:|---:|
| CT → CT | 1976 | 0.945 |
| XR → XR | 1814 | 0.938 |
| CT → XR | 1732 | 0.974 |
| MAMMO → MAMMO | 1380 | 0.979 |
| XR → CT | 1340 | 0.935 |
| MRI → XR | 953 | 0.965 |
| CT → MRI | 818 | 0.966 |
| XR → MRI | 807 | 0.970 |
| MRI → CT | 727 | 0.889 |
| CT → MAMMO | 781 | 1.000 |
| MRI → MRI | 584 | 0.913 |

**By time delta**
| Δ | n | Accuracy |
|---|---:|---:|
| < 1 month | 2149 | 0.848 |
| 1–12 months | 4210 | 0.938 |
| 1–3 years | 6493 | 0.940 |
| 3–5 years | 2917 | 0.945 |
| > 5 years | 11845 | 0.958 |

Recent priors (<1 month) are harder than old ones — same-day/next-day follow-up scans of unrelated body parts look more confusing than year-old mismatches.

**By label class**
| Class | n | Accuracy |
|---|---:|---:|
| positives | 6567 | 0.809 |
| negatives | 21047 | 0.982 |

The system is conservative. High precision (93.3%), moderate recall (80.9%). In a radiologist-facing UX, missing a relevant prior is usually less disruptive than surfacing an irrelevant one, so the skew is acceptable for v1.

### Experiments that didn't pay off

- **Region adjacency graph** (`elbow↔shoulder↔wrist`, `hip↔pelvis↔lspine`, etc.). Intuitively clinically useful; empirically it added ~4× more FPs (990 extra) than TPs (366 extra) on the public split, costing 2.3 accuracy points. Retained as a tunable flag (`allowAdjacentRegion`) — may be worth re-enabling if the private split distribution differs.
- **Broad region equivalences** (`TSPINE↔CHEST`, `HEART↔CHEST`, `CAROTID↔HEAD`, `LSPINE↔ABDOMEN`). These recovered some FNs but generated a larger set of FPs (e.g. CT heart predicting relevance against chest X-ray priors that the labels mark as irrelevant). Removed.
- **Same-modality + one side has unknown region ⇒ relevant**. Recovered 158 TPs but cost 228 FPs.
- **Logistic regression on the same feature set** (`exact_match`, `modality_same`, `modality_compat`, `region_overlap`, `region_adjacent`, `both_unknown_region`, `laterality_conflict`, `days_years_scaled`). 92.84% — 1.22 points behind the hand-tuned rules. That the hand rules beat LR on the same features is the strongest evidence that the remaining error is not a thresholding problem; it requires richer features.

## API contract

### POST /predict

```json
{
  "challenge_id": "relevant-priors-v1",
  "schema_version": 1,
  "generated_at": "2026-04-16T12:00:00.000Z",
  "cases": [
    {
      "case_id": "1001016",
      "patient_id": "606707",
      "patient_name": "Andrews, Micheal",
      "current_study": {
        "study_id": "3100042",
        "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        "study_date": "2026-03-08"
      },
      "prior_studies": [
        { "study_id": "2453245", "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST", "study_date": "2020-03-08" },
        { "study_id": "992654",  "study_description": "CT HEAD WITHOUT CNTRST",                "study_date": "2021-03-08" }
      ]
    }
  ]
}
```

Response (one prediction per prior, always):
```json
{
  "predictions": [
    { "case_id": "1001016", "study_id": "2453245", "predicted_is_relevant": true },
    { "case_id": "1001016", "study_id": "992654",  "predicted_is_relevant": false }
  ]
}
```

### GET /health

Returns `{ "ok": true, "cache": {...} }` for platform health checks.

## Engineering

- **Bulk inference.** The handler iterates cases, iterates priors per case. Pure-function per pair, no network in the hot path. Full public split (27,614 priors) in ~270 ms on a single shared-CPU Fly.io machine.
- **Every prior gets a prediction.** Enforced by assertion before responding; guards against silent drops.
- **Cache.** In-process LRU keyed on `(current_description, prior_description, current_date, prior_date)`, capacity 1,000,000. Evaluator retries and duplicate pairs across a batch become O(1). Hit rate is logged per request.
- **Validation.** Zod schema rejects malformed requests with 400. Extra/unknown fields are ignored so evaluator schema bumps don't break us.
- **Observability.** One structured JSON log per lifecycle event (`predict.start`, `predict.done`, `predict.bad_request`) with `request_id`, case count, total priors, elapsed ms, cache stats.
- **Tests.** Vitest suite covers parser cases from the real descriptions seen in the public split (`CERVICL` typo, `KNEE,` trailing comma, `MAM` abbreviations), the core rules (exact match, cross-modality, laterality conflict, DXA isolation, adjacency tunable), and the end-to-end contract (every prior gets a prediction, response validates).

## Dev scripts

These power the write-up and let you iterate without touching the server:

- `npm run score` — overall accuracy + confusion matrix against the public split.
- `npm run benchmark` — the full baseline ladder table.
- `npm run analyze` — top failure patterns + random samples of FPs and FNs.
- `npm run slice` — accuracy sliced by modality pair / region pair / time-delta bucket / label class.
- `npm run suggest` — fits a logistic regression on the same features for calibration.

## Run locally

```sh
npm install
npm test
npm run score          # expects data/public.json to contain cases + truth keys
npm run dev            # listens on :8080
```

## Deploy (Fly.io)

```sh
flyctl launch --no-deploy     # answer defaults; fly.toml is already present
flyctl deploy
curl https://<your-app>.fly.dev/health
```

## Next improvements

Graded by expected accuracy lift × engineering cost on the 1257 remaining FNs + 383 FPs:

1. **Breast-context inference on bare US descriptions.** The top FN pattern (66 cases) is `MAMMO(BREAST) ↔ US(-)` where the ultrasound description is `"ULTRASOUND BILAT SCREEN COMP"` or similar. A narrow rule — "US modality + `BILAT` + `SCREEN` ⇒ add `BREAST` region" — would recover most of these without the broad-region regression we saw earlier.
2. **Batched LLM second pass for ambiguous pairs only.** Restrict to pairs where rules are uncertain (one side has `UNKNOWN` modality or zero regions, plus same-modality same-date), batch all such pairs in a single Claude / GPT call per request. The brief explicitly warns per-prior calls time out; per-request-batched should cost one LLM call per request and target only ~5% of pairs.
3. **Sentence-embedding similarity as a feature.** Add `embedding_cosine(current_desc, prior_desc)` from a small model loaded at startup. Use it as a tiebreaker when rule outputs are ambiguous, not as a standalone classifier. Would need benchmarking for load-time and memory on Fly's 512 MB VM.
4. **Learn a cross-modality pair table from the labels** rather than hand-whitelisting. With 27,614 labels and ~100 unique modality pairs, a per-pair prior probability + Laplace smoothing would replace the hand-coded `CROSS_MODALITY_PAIRS` list with empirical evidence. Probably worth ~0.5 points.
5. **Richer vocabulary.** Walk every unique prior description in the public split, bucket the ones currently parsed as `UNKNOWN(-)`, add rules for the clusters (e.g. `ComboHD`, `FFR`, `scoliosis srvy`, `Intrprtn outsd`). The analyze-errors output already names the top offenders.
6. **Confidence calibration.** Internal score is currently binary. A confidence field on each prediction would enable a radiologist UX where borderline priors surface in a secondary list rather than be hidden outright — useful even though the challenge scores only `true`/`false`.

## Limitations

- Signals are limited to two fields per exam. With radiology reports or structured codes (CPT, ICD-10), a much richer representation is possible; none of that is available in this challenge.
- The vocabulary is tuned against 996 public cases from a specific practice. A different health system's description conventions (e.g. more verbose, different abbreviations) will degrade accuracy until the vocabulary is extended.
- Hand-tuned rules have a ceiling. The 94%→100% gap is probably closable only with report text or with an LLM pass — both out of scope for v1.
