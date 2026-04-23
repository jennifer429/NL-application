import { loadPublic, labelIndex } from "./_data.ts";
import { predict } from "../src/predict.ts";
import { DEFAULT_CONFIG } from "../src/rules.ts";

function main() {
  const ds = loadPublic();
  const labels = labelIndex(ds.truth);

  const t0 = Date.now();
  const out = predict({ cases: ds.cases }, DEFAULT_CONFIG);
  const ms = Date.now() - t0;

  let tp = 0, tn = 0, fp = 0, fn = 0, missing = 0;
  for (const p of out.predictions) {
    const actual = labels.get(`${p.case_id}|${p.study_id}`);
    if (actual === undefined) { missing++; continue; }
    if (p.predicted_is_relevant && actual) tp++;
    else if (!p.predicted_is_relevant && !actual) tn++;
    else if (p.predicted_is_relevant && !actual) fp++;
    else fn++;
  }
  const total = tp + tn + fp + fn;
  const accuracy = (tp + tn) / total;
  const precision = tp / Math.max(tp + fp, 1);
  const recall = tp / Math.max(tp + fn, 1);
  const f1 = (2 * precision * recall) / Math.max(precision + recall, 1e-9);

  console.log(JSON.stringify({
    cases: ds.cases.length,
    total_priors: total,
    missing,
    ms,
    accuracy: +accuracy.toFixed(4),
    precision: +precision.toFixed(4),
    recall: +recall.toFixed(4),
    f1: +f1.toFixed(4),
    confusion: { tp, tn, fp, fn },
  }, null, 2));
}

main();
