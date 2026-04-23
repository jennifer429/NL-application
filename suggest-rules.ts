import { loadPublic, labelIndex } from "./_data.ts";
import { parse } from "../src/parse.ts";
import { modalitiesCompatible, regionsAdjacent } from "../src/vocabulary.ts";
import { daysBetween } from "../src/rules.ts";

type Row = { x: number[]; y: number };

function features(current: string, prior: string, days: number): number[] {
  const cur = parse(current);
  const pr = parse(prior);
  const exact = current.trim().toUpperCase() === prior.trim().toUpperCase() ? 1 : 0;
  const modalitySame = cur.modality === pr.modality ? 1 : 0;
  const modalityCompat = modalitiesCompatible(cur.modality, pr.modality) ? 1 : 0;
  const regionAny = cur.regions.some((r) => pr.regions.includes(r)) ? 1 : 0;
  const regionAdjacent = cur.regions.some((r) => pr.regions.some((q) => regionsAdjacent(r, q))) ? 1 : 0;
  const bothUnknownRegion = cur.regions.length === 0 && pr.regions.length === 0 ? 1 : 0;
  const laterConflict =
    cur.laterality && pr.laterality && cur.laterality !== "BI" && pr.laterality !== "BI" && cur.laterality !== pr.laterality ? 1 : 0;
  const dYears = Math.min(days / 365, 10) / 10;

  return [1, exact, modalitySame, modalityCompat, regionAny, regionAdjacent, bothUnknownRegion, laterConflict, dYears];
}

const FEATURE_NAMES = [
  "bias",
  "exact_match",
  "modality_same",
  "modality_compat",
  "region_overlap",
  "region_adjacent",
  "both_unknown_region",
  "laterality_conflict",
  "days_years_scaled",
];

function sigmoid(z: number): number { return 1 / (1 + Math.exp(-z)); }

function train(rows: Row[], steps = 400, lr = 0.3): number[] {
  const d = rows[0].x.length;
  const w = new Array(d).fill(0);
  for (let step = 0; step < steps; step++) {
    const g = new Array(d).fill(0);
    for (const r of rows) {
      let z = 0;
      for (let i = 0; i < d; i++) z += w[i] * r.x[i];
      const p = sigmoid(z);
      const err = p - r.y;
      for (let i = 0; i < d; i++) g[i] += err * r.x[i];
    }
    for (let i = 0; i < d; i++) w[i] -= (lr * g[i]) / rows.length;
  }
  return w;
}

function main() {
  const ds = loadPublic();
  const labels = labelIndex(ds.truth);
  const rows: Row[] = [];
  for (const c of ds.cases) {
    for (const p of c.prior_studies) {
      const y = labels.get(`${c.case_id}|${p.study_id}`);
      if (y === undefined) continue;
      const d = daysBetween(c.current_study.study_date, p.study_date);
      rows.push({ x: features(c.current_study.study_description, p.study_description, d), y: y ? 1 : 0 });
    }
  }

  const w = train(rows);

  console.log("LEARNED WEIGHTS (logistic regression on public labels):");
  for (let i = 0; i < w.length; i++) {
    console.log(`  ${FEATURE_NAMES[i].padEnd(22)}  ${w[i].toFixed(4)}`);
  }

  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const r of rows) {
    let z = 0;
    for (let i = 0; i < w.length; i++) z += w[i] * r.x[i];
    const p = sigmoid(z) >= 0.5 ? 1 : 0;
    if (p === 1 && r.y === 1) tp++;
    else if (p === 0 && r.y === 0) tn++;
    else if (p === 1 && r.y === 0) fp++;
    else fn++;
  }
  const acc = (tp + tn) / (tp + tn + fp + fn);
  console.log(`\nLogistic model public accuracy @0.5: ${acc.toFixed(4)}  (tp=${tp} tn=${tn} fp=${fp} fn=${fn})`);
}

main();
