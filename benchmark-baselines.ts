import { loadPublic, labelIndex } from "./_data.ts";
import { parse } from "../src/parse.ts";
import { DEFAULT_CONFIG, daysBetween, isRelevant, type RuleConfig } from "../src/rules.ts";
import type { Case } from "../src/types.ts";

type Predictor = (c: Case, p: Case["prior_studies"][number]) => boolean;

function run(name: string, pred: Predictor, cases: Case[], labels: Map<string, boolean>) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const c of cases) {
    for (const pr of c.prior_studies) {
      const actual = labels.get(`${c.case_id}|${pr.study_id}`);
      if (actual === undefined) continue;
      const predv = pred(c, pr);
      if (predv && actual) tp++;
      else if (!predv && !actual) tn++;
      else if (predv && !actual) fp++;
      else fn++;
    }
  }
  const total = tp + tn + fp + fn;
  const acc = (tp + tn) / total;
  const prec = tp / Math.max(tp + fp, 1);
  const rec = tp / Math.max(tp + fn, 1);
  const f1 = (2 * prec * rec) / Math.max(prec + rec, 1e-9);
  return { name, accuracy: +acc.toFixed(4), precision: +prec.toFixed(4), recall: +rec.toFixed(4), f1: +f1.toFixed(4), tp, tn, fp, fn };
}

function withConfig(cfg: RuleConfig): Predictor {
  return (c, p) => {
    const cur = parse(c.current_study.study_description);
    const pr = parse(p.study_description);
    const d = daysBetween(c.current_study.study_date, p.study_date);
    return isRelevant(cur, pr, d, cfg);
  };
}

function main() {
  const ds = loadPublic();
  const labels = labelIndex(ds.truth);
  const cases = ds.cases;

  const rows: ReturnType<typeof run>[] = [];

  rows.push(run("all_true", () => true, cases, labels));
  rows.push(run("all_false", () => false, cases, labels));
  rows.push(run("exact_match",
    (c, p) => c.current_study.study_description.trim().toUpperCase() ===
              p.study_description.trim().toUpperCase(),
    cases, labels));

  rows.push(run("modality_only", withConfig({
    allowCrossModality: false,
    allowAdjacentRegion: false,
    exactMatchAlwaysRelevant: true,
    sameModalitySameRegionRelevant: true,
    sameModalityUnknownRegionRelevant: true,
    ignoreLateralityMismatch: true,
    maxDaysRelevant: null,
  }), cases, labels));

  rows.push(run("modality_plus_region", withConfig({
    allowCrossModality: false,
    allowAdjacentRegion: false,
    exactMatchAlwaysRelevant: true,
    sameModalitySameRegionRelevant: true,
    sameModalityUnknownRegionRelevant: false,
    ignoreLateralityMismatch: false,
    maxDaysRelevant: null,
  }), cases, labels));

  rows.push(run("+cross_modality", withConfig({
    allowCrossModality: true,
    allowAdjacentRegion: false,
    exactMatchAlwaysRelevant: true,
    sameModalitySameRegionRelevant: true,
    sameModalityUnknownRegionRelevant: false,
    ignoreLateralityMismatch: false,
    maxDaysRelevant: null,
  }), cases, labels));

  rows.push(run("+adjacent_region", withConfig({
    allowCrossModality: true,
    allowAdjacentRegion: true,
    exactMatchAlwaysRelevant: true,
    sameModalitySameRegionRelevant: true,
    sameModalityUnknownRegionRelevant: false,
    ignoreLateralityMismatch: false,
    maxDaysRelevant: null,
  }), cases, labels));

  rows.push(run("default_cfg", withConfig(DEFAULT_CONFIG), cases, labels));

  const header = ["name", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"];
  const col = (s: string, n: number) => s.padEnd(n);
  console.log(col("baseline", 24) + header.slice(1).map((h) => col(h, 10)).join(""));
  for (const r of rows) {
    console.log(
      col(r.name, 24) +
      col(String(r.accuracy), 10) +
      col(String(r.precision), 10) +
      col(String(r.recall), 10) +
      col(String(r.f1), 10) +
      col(String(r.tp), 10) +
      col(String(r.tn), 10) +
      col(String(r.fp), 10) +
      col(String(r.fn), 10),
    );
  }
}

main();
