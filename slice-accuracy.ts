import { loadPublic, labelIndex } from "./_data.ts";
import { parse } from "../src/parse.ts";
import { DEFAULT_CONFIG, daysBetween, isRelevant } from "../src/rules.ts";

function bucketDays(d: number): string {
  if (d <= 31) return "<1mo";
  if (d <= 365) return "1-12mo";
  if (d <= 365 * 3) return "1-3yr";
  if (d <= 365 * 5) return "3-5yr";
  return ">5yr";
}

function main() {
  const ds = loadPublic();
  const labels = labelIndex(ds.truth);

  const byMod = new Map<string, { correct: number; total: number }>();
  const byRegion = new Map<string, { correct: number; total: number }>();
  const byDays = new Map<string, { correct: number; total: number }>();
  const byLabel = { pos: { correct: 0, total: 0 }, neg: { correct: 0, total: 0 } };

  const bump = (m: Map<string, { correct: number; total: number }>, k: string, ok: boolean) => {
    let v = m.get(k);
    if (!v) { v = { correct: 0, total: 0 }; m.set(k, v); }
    v.total++; if (ok) v.correct++;
  };

  for (const c of ds.cases) {
    const cur = parse(c.current_study.study_description);
    for (const p of c.prior_studies) {
      const actual = labels.get(`${c.case_id}|${p.study_id}`);
      if (actual === undefined) continue;
      const pr = parse(p.study_description);
      const days = daysBetween(c.current_study.study_date, p.study_date);
      const pred = isRelevant(cur, pr, days, DEFAULT_CONFIG);
      const ok = pred === actual;

      bump(byMod, `${cur.modality}→${pr.modality}`, ok);
      const reg = (cur.regions[0] ?? "OTHER") + "→" + (pr.regions[0] ?? "OTHER");
      bump(byRegion, reg, ok);
      bump(byDays, bucketDays(days), ok);
      (actual ? byLabel.pos : byLabel.neg).total++;
      if (ok) (actual ? byLabel.pos : byLabel.neg).correct++;
    }
  }

  const print = (title: string, m: Map<string, { correct: number; total: number }>, topN = 15) => {
    console.log(`\n${title}`);
    const rows = [...m.entries()].sort((a, b) => b[1].total - a[1].total).slice(0, topN);
    for (const [k, v] of rows) {
      const acc = (v.correct / v.total).toFixed(3);
      console.log(`  ${k.padEnd(28)}  n=${String(v.total).padStart(5)}  acc=${acc}`);
    }
  };

  print("BY MODALITY PAIR (current→prior)", byMod, 20);
  print("BY FIRST-REGION PAIR (current→prior)", byRegion, 20);
  print("BY TIME DELTA", byDays);
  console.log(`\nBY LABEL:`);
  console.log(`  positives acc=${(byLabel.pos.correct / byLabel.pos.total).toFixed(3)}  n=${byLabel.pos.total}`);
  console.log(`  negatives acc=${(byLabel.neg.correct / byLabel.neg.total).toFixed(3)}  n=${byLabel.neg.total}`);
}

main();
