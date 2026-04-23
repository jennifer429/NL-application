import { loadPublic, labelIndex } from "./_data.ts";
import { parse } from "../src/parse.ts";
import { DEFAULT_CONFIG, daysBetween, isRelevant } from "../src/rules.ts";

type ErrRow = {
  kind: "FP" | "FN";
  current: string;
  prior: string;
  current_mod: string;
  prior_mod: string;
  current_regions: string;
  prior_regions: string;
  days: number;
};

function main() {
  const ds = loadPublic();
  const labels = labelIndex(ds.truth);
  const fps: ErrRow[] = [];
  const fns: ErrRow[] = [];
  const patternCount = new Map<string, number>();

  for (const c of ds.cases) {
    const cur = parse(c.current_study.study_description);
    for (const p of c.prior_studies) {
      const actual = labels.get(`${c.case_id}|${p.study_id}`);
      if (actual === undefined) continue;
      const pr = parse(p.study_description);
      const days = daysBetween(c.current_study.study_date, p.study_date);
      const pred = isRelevant(cur, pr, days, DEFAULT_CONFIG);
      if (pred === actual) continue;
      const kind = pred ? "FP" : "FN";
      const row: ErrRow = {
        kind,
        current: c.current_study.study_description,
        prior: p.study_description,
        current_mod: cur.modality,
        prior_mod: pr.modality,
        current_regions: cur.regions.join(",") || "-",
        prior_regions: pr.regions.join(",") || "-",
        days: Math.round(days),
      };
      (kind === "FP" ? fps : fns).push(row);
      const key = `${kind}: ${cur.modality}(${row.current_regions}) vs ${pr.modality}(${row.prior_regions})`;
      patternCount.set(key, (patternCount.get(key) ?? 0) + 1);
    }
  }

  console.log(`total FP: ${fps.length}   total FN: ${fns.length}\n`);
  console.log("TOP 20 FAILURE PATTERNS:");
  const patterns = [...patternCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20);
  for (const [k, v] of patterns) console.log(`  ${String(v).padStart(5)}  ${k}`);

  console.log("\n20 RANDOM FALSE POSITIVES:");
  for (const r of sample(fps, 20)) {
    console.log(`  FP | ${r.current_mod}(${r.current_regions}) "${r.current}"  <->  ${r.prior_mod}(${r.prior_regions}) "${r.prior}"  [${r.days}d]`);
  }

  console.log("\n20 RANDOM FALSE NEGATIVES:");
  for (const r of sample(fns, 20)) {
    console.log(`  FN | ${r.current_mod}(${r.current_regions}) "${r.current}"  <->  ${r.prior_mod}(${r.prior_regions}) "${r.prior}"  [${r.days}d]`);
  }
}

function sample<T>(arr: T[], n: number): T[] {
  const copy = arr.slice();
  const out: T[] = [];
  while (out.length < n && copy.length) {
    out.push(copy.splice(Math.floor(Math.random() * copy.length), 1)[0]);
  }
  return out;
}

main();
