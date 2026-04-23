import fs from "node:fs";
import path from "node:path";
import type { Request } from "../src/types.ts";

export type Label = { case_id: string; study_id: string; is_relevant_to_current: boolean };
export type PublicDataset = Request & { truth: Label[] };

export function loadPublic(file?: string): PublicDataset {
  const p = file ?? path.join(process.cwd(), "data", "public.json");
  if (!fs.existsSync(p)) {
    throw new Error(
      `Public dataset not found at ${p}. Place it under data/public.json (cases + truth keys).`,
    );
  }
  const raw = JSON.parse(fs.readFileSync(p, "utf8"));
  if (!raw.cases || !raw.truth) {
    throw new Error(`Expected both 'cases' and 'truth' in ${p}`);
  }
  return raw as PublicDataset;
}

export function labelIndex(truth: Label[]): Map<string, boolean> {
  const m = new Map<string, boolean>();
  for (const t of truth) m.set(`${t.case_id}|${t.study_id}`, t.is_relevant_to_current);
  return m;
}
