import type { Request, Response, Case } from "./types.ts";
import { parse } from "./parse.ts";
import { DEFAULT_CONFIG, daysBetween, isRelevant, type RuleConfig } from "./rules.ts";
import { LRUCache } from "./cache.ts";

type PairKey = string;

export function predict(
  req: Request,
  cfg: RuleConfig = DEFAULT_CONFIG,
  cache: LRUCache<boolean> = new LRUCache<boolean>(),
): Response {
  const predictions: Response["predictions"] = [];

  for (const c of req.cases) {
    const cur = parse(c.current_study.study_description);
    for (const p of c.prior_studies) {
      const key = cacheKey(c, p);
      let hit = cache.get(key);
      if (hit === undefined) {
        const prior = parse(p.study_description);
        const days = daysBetween(c.current_study.study_date, p.study_date);
        hit = isRelevant(cur, prior, days, cfg);
        cache.set(key, hit);
      }
      predictions.push({
        case_id: c.case_id,
        study_id: p.study_id,
        predicted_is_relevant: hit,
      });
    }
  }

  return { predictions };
}

function cacheKey(c: Case, p: Case["prior_studies"][number]): PairKey {
  return `${c.current_study.study_description}${p.study_description}${c.current_study.study_date}${p.study_date}`;
}
