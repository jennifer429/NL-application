import {
  modalitiesCompatible,
  regionsAdjacent,
  regionsEquivalent,
  isBoneScanCompatible,
  type Region,
} from "./vocabulary.ts";
import type { Parsed } from "./parse.ts";

export type RuleConfig = {
  allowCrossModality: boolean;
  allowAdjacentRegion: boolean;
  exactMatchAlwaysRelevant: boolean;
  sameModalitySameRegionRelevant: boolean;
  sameModalityUnknownRegionRelevant: boolean;
  ignoreLateralityMismatch: boolean;
  maxDaysRelevant: number | null;
};

export const DEFAULT_CONFIG: RuleConfig = {
  allowCrossModality: true,
  allowAdjacentRegion: false,
  exactMatchAlwaysRelevant: true,
  sameModalitySameRegionRelevant: true,
  sameModalityUnknownRegionRelevant: false,
  ignoreLateralityMismatch: false,
  maxDaysRelevant: null,
};

export function daysBetween(a: string, b: string): number {
  const da = new Date(a).getTime();
  const db = new Date(b).getTime();
  if (Number.isNaN(da) || Number.isNaN(db)) return 0;
  return Math.abs(da - db) / (1000 * 60 * 60 * 24);
}

function regionsOverlap(a: Region[], b: Region[], allowAdjacent: boolean): boolean {
  if (a.length === 0 && b.length === 0) return true;
  if (a.length === 0 || b.length === 0) return false;
  const aHasBone = a.includes("BONE");
  const bHasBone = b.includes("BONE");
  if (aHasBone && b.some((r) => isBoneScanCompatible(r))) return true;
  if (bHasBone && a.some((r) => isBoneScanCompatible(r))) return true;
  for (const x of a) {
    for (const y of b) {
      if (regionsEquivalent(x, y)) return true;
      if (allowAdjacent && regionsAdjacent(x, y)) return true;
    }
  }
  return false;
}

function lateralityConflict(
  a: Parsed["laterality"],
  b: Parsed["laterality"],
): boolean {
  if (a === null || b === null) return false;
  if (a === "BI" || b === "BI") return false;
  return a !== b;
}

export function isRelevant(
  current: Parsed,
  prior: Parsed,
  daysApart: number,
  cfg: RuleConfig = DEFAULT_CONFIG,
): boolean {
  if (cfg.exactMatchAlwaysRelevant && current.normalized === prior.normalized) {
    return true;
  }

  if (cfg.maxDaysRelevant !== null && daysApart > cfg.maxDaysRelevant) {
    return false;
  }

  if (!cfg.ignoreLateralityMismatch) {
    if (lateralityConflict(current.laterality, prior.laterality)) return false;
  }

  const modalityOK = cfg.allowCrossModality
    ? modalitiesCompatible(current.modality, prior.modality)
    : current.modality === prior.modality;
  if (!modalityOK) return false;

  if (
    cfg.sameModalityUnknownRegionRelevant &&
    current.modality === prior.modality &&
    (current.regions.length === 0 || prior.regions.length === 0)
  ) {
    return true;
  }

  const regionOK = regionsOverlap(
    current.regions,
    prior.regions,
    cfg.allowAdjacentRegion,
  );
  if (!regionOK) return false;

  if (!cfg.sameModalitySameRegionRelevant && current.modality === prior.modality) {
    return false;
  }

  return true;
}
