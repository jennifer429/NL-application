import { findModality, findRegions, type Modality, type Region } from "./vocabulary.ts";

export type Parsed = {
  raw: string;
  normalized: string;
  modality: Modality;
  regions: Region[];
  laterality: "LT" | "RT" | "BI" | null;
  withContrast: boolean | null;
  screening: boolean;
};

export function normalize(s: string): string {
  return s
    .toUpperCase()
    .replace(/[._/]/g, " ")
    .replace(/[^A-Z0-9 \-&]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function findLaterality(text: string): Parsed["laterality"] {
  if (/\bBI(LATERAL)?\b|\bBIL\b|\bBOTH\b/.test(text)) return "BI";
  if (/\bLEFT\b|\bLT\b|\bL\b(?!\s*SPINE|\s*1|\s*2|\s*3|\s*4|\s*5)/.test(text)) return "LT";
  if (/\bRIGHT\b|\bRT\b|\bR\b(?!\s*SPINE|\s*1|\s*2|\s*3|\s*4|\s*5)/.test(text)) return "RT";
  return null;
}

function findContrast(text: string): boolean | null {
  if (/\bWO[/ ]?W\s*(CONTRAST|CNTRST|CON)\b|\bWITH AND WITHOUT\b/.test(text)) return true;
  if (/\bWITH\s*(CONTRAST|CNTRST|CON)\b|\bW\s*(CONTRAST|CNTRST|CON)\b/.test(text)) return true;
  if (/\bWITHOUT\s*(CONTRAST|CNTRST|CON)\b|\bWO\s*(CONTRAST|CNTRST|CON)\b/.test(text)) return false;
  return null;
}

function isScreening(text: string): boolean {
  return /\bSCREEN(ING)?\b/.test(text);
}

export function parse(description: string): Parsed {
  const normalized = normalize(description);
  const modality = findModality(normalized);
  const regions = findRegions(normalized);
  // Mammography implies breast even when the description drops the word.
  if (modality === "MAMMO" && !regions.includes("BREAST")) regions.push("BREAST");
  return {
    raw: description,
    normalized,
    modality,
    regions,
    laterality: findLaterality(normalized),
    withContrast: findContrast(normalized),
    screening: isScreening(normalized),
  };
}
