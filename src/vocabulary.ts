export type Modality =
  | "CT"
  | "MRI"
  | "XR"
  | "US"
  | "MAMMO"
  | "NM"
  | "PET"
  | "ECHO"
  | "DXA"
  | "VAS"
  | "FLUORO"
  | "ANGIO"
  | "UNKNOWN";

export type Region =
  | "BRAIN"
  | "HEAD"
  | "FACE"
  | "NECK"
  | "CSPINE"
  | "TSPINE"
  | "LSPINE"
  | "SPINE"
  | "CHEST"
  | "HEART"
  | "ABDOMEN"
  | "PELVIS"
  | "BREAST"
  | "SHOULDER"
  | "CLAVICLE"
  | "ELBOW"
  | "FOREARM"
  | "WRIST"
  | "HAND"
  | "FINGER"
  | "HIP"
  | "FEMUR"
  | "KNEE"
  | "TIBIA"
  | "ANKLE"
  | "FOOT"
  | "TOE"
  | "KIDNEY"
  | "BLADDER"
  | "PROSTATE"
  | "THYROID"
  | "CAROTID"
  | "LEG"
  | "ARM"
  | "AORTA"
  | "BONE"
  | "PULMONARY"
  | "LIVER"
  | "RECTUM"
  | "UTERUS"
  | "OTHER";

type Rule = { pattern: RegExp; value: Modality | Region };

export const MODALITY_RULES: Rule[] = [
  { pattern: /\bPET[ \-/]?CT\b/i, value: "PET" },
  { pattern: /\bPET\b/i, value: "PET" },
  // CT/MRI angiography — treat as the base CT/MRI modality, not a separate bucket.
  { pattern: /\bCT\s*ANGIO/i, value: "CT" },
  { pattern: /\bCTA\b/i, value: "CT" },
  { pattern: /\bMR\s*ANGIO/i, value: "MRI" },
  { pattern: /\bMRA\b/i, value: "MRI" },
  { pattern: /\bCT\b/i, value: "CT" },
  { pattern: /\bMRI\b/i, value: "MRI" },
  { pattern: /\bMR\b(?!I)/i, value: "MRI" },
  { pattern: /\bECHO\b|echocardiogra/i, value: "ECHO" },
  { pattern: /\bNM\b/i, value: "NM" },
  { pattern: /SPECT/i, value: "NM" },
  { pattern: /BONE\s*SCAN/i, value: "NM" },
  { pattern: /\bmyo\s*perf/i, value: "NM" },
  { pattern: /lymphoscint/i, value: "NM" },
  { pattern: /\bDXA\b|DEXA|bone\s*density/i, value: "DXA" },
  // MAM(MO)?(GRAPHY)? + tomo + digital combo screeners + seed localization
  {
    pattern:
      /\bMAMM?O?\b|MAMMOGR|\bTOMO\b|\bMG\b|mammo|SEED\s*LOC|SCREEN(ING)?\s*3D|STANDARD\s*SCREEN(ING)?|DIGITAL\s*SCREEN(ER|ING)|COMBOHD/i,
    value: "MAMMO",
  },
  { pattern: /\bANGIOGR/i, value: "ANGIO" },
  { pattern: /\bVAS\b|venous\s*doppler|arterial\s*doppler|\bdoppler\b/i, value: "VAS" },
  { pattern: /\bUS\b|ULTRASOUND|ULTRASND/i, value: "US" },
  { pattern: /\bXR\b|X[- ]?RAY|\bCXR\b/i, value: "XR" },
  { pattern: /FLUORO/i, value: "FLUORO" },
  // XR fallbacks: "N VIEW(S)", AP/LAT/OBL/FRONTAL, FLX/EXT, or a bare
  // musculoskeletal/plain-film body-part description with no other modality.
  { pattern: /\b\d+\s*VIEW/i, value: "XR" },
  { pattern: /\bVIEWS?\b/i, value: "XR" },
  { pattern: /\bAP[_/ ]?LAT\b|\bAP\s*AND\s*LAT/i, value: "XR" },
  { pattern: /\bFRONTAL\b/i, value: "XR" },
  { pattern: /\bOBLIQUE\b/i, value: "XR" },
  { pattern: /\bFLX[_/ ]?EXT\b|FLEXION.*EXTENSION/i, value: "XR" },
  { pattern: /\bCHEST\b|\bCLAVICLE\b|\bANKLE\b|\bKNEE\b|\bWRIST\b|\bHAND\b|\bFOOT\b|\bLUMBAR\b|CERVICL|CERVICAL\s*SPINE|\bSHOULDER\b|\bELBOW\b/i, value: "XR" },
];

const REGION_PATTERNS: Rule[] = [
  { pattern: /\bBRAIN\b|IAC\b|STROKE/i, value: "BRAIN" },
  { pattern: /\bHEAD\b|MAXFACIAL|MAX\s*FACIAL|SINUS|COOKIE\s*SWALLOW|SWALLOW/i, value: "HEAD" },
  { pattern: /\bFACE\b|FACIAL|ORBIT|MAXILLOFACIAL/i, value: "FACE" },
  { pattern: /\bNECK\b|SOFT\s*TISSUE\s*NECK/i, value: "NECK" },
  { pattern: /\bCAROTID/i, value: "CAROTID" },
  { pattern: /\bTHYROID/i, value: "THYROID" },
  { pattern: /CERVIC[AL]*\s*SPINE|CSPINE|\bC[- ]?SPINE\b|CERVICL/i, value: "CSPINE" },
  { pattern: /THORACIC\s*SPINE|TSPINE|\bT[- ]?SPINE\b/i, value: "TSPINE" },
  { pattern: /LUMBAR\s*SPINE|LSPINE|\bL[- ]?SPINE\b|\bLUM\b|\bLUMBAR\b/i, value: "LSPINE" },
  { pattern: /\bSPINE\b/i, value: "SPINE" },
  { pattern: /\bCLAVICLE/i, value: "CLAVICLE" },
  { pattern: /\bSHOULDER/i, value: "SHOULDER" },
  { pattern: /\bELBOW/i, value: "ELBOW" },
  { pattern: /\bFOREARM|\bRADIUS\b|\bULNA\b/i, value: "FOREARM" },
  { pattern: /\bWRIST/i, value: "WRIST" },
  { pattern: /\bHAND\b/i, value: "HAND" },
  { pattern: /\bFINGER|THUMB/i, value: "FINGER" },
  { pattern: /\bHIP\b/i, value: "HIP" },
  { pattern: /\bFEMUR\b|THIGH/i, value: "FEMUR" },
  { pattern: /\bKNEE/i, value: "KNEE" },
  { pattern: /\bTIBIA\b|\bFIBULA\b|\bLEG\b/i, value: "TIBIA" },
  { pattern: /\bANKLE/i, value: "ANKLE" },
  { pattern: /\bFOOT|HEEL|CALCANEUS/i, value: "FOOT" },
  { pattern: /\bTOE\b/i, value: "TOE" },
  { pattern: /\bCHEST\b|\bLUNG|\bCXR\b|THORAC(?!IC\s*SPINE)|pul\s*perfusion|pulmonary/i, value: "CHEST" },
  { pattern: /\bHEART\b|CARDIAC|CORONARY|\bmyo\b|\bTTE\b|TRANSTHORAC/i, value: "HEART" },
  { pattern: /\bABD\b|ABDOMEN|ABDOM/i, value: "ABDOMEN" },
  { pattern: /\bPELV|\bPEL\b/i, value: "PELVIS" },
  { pattern: /\bRECTUM|RECTAL/i, value: "RECTUM" },
  { pattern: /\bUTER|TRANSVAGINAL|ENDOMETR/i, value: "UTERUS" },
  { pattern: /\bBREAST|\bmam\b|MAMMO|SPECIMEN/i, value: "BREAST" },
  { pattern: /\bKIDNEY|RENAL/i, value: "KIDNEY" },
  { pattern: /\bBLADDER|CYSTO/i, value: "BLADDER" },
  { pattern: /\bPROSTATE/i, value: "PROSTATE" },
  { pattern: /\bLIVER|HEPATIC/i, value: "LIVER" },
  { pattern: /\bAORTA|AAA|AORTIC/i, value: "AORTA" },
  { pattern: /\bBONE\s*SCAN|TOTAL\s*BODY|SKULL\s*TO\s*THIGH/i, value: "BONE" },
  { pattern: /\bLEG\b|LOWER\s*EXTR/i, value: "LEG" },
  { pattern: /\bARM\b|UPPER\s*EXTR/i, value: "ARM" },
];

export function findModality(text: string): Modality {
  for (const { pattern, value } of MODALITY_RULES) {
    if (pattern.test(text)) return value as Modality;
  }
  return "UNKNOWN";
}

export function findRegions(text: string): Region[] {
  const found = new Set<Region>();
  for (const { pattern, value } of REGION_PATTERNS) {
    if (pattern.test(text)) found.add(value as Region);
  }
  // Drop generic SPINE when a specific spine level (CSPINE / TSPINE / LSPINE) is
  // already present, otherwise generic SPINE causes cross-level false positives.
  if (found.has("CSPINE") || found.has("TSPINE") || found.has("LSPINE")) {
    found.delete("SPINE");
  }
  return [...found];
}

// Region equivalences: clinically interchangeable labels that describe the same
// anatomy. Kept intentionally small — broader equivalences were tested against
// the public labels and created more false positives than they resolved.
const REGION_EQUIV: Array<[Region, Region]> = [
  ["BRAIN", "HEAD"],
  ["LIVER", "ABDOMEN"],
  ["KIDNEY", "ABDOMEN"],
  ["UTERUS", "PELVIS"],
  ["BLADDER", "PELVIS"],
  ["PROSTATE", "PELVIS"],
  ["RECTUM", "PELVIS"],
  ["PULMONARY", "CHEST"],
];

export function regionsEquivalent(a: Region, b: Region): boolean {
  if (a === b) return true;
  for (const [x, y] of REGION_EQUIV) {
    if ((a === x && b === y) || (a === y && b === x)) return true;
  }
  return false;
}

// Bone Scan (NM whole-body) effectively covers every bony region.
const BONE_SCAN_COMPATIBLE: Region[] = [
  "BONE", "CHEST", "ABDOMEN", "PELVIS", "SPINE", "CSPINE", "TSPINE", "LSPINE",
  "HIP", "FEMUR", "KNEE", "TIBIA", "ANKLE", "FOOT", "SHOULDER", "CLAVICLE",
  "ELBOW", "FOREARM", "WRIST", "HAND", "FINGER", "LEG", "ARM", "HEAD", "FACE", "NECK",
];

export function isBoneScanCompatible(r: Region): boolean {
  return BONE_SCAN_COMPATIBLE.includes(r);
}

// Body-region adjacency graph. Same-modality scans of adjacent regions are
// often still clinically relevant (elbow pain → shoulder MRI, for example).
export const ADJACENCY: Record<Region, Region[]> = {
  BRAIN: ["HEAD", "FACE"],
  HEAD: ["BRAIN", "FACE", "NECK"],
  FACE: ["HEAD", "BRAIN", "NECK"],
  NECK: ["HEAD", "CSPINE", "THYROID", "CAROTID", "CHEST"],
  CAROTID: ["NECK", "BRAIN", "HEAD"],
  THYROID: ["NECK"],
  CSPINE: ["NECK", "TSPINE", "HEAD", "SPINE"],
  TSPINE: ["CSPINE", "LSPINE", "CHEST", "SPINE"],
  LSPINE: ["TSPINE", "PELVIS", "HIP", "SPINE"],
  SPINE: ["CSPINE", "TSPINE", "LSPINE"],
  CLAVICLE: ["SHOULDER", "CHEST"],
  SHOULDER: ["CLAVICLE", "ELBOW", "ARM", "CSPINE"],
  ELBOW: ["SHOULDER", "FOREARM", "WRIST", "ARM"],
  FOREARM: ["ELBOW", "WRIST", "ARM"],
  WRIST: ["FOREARM", "ELBOW", "HAND"],
  HAND: ["WRIST", "FINGER"],
  FINGER: ["HAND"],
  HIP: ["PELVIS", "LSPINE", "FEMUR", "LEG"],
  FEMUR: ["HIP", "KNEE", "LEG"],
  KNEE: ["FEMUR", "TIBIA", "LEG"],
  TIBIA: ["KNEE", "ANKLE", "LEG"],
  ANKLE: ["TIBIA", "FOOT", "LEG"],
  FOOT: ["ANKLE", "TOE"],
  TOE: ["FOOT"],
  CHEST: ["HEART", "TSPINE", "AORTA", "CLAVICLE", "PULMONARY"],
  HEART: ["CHEST", "AORTA"],
  ABDOMEN: ["PELVIS", "KIDNEY", "LIVER", "AORTA", "LSPINE"],
  PELVIS: ["ABDOMEN", "HIP", "LSPINE", "BLADDER", "UTERUS", "PROSTATE", "RECTUM"],
  BREAST: [],
  KIDNEY: ["ABDOMEN", "BLADDER"],
  BLADDER: ["PELVIS", "KIDNEY", "PROSTATE", "UTERUS"],
  PROSTATE: ["PELVIS", "BLADDER"],
  UTERUS: ["PELVIS", "BLADDER"],
  RECTUM: ["PELVIS"],
  LIVER: ["ABDOMEN"],
  AORTA: ["CHEST", "ABDOMEN", "HEART"],
  BONE: [],
  PULMONARY: ["CHEST", "HEART"],
  LEG: ["HIP", "KNEE", "ANKLE", "FEMUR", "TIBIA", "FOOT"],
  ARM: ["SHOULDER", "ELBOW", "FOREARM", "WRIST", "HAND"],
  OTHER: [],
};

export function regionsAdjacent(a: Region, b: Region): boolean {
  if (a === b) return true;
  return (ADJACENCY[a] ?? []).includes(b) || (ADJACENCY[b] ?? []).includes(a);
}

// Cross-modality pairs that are clinically substitutable for the same region.
// Primary rule: same modality + same region = relevant. Exception set below.
export const CROSS_MODALITY_PAIRS: Array<[Modality, Modality]> = [
  ["CT", "MRI"],
  ["CT", "XR"],
  ["MRI", "XR"],
  ["CT", "NM"],
  ["NM", "PET"],
  ["CT", "PET"],
  ["MRI", "PET"],
  ["US", "MRI"],
  ["US", "CT"],
  ["MAMMO", "US"],
  ["MAMMO", "MRI"],
  ["MAMMO", "NM"],
  ["ECHO", "NM"],
  ["ECHO", "CT"],
  ["ECHO", "MRI"],
  ["XR", "NM"],
  ["XR", "PET"],
  ["US", "NM"],
  ["US", "XR"],
];

export function modalitiesCompatible(a: Modality, b: Modality): boolean {
  if (a === b) return true;
  for (const [x, y] of CROSS_MODALITY_PAIRS) {
    if ((a === x && b === y) || (a === y && b === x)) return true;
  }
  return false;
}
