import { describe, it, expect } from "vitest";
import { parse } from "../src/parse.ts";
import { isRelevant, DEFAULT_CONFIG, daysBetween } from "../src/rules.ts";

function rel(a: string, b: string, days = 30) {
  return isRelevant(parse(a), parse(b), days, DEFAULT_CONFIG);
}

describe("relevance rules", () => {
  it("exact match is relevant", () => {
    expect(rel("US PARACENTESIS", "US PARACENTESIS")).toBe(true);
  });

  it("same modality + same region is relevant", () => {
    expect(rel("CT CHEST WITHOUT CONTRAST", "CT CHEST WITH CONTRAST")).toBe(true);
    expect(rel("MRI lumbar spine wo/w con", "LUMBAR SPINE, LATERAL VIEW ONLY")).toBe(true);
  });

  it("DXA vs unrelated modality is not relevant", () => {
    expect(rel("CT ABDOMEN PELVIS W CONTRAST", "DXA (Hip/Spine Only)")).toBe(false);
  });

  it("adjacency is off by default (empirically hurts accuracy) but configurable", () => {
    expect(rel("MRI HAND LEFT", "MRI WRIST LEFT")).toBe(false);
    const on = isRelevant(
      parse("MRI HAND LEFT"),
      parse("MRI WRIST LEFT"),
      30,
      { ...DEFAULT_CONFIG, allowAdjacentRegion: true, ignoreLateralityMismatch: true },
    );
    expect(on).toBe(true);
  });

  it("different region, same modality: MRI BRAIN vs MRI LUMBAR is not relevant", () => {
    expect(rel("MRI BRAIN WITH IAC", "MRI LUMBAR SPINE WO CON")).toBe(false);
  });

  it("laterality conflict blocks same-side match", () => {
    expect(rel("XR KNEE LEFT", "XR KNEE RIGHT")).toBe(false);
    expect(rel("XR KNEE LEFT", "XR KNEE BILATERAL")).toBe(true);
  });

  it("days apart field is accepted", () => {
    expect(daysBetween("2024-01-01", "2024-01-31")).toBeCloseTo(30, 0);
  });
});
