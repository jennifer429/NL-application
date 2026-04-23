import { describe, it, expect } from "vitest";
import { predict } from "../src/predict.ts";
import { RequestSchema } from "../src/types.ts";

const sample = {
  challenge_id: "relevant-priors-v1",
  schema_version: 1,
  cases: [
    {
      case_id: "c1",
      patient_id: "p1",
      patient_name: "X",
      current_study: {
        study_id: "s0",
        study_description: "MRI BRAIN WITHOUT CONTRAST",
        study_date: "2026-03-08",
      },
      prior_studies: [
        { study_id: "s1", study_description: "MRI BRAIN WITHOUT CONTRAST", study_date: "2020-03-08" },
        { study_id: "s2", study_description: "CT HEAD WITHOUT CNTRST", study_date: "2021-03-08" },
        { study_id: "s3", study_description: "DXA (Hip/Spine Only)", study_date: "2019-03-08" },
      ],
    },
    {
      case_id: "c2",
      current_study: { study_id: "sA", study_description: "XR CHEST 2V PA/LAT", study_date: "2026-01-01" },
      prior_studies: [],
    },
  ],
};

describe("predict", () => {
  it("returns exactly one prediction per prior across all cases", () => {
    const parsed = RequestSchema.parse(sample);
    const out = predict(parsed);
    const expected = sample.cases.reduce((n, c) => n + c.prior_studies.length, 0);
    expect(out.predictions).toHaveLength(expected);
    const keys = new Set(out.predictions.map((p) => `${p.case_id}|${p.study_id}`));
    expect(keys.size).toBe(expected);
  });

  it("predicts true for an exact-match prior", () => {
    const out = predict(RequestSchema.parse(sample));
    const p = out.predictions.find((x) => x.study_id === "s1");
    expect(p?.predicted_is_relevant).toBe(true);
  });

  it("predicts false for DXA prior vs MRI BRAIN current", () => {
    const out = predict(RequestSchema.parse(sample));
    const p = out.predictions.find((x) => x.study_id === "s3");
    expect(p?.predicted_is_relevant).toBe(false);
  });

  it("rejects malformed requests via Zod", () => {
    const bad = RequestSchema.safeParse({ cases: [{ case_id: 1 }] });
    expect(bad.success).toBe(false);
  });
});
