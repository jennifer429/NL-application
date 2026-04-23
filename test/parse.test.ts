import { describe, it, expect } from "vitest";
import { parse } from "../src/parse.ts";

describe("parse", () => {
  it("picks MRI over XR when description leads with MRI", () => {
    const p = parse("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST");
    expect(p.modality).toBe("MRI");
    expect(p.regions).toContain("BRAIN");
    expect(p.withContrast).toBe(false);
  });

  it("parses CT HEAD WITHOUT CNTRST", () => {
    const p = parse("CT HEAD WITHOUT CNTRST");
    expect(p.modality).toBe("CT");
    expect(p.regions).toContain("HEAD");
    expect(p.withContrast).toBe(false);
  });

  it("catches CERVICL typo as CSPINE", () => {
    const p = parse("CERVICL SPINE, LIMITED");
    expect(p.regions).toContain("CSPINE");
  });

  it("parses trailing comma on KNEE", () => {
    const p = parse("KNEE, LEFT - 3 VIEWS");
    expect(p.regions).toContain("KNEE");
    expect(p.laterality).toBe("LT");
  });

  it("treats NM myo perf SPECT as NM of the heart", () => {
    const p = parse("NM myo perf SPECT rest & str");
    expect(p.modality).toBe("NM");
    expect(p.regions).toContain("HEART");
  });

  it("treats MAM variants as MAMMO of the breast", () => {
    for (const d of ["MAM screen BI with tomo", "MAMMOGRAPHY", "MAM diag mammo LT"]) {
      const p = parse(d);
      expect(p.modality).toBe("MAMMO");
      expect(p.regions).toContain("BREAST");
    }
  });

  it("picks up BI laterality", () => {
    expect(parse("MAM screen BI with tomo").laterality).toBe("BI");
    expect(parse("MAM US BI breast screening").laterality).toBe("BI");
  });

  it("detects WITH and WITHOUT contrast", () => {
    expect(parse("CT CHEST WITH CONTRAST").withContrast).toBe(true);
    expect(parse("CT CHEST WITHOUT CONTRAST").withContrast).toBe(false);
    expect(parse("CT CHEST WO CNTRST").withContrast).toBe(false);
    expect(parse("MRI lumbar spine wo/w con").withContrast).toBe(true);
  });
});
