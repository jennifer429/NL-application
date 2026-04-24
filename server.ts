import express from "express";
import { randomUUID } from "node:crypto";
import { RequestSchema } from "./types.ts";
import { predict } from "./predict.ts";
import { LRUCache } from "./cache.ts";
import { log } from "./logger.ts";

const app = express();
app.use(express.json({ limit: "50mb" }));

const pairCache = new LRUCache<boolean>(1_000_000);

app.get("/health", (_req, res) => {
  res.json({ ok: true, cache: pairCache.stats() });
});

app.post("/predict", (req, res) => {
  const request_id = randomUUID();
  const started = Date.now();

  const parsed = RequestSchema.safeParse(req.body);
  if (!parsed.success) {
    log("predict.bad_request", { request_id, issues: parsed.error.issues.slice(0, 5) });
    return res.status(400).json({ error: "invalid_request", issues: parsed.error.issues });
  }

  const request = parsed.data;
  const totalPriors = request.cases.reduce((n, c) => n + c.prior_studies.length, 0);

  log("predict.start", {
    request_id,
    challenge_id: request.challenge_id,
    cases: request.cases.length,
    total_priors: totalPriors,
  });

  let response: { predictions: Array<{ case_id: string; study_id: string; predicted_is_relevant: boolean }> };
  try {
    response = predict(request, undefined, pairCache);
  } catch (err) {
    log("predict.error", {
      request_id,
      error: err instanceof Error ? err.message : String(err),
    });
    response = { predictions: [] };
  }

  // Belt-and-suspenders: guarantee one prediction per prior even if predict()
  // threw partway through or returned an unexpected shape. Skips count as
  // incorrect under the challenge scoring rule, so a conservative "false" is
  // better than a missing row (base rate of priors is 76% not-relevant).
  if (response.predictions.length !== totalPriors) {
    const seen = new Set(response.predictions.map((p) => `${p.case_id}|${p.study_id}`));
    let padded = 0;
    for (const c of request.cases) {
      for (const p of c.prior_studies) {
        const k = `${c.case_id}|${p.study_id}`;
        if (!seen.has(k)) {
          response.predictions.push({
            case_id: c.case_id,
            study_id: p.study_id,
            predicted_is_relevant: false,
          });
          padded++;
        }
      }
    }
    log("predict.padded", {
      request_id,
      expected: totalPriors,
      original: response.predictions.length - padded,
      padded,
    });
  }

  const ms = Date.now() - started;
  log("predict.done", {
    request_id,
    cases: request.cases.length,
    total_priors: totalPriors,
    predictions: response.predictions.length,
    ms,
    cache: pairCache.stats(),
  });

  res.json(response);
});

const port = Number(process.env.PORT ?? 8080);
app.listen(port, () => {
  log("server.listen", { port });
});
