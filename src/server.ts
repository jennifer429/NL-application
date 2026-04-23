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

  const response = predict(request, undefined, pairCache);

  if (response.predictions.length !== totalPriors) {
    log("predict.assertion_failed", {
      request_id,
      expected: totalPriors,
      got: response.predictions.length,
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
