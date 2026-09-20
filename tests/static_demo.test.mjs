import test from "node:test";
import assert from "node:assert/strict";

import { chunks, evaluationQueries } from "../docs/data.js";
import {
  bm25Scores,
  evaluateRetrieval,
  reciprocalRankFusion,
  runRetrieval,
  tokenize,
} from "../docs/retrieval.js";

test("tokenizer normalizes punctuation and removes common stopwords", () => {
  assert.deepEqual(tokenize("What is the A1c result?"), ["a1c", "result"]);
});

test("BM25 assigns the strongest exact-match score to the nodule follow-up", () => {
  const scores = bm25Scores("follow-up pulmonary nodule", chunks);
  const topIndex = scores.indexOf(Math.max(...scores));
  assert.equal(chunks[topIndex].id, "ct-02");
});

test("RRF rewards candidates appearing high in both lists", () => {
  const scores = reciprocalRankFusion([[0, 1, 2], [1, 0, 2]]);
  assert.equal(scores.get(0), scores.get(1));
  assert.ok(scores.get(0) > scores.get(2));
});

test("every preset query returns a relevant chunk first after reranking", () => {
  for (const query of evaluationQueries) {
    const [top] = runRetrieval(query.question, chunks);
    assert.ok(query.relevant.includes(top.id), `${query.id}: unexpected top result ${top.id}`);
  }
});

test("evaluation metrics stay within valid bounds", () => {
  const metrics = evaluateRetrieval(evaluationQueries, chunks, 3);
  for (const stage of Object.values(metrics)) {
    assert.ok(stage.recallAtK >= 0 && stage.recallAtK <= 1);
    assert.ok(stage.mrr >= 0 && stage.mrr <= 1);
  }
});

