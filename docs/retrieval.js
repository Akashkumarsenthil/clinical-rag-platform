const STOPWORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "will", "would",
  "could", "should", "may", "might", "can", "not", "no", "that", "this",
  "these", "those", "it", "its", "what", "how", "was", "given",
]);

const CONCEPTS = [
  ["hypertension", "pressure", "bp", "lisinopril", "potassium", "kidney"],
  ["copd", "dyspnea", "breathlessness", "wheezing", "albuterol", "tiotropium", "prednisone", "bronchodilator", "oxygen"],
  ["diabetes", "a1c", "glucose", "metformin", "hypoglycemia", "glycemic"],
  ["nodule", "pulmonary", "chest", "ct", "imaging", "radiology", "lobe"],
  ["apixaban", "anticoagulation", "bleeding", "blood", "stools", "atrial", "fibrillation", "nsaids"],
  ["monitor", "record", "repeat", "follow", "return", "review", "months", "weeks"],
  ["medication", "dose", "daily", "twice", "increase", "continue", "complete", "treatment", "treated", "prescribed"],
  ["urgent", "warning", "safety", "severe", "worsening", "evaluation"],
];

export function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token && !STOPWORDS.has(token));
}

function termFrequency(tokens) {
  const counts = new Map();
  for (const token of tokens) counts.set(token, (counts.get(token) || 0) + 1);
  return counts;
}

export function bm25Scores(query, corpus, { k1 = 1.5, b = 0.75 } = {}) {
  const tokenized = corpus.map((chunk) => tokenize(chunk.text));
  const queryTokens = [...new Set(tokenize(query))];
  const averageLength = tokenized.reduce((sum, tokens) => sum + tokens.length, 0) / Math.max(tokenized.length, 1);
  const documentFrequency = new Map();

  for (const token of queryTokens) {
    documentFrequency.set(token, tokenized.filter((doc) => doc.includes(token)).length);
  }

  return tokenized.map((tokens) => {
    const frequencies = termFrequency(tokens);
    return queryTokens.reduce((score, token) => {
      const frequency = frequencies.get(token) || 0;
      if (!frequency) return score;
      const containing = documentFrequency.get(token) || 0;
      const idf = Math.log(1 + (corpus.length - containing + 0.5) / (containing + 0.5));
      const lengthPenalty = k1 * (1 - b + b * (tokens.length / Math.max(averageLength, 1)));
      return score + idf * ((frequency * (k1 + 1)) / (frequency + lengthPenalty));
    }, 0);
  });
}

function cosine(left, right) {
  const dot = left.reduce((sum, value, index) => sum + value * right[index], 0);
  const leftNorm = Math.sqrt(left.reduce((sum, value) => sum + value * value, 0));
  const rightNorm = Math.sqrt(right.reduce((sum, value) => sum + value * value, 0));
  return leftNorm && rightNorm ? dot / (leftNorm * rightNorm) : 0;
}

function conceptVector(text) {
  const tokens = tokenize(text);
  return CONCEPTS.map((terms) => tokens.reduce(
    (sum, token) => sum + (terms.includes(token) ? 1 : 0),
    0,
  ));
}

function tokenVector(text, vocabulary) {
  const frequencies = termFrequency(tokenize(text));
  return vocabulary.map((token) => frequencies.get(token) || 0);
}

export function semanticProxyScores(query, corpus) {
  const vocabulary = [...new Set([tokenize(query), ...corpus.map((chunk) => tokenize(chunk.text))].flat())];
  const queryConcepts = conceptVector(query);
  const queryTokens = tokenVector(query, vocabulary);

  return corpus.map((chunk) => {
    const conceptSimilarity = cosine(queryConcepts, conceptVector(chunk.text));
    const tokenSimilarity = cosine(queryTokens, tokenVector(chunk.text, vocabulary));
    return (0.75 * conceptSimilarity) + (0.25 * tokenSimilarity);
  });
}

function rankedIndices(scores) {
  return scores
    .map((score, index) => ({ score, index }))
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map((entry) => entry.index);
}

export function reciprocalRankFusion(rankings, k = 60) {
  const scores = new Map();
  rankings.forEach((ranking) => {
    ranking.forEach((index, rank) => {
      scores.set(index, (scores.get(index) || 0) + (1 / (k + rank + 1)));
    });
  });
  return scores;
}

function normalize(scores) {
  const maximum = Math.max(...scores, 0);
  return maximum ? scores.map((score) => score / maximum) : scores.map(() => 0);
}

function queryCoverage(query, text) {
  const queryTokens = [...new Set(tokenize(query))];
  const documentTokens = new Set(tokenize(text));
  return queryTokens.length
    ? queryTokens.filter((token) => documentTokens.has(token)).length / queryTokens.length
    : 0;
}

export function runRetrieval(query, corpus) {
  const lexical = bm25Scores(query, corpus);
  const semantic = semanticProxyScores(query, corpus);
  const lexicalRanking = rankedIndices(lexical);
  const semanticRanking = rankedIndices(semantic);
  const fusedMap = reciprocalRankFusion([lexicalRanking, semanticRanking]);
  const fused = corpus.map((_, index) => fusedMap.get(index) || 0);
  const normalizedLexical = normalize(lexical);
  const normalizedFused = normalize(fused);
  const reranked = corpus.map((chunk, index) => (
    (0.35 * normalizedLexical[index])
    + (0.40 * semantic[index])
    + (0.15 * queryCoverage(query, chunk.text))
    + (0.10 * normalizedFused[index])
  ));

  const stages = { lexical, semantic, fused, reranked };
  const order = Object.fromEntries(
    Object.entries(stages).map(([stage, scores]) => [stage, rankedIndices(scores)]),
  );
  const ranks = Object.fromEntries(
    Object.entries(order).map(([stage, indices]) => [
      stage,
      Object.fromEntries(indices.map((index, rank) => [index, rank + 1])),
    ]),
  );

  return corpus
    .map((chunk, index) => ({
      ...chunk,
      lexical: lexical[index],
      semantic: semantic[index],
      fused: fused[index],
      reranked: reranked[index],
      ranks: Object.fromEntries(Object.keys(stages).map((stage) => [stage, ranks[stage][index]])),
    }))
    .sort((left, right) => right.reranked - left.reranked || left.id.localeCompare(right.id));
}

export function evaluateRetrieval(queries, corpus, topK = 3) {
  const stageNames = ["lexical", "semantic", "fused", "reranked"];
  const totals = Object.fromEntries(stageNames.map((stage) => [stage, { recall: 0, reciprocalRank: 0 }]));

  for (const evaluation of queries) {
    const results = runRetrieval(evaluation.question, corpus);
    for (const stage of stageNames) {
      const ranked = [...results].sort((left, right) => right[stage] - left[stage] || left.id.localeCompare(right.id));
      const topIds = ranked.slice(0, topK).map((item) => item.id);
      const relevantInTopK = evaluation.relevant.filter((id) => topIds.includes(id)).length;
      const firstRelevantIndex = ranked.findIndex((item) => evaluation.relevant.includes(item.id));
      totals[stage].recall += relevantInTopK / evaluation.relevant.length;
      totals[stage].reciprocalRank += firstRelevantIndex >= 0 ? 1 / (firstRelevantIndex + 1) : 0;
    }
  }

  return Object.fromEntries(stageNames.map((stage) => [stage, {
    recallAtK: totals[stage].recall / queries.length,
    mrr: totals[stage].reciprocalRank / queries.length,
  }]));
}

