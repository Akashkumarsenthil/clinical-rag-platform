import { chunks, documents, evaluationQueries } from "./data.js";
import { evaluateRetrieval, runRetrieval } from "./retrieval.js";

const queryInput = document.querySelector("#query-input");
const scopeSelect = document.querySelector("#scope-select");
const runButton = document.querySelector("#run-query");
const presetList = document.querySelector("#preset-list");
const resultList = document.querySelector("#result-list");
const resultCount = document.querySelector("#result-count");
const rankingTitle = document.querySelector("#ranking-title");
const contextList = document.querySelector("#context-list");
const answerPreview = document.querySelector("#answer-preview");
const metricChart = document.querySelector("#metric-chart");
const stageButtons = [...document.querySelectorAll("[data-stage]")];

const stageLabels = {
  lexical: "BM25 lexical results",
  semantic: "Semantic-proxy results",
  fused: "Reciprocal-rank fusion",
  reranked: "Reranked results",
};

let activeStage = "reranked";
let currentResults = [];
let selectedChunkId = null;

function getDocument(docId) {
  return documents.find((doc) => doc.id === docId);
}

function formatScore(score) {
  return Number.isFinite(score) ? score.toFixed(3) : "0.000";
}

function buildScopeOptions() {
  for (const item of documents) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.title;
    scopeSelect.append(option);
  }
}

function buildPresets() {
  evaluationQueries.forEach((query) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = query.label;
    button.dataset.queryId = query.id;
    button.addEventListener("click", () => {
      queryInput.value = query.question;
      scopeSelect.value = "all";
      document.querySelectorAll("[data-query-id]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      executeQuery();
    });
    presetList.append(button);
  });
}

function scoreBadges(result) {
  return [
    ["lexical", "BM25"],
    ["semantic", "SEM"],
    ["fused", "RRF"],
    ["reranked", "RERANK"],
  ].map(([stage, label]) => (
    `<span class="${stage === activeStage ? "active" : ""}"><b>${label}</b>${formatScore(result[stage])}</span>`
  )).join("");
}

function renderResults() {
  const ranked = [...currentResults].sort((left, right) => (
    right[activeStage] - left[activeStage] || left.id.localeCompare(right.id)
  ));
  resultList.replaceChildren();
  resultCount.textContent = `${ranked.length} chunk${ranked.length === 1 ? "" : "s"}`;
  rankingTitle.textContent = stageLabels[activeStage];

  ranked.slice(0, 6).forEach((result, index) => {
    const source = getDocument(result.docId);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `result-card${result.id === selectedChunkId ? " selected" : ""}`;
    card.setAttribute("aria-label", `Inspect result ${index + 1}: ${source.title}`);
    card.innerHTML = `
      <div class="result-card-top">
        <span class="result-rank">${String(index + 1).padStart(2, "0")}</span>
        <strong>${source.title} · ${result.heading}</strong>
        <span class="result-score">${formatScore(result[activeStage])}</span>
      </div>
      <p>${result.text}</p>
      <div class="score-row">${scoreBadges(result)}</div>
    `;
    card.addEventListener("click", () => {
      selectedChunkId = result.id;
      renderResults();
      renderContext(ranked);
    });
    resultList.append(card);
  });

  renderContext(ranked);
}

function renderContext(ranked) {
  const top = ranked.slice(0, 3);
  contextList.replaceChildren();

  top.forEach((result, index) => {
    const source = getDocument(result.docId);
    const card = document.createElement("article");
    card.className = "context-card";
    card.innerHTML = `
      <header><strong>[${index + 1}] ${source.title}</strong><span>p.${result.page} · ${result.id}</span></header>
      <p>${result.text}</p>
    `;
    contextList.append(card);
  });

  if (!top.length) {
    answerPreview.textContent = "No context was retrieved for this scope.";
    return;
  }

  const preview = top.slice(0, 2).map((result, index) => `[${index + 1}] ${result.text}`).join(" ");
  answerPreview.textContent = `Retrieved evidence: ${preview}`;
}

function executeQuery() {
  const question = queryInput.value.trim();
  if (!question) {
    queryInput.focus();
    return;
  }

  const corpus = scopeSelect.value === "all"
    ? chunks
    : chunks.filter((chunk) => chunk.docId === scopeSelect.value);

  currentResults = runRetrieval(question, corpus);
  selectedChunkId = currentResults[0]?.id ?? null;
  renderResults();
}

function renderMetrics() {
  const metrics = evaluateRetrieval(evaluationQueries, chunks, 3);
  const labels = {
    lexical: "BM25",
    semantic: "Concept proxy",
    fused: "RRF fusion",
    reranked: "Rule reranker",
  };

  metricChart.replaceChildren();
  Object.entries(metrics).forEach(([stage, values]) => {
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `
      <strong>${labels[stage]}</strong>
      <div class="metric-track" title="Recall@3 ${(values.recallAtK * 100).toFixed(0)}%">
        <div class="metric-fill" style="width: ${(values.recallAtK * 100).toFixed(1)}%"></div>
      </div>
      <span class="metric-value">${(values.recallAtK * 100).toFixed(0)}%</span>
    `;
    const mrr = document.createElement("span");
    mrr.className = "metric-value";
    mrr.style.gridColumn = "2 / 4";
    mrr.textContent = `MRR ${values.mrr.toFixed(2)}`;
    row.append(mrr);
    metricChart.append(row);
  });

  const legend = document.createElement("div");
  legend.className = "metric-legend";
  legend.innerHTML = "<span>Recall@3</span><span>Remaining</span>";
  metricChart.append(legend);
}

stageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeStage = button.dataset.stage;
    stageButtons.forEach((item) => item.classList.toggle("active", item === button));
    renderResults();
  });
});

runButton.addEventListener("click", executeQuery);
scopeSelect.addEventListener("change", executeQuery);
queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") executeQuery();
});
queryInput.addEventListener("input", () => {
  document.querySelectorAll("[data-query-id]").forEach((item) => item.classList.remove("active"));
});

buildScopeOptions();
buildPresets();
renderMetrics();
executeQuery();

