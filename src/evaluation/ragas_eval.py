"""RAGAS evaluation runner for the Clinical RAG Platform."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.evaluation.eval_dataset import QAPair

logger = structlog.get_logger(__name__)


class EvaluationError(Exception):
    """Raised when RAGAS evaluation fails."""


@dataclass
class EvalResults:
    """Per-metric RAGAS scores and aggregate statistics."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    mean_score: float
    num_samples: int
    raw: dict[str, float] = field(default_factory=dict)


class RAGASEvaluator:
    """Runs RAGAS evaluation metrics over a list of QAPair objects.

    RAGAS metrics computed:
    - **faithfulness**: fraction of answer claims grounded in the context.
    - **answer_relevancy**: how well the answer addresses the question.
    - **context_precision**: proportion of retrieved contexts that are relevant.
    - **context_recall**: fraction of ground-truth answer covered by context.
    """

    def evaluate(self, dataset: list[QAPair]) -> EvalResults:
        """Run RAGAS evaluation over the provided QA dataset.

        Args:
            dataset: List of QAPair objects with question, answer, and contexts.

        Returns:
            EvalResults with per-metric scores and overall mean.

        Raises:
            EvaluationError: If the RAGAS evaluation call fails.
        """
        if not dataset:
            raise EvaluationError("Cannot evaluate an empty dataset.")

        logger.info("ragas_eval_start", num_samples=len(dataset))

        ragas_data = {
            "question": [p.question for p in dataset],
            "answer": [p.answer for p in dataset],
            "contexts": [p.contexts for p in dataset],
            "ground_truth": [p.answer for p in dataset],  # answer serves as ground truth
        }

        hf_dataset = Dataset.from_dict(ragas_data)

        try:
            result = evaluate(
                hf_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
        except Exception as exc:
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

        scores: dict[str, float] = dict(result)
        faith = float(scores.get("faithfulness", 0.0))
        ans_rel = float(scores.get("answer_relevancy", 0.0))
        ctx_prec = float(scores.get("context_precision", 0.0))
        ctx_rec = float(scores.get("context_recall", 0.0))
        mean = (faith + ans_rel + ctx_prec + ctx_rec) / 4.0

        logger.info(
            "ragas_eval_done",
            faithfulness=round(faith, 4),
            answer_relevancy=round(ans_rel, 4),
            context_precision=round(ctx_prec, 4),
            context_recall=round(ctx_rec, 4),
            mean=round(mean, 4),
        )

        return EvalResults(
            faithfulness=faith,
            answer_relevancy=ans_rel,
            context_precision=ctx_prec,
            context_recall=ctx_rec,
            mean_score=mean,
            num_samples=len(dataset),
            raw=scores,
        )
