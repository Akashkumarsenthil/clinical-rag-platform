"""Benchmark runner comparing all chunking strategies via RAGAS."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.evaluation.eval_dataset import EvalDatasetGenerator, QAPair
from src.evaluation.ragas_eval import EvalResults, RAGASEvaluator
from src.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger(__name__)

STRATEGIES = ["recursive", "semantic", "sliding_window"]
SAMPLE_DOCS: list[str] = []  # Populated by scripts/ingest_sample_docs.py at runtime


@dataclass
class StrategyResult:
    """RAGAS results for a single chunking strategy."""

    strategy: str
    chunks_created: int
    eval_results: EvalResults


@dataclass
class BenchmarkReport:
    """Comparison of all strategies."""

    results: list[StrategyResult]
    markdown_table: str


class BenchmarkRunner:
    """Orchestrates end-to-end benchmarking across all chunking strategies.

    For each strategy the runner:
    1. Ingests sample documents with that strategy.
    2. Generates an evaluation dataset from the resulting chunks.
    3. Runs RAGAS metrics.

    Parameters
    ----------
    sample_doc_paths:
        Paths to PDF files to ingest.  Defaults to an empty list (the caller
        must populate this before calling :meth:`run_all_strategies`).
    num_eval_pairs:
        Number of QA pairs to generate per strategy evaluation.
    """

    def __init__(
        self,
        sample_doc_paths: list[str] | None = None,
        num_eval_pairs: int = 25,
    ) -> None:
        self._sample_docs = sample_doc_paths or SAMPLE_DOCS
        self._num_eval_pairs = num_eval_pairs
        self._evaluator = RAGASEvaluator()

    def run_all_strategies(self) -> BenchmarkReport:
        """Run the full benchmark for each chunking strategy.

        Returns:
            BenchmarkReport with per-strategy results and a formatted markdown table.
        """
        logger.info("benchmark_start", strategies=STRATEGIES)
        results: list[StrategyResult] = []

        for strategy in STRATEGIES:
            logger.info("benchmark_strategy_start", strategy=strategy)

            total_chunks = 0
            if self._sample_docs:
                pipeline = IngestionPipeline(chunk_strategy=strategy)
                for doc_path in self._sample_docs:
                    try:
                        result = pipeline.run(doc_path)
                        total_chunks += result.chunks_created
                    except Exception as exc:
                        logger.warning(
                            "benchmark_ingest_failed",
                            strategy=strategy,
                            doc=doc_path,
                            error=str(exc),
                        )

            generator = EvalDatasetGenerator()
            try:
                dataset: list[QAPair] = generator.generate(num_pairs=self._num_eval_pairs)
                eval_results = self._evaluator.evaluate(dataset)
            except Exception as exc:
                logger.error("benchmark_eval_failed", strategy=strategy, error=str(exc))
                from src.evaluation.ragas_eval import EvalResults

                eval_results = EvalResults(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    mean_score=0.0,
                    num_samples=0,
                )

            results.append(
                StrategyResult(
                    strategy=strategy,
                    chunks_created=total_chunks,
                    eval_results=eval_results,
                )
            )
            logger.info(
                "benchmark_strategy_done",
                strategy=strategy,
                mean_score=round(eval_results.mean_score, 4),
            )

        table = self._render_markdown_table(results)
        logger.info("benchmark_complete")
        return BenchmarkReport(results=results, markdown_table=table)

    @staticmethod
    def _render_markdown_table(results: list[StrategyResult]) -> str:
        """Render a Markdown comparison table from strategy results."""
        header = (
            "| Strategy | Chunks | Faithfulness | Answer Relevancy | "
            "Context Precision | Context Recall | Mean |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        rows: list[str] = []
        for r in results:
            e = r.eval_results
            rows.append(
                f"| {r.strategy} | {r.chunks_created} "
                f"| {e.faithfulness:.3f} "
                f"| {e.answer_relevancy:.3f} "
                f"| {e.context_precision:.3f} "
                f"| {e.context_recall:.3f} "
                f"| {e.mean_score:.3f} |"
            )
        return header + "\n".join(rows)
