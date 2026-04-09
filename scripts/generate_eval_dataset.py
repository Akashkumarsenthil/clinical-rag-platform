#!/usr/bin/env python3
"""Generate an evaluation dataset of 100 QA pairs and save to JSON."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.eval_dataset import EvalDatasetGenerator, QAPair

log = structlog.get_logger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"


def main(num_pairs: int = 100) -> None:
    """Generate QA pairs and save them to disk.

    Args:
        num_pairs: Number of question-answer pairs to generate (default 100).
    """
    if len(sys.argv) > 1:
        try:
            num_pairs = int(sys.argv[1])
        except ValueError:
            pass

    log.info("generating_eval_dataset", num_pairs=num_pairs)
    generator = EvalDatasetGenerator()

    dataset: list[QAPair] = generator.generate(num_pairs=num_pairs)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Serialise to JSON
    records = [
        {
            "question": pair.question,
            "answer": pair.answer,
            "contexts": pair.contexts,
        }
        for pair in dataset
    ]
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    # Print statistics
    total_contexts = sum(len(p.contexts) for p in dataset)
    avg_q_len = sum(len(p.question) for p in dataset) / max(len(dataset), 1)
    avg_a_len = sum(len(p.answer) for p in dataset) / max(len(dataset), 1)

    print("\n" + "=" * 60)
    print("EVALUATION DATASET STATISTICS")
    print("=" * 60)
    print(f"Total QA pairs         : {len(dataset)}")
    print(f"Total context chunks   : {total_contexts}")
    print(f"Avg question length    : {avg_q_len:.0f} chars")
    print(f"Avg answer length      : {avg_a_len:.0f} chars")
    print(f"Output file            : {OUTPUT_PATH}")
    print("=" * 60)

    log.info(
        "eval_dataset_saved",
        path=str(OUTPUT_PATH),
        num_pairs=len(dataset),
    )


if __name__ == "__main__":
    main()
