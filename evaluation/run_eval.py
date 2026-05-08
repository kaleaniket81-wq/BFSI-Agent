"""
evaluation/run_eval.py — Run the BFSI eval dataset against the pipeline.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --output results/eval_$(date +%Y%m%d).json
"""

import json
import argparse
from pathlib import Path
from evaluation.src.evaluator import RAGEvaluator
from rag.src.pipeline.rag_pipeline_v2 import RAGPipelineV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/datasets/bfsi_eval_dataset.json")
    parser.add_argument("--output",  default="evaluation/results/latest.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    pipeline  = RAGPipelineV2()
    evaluator = RAGEvaluator(pipeline)

    summary = evaluator.evaluate(dataset)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
