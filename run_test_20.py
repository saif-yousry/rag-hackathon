"""Run the 20-question mixed QA test.

Usage (from the rag_project root, with GROQ_API_KEY set):
    python run_test_20.py data/questions_test_20.json

Expected behavior encoded in this script:
- Questions 1-15 (in-scope)  -> should be answered (gate_decision == "generated"),
                                  overall_score >= 0.77
- Questions 16-20 (out-of-scope) -> should be refused (gate_decision == "refused")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.pipeline import build_retrieval_stack
from src.qa.engine import QAEngine, format_answer
from src.qa.logger import TraceLogger


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    questions_path = Path(sys.argv[1]).resolve()
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    print(f"[test] loaded {len(questions)} questions from {questions_path.name}")

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Set it first, e.g.:")
        print("  $env:GROQ_API_KEY = 'gsk_xxxxx'   (PowerShell)")
        sys.exit(1)

    cfg = AppConfig.from_yaml(ROOT / "config" / "config.yaml")
    print("[test] building retrieval stack (uses your index + hybrid retrieval)...")
    retriever = build_retrieval_stack(cfg)

    from groq import Groq

    engine = QAEngine(retriever, Groq(api_key=os.environ["GROQ_API_KEY"]))

    log_dir = ROOT / "qa_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = sorted(log_dir.glob("test_*.json"))
    idx = len(log_files)

    correct = 0
    total = len(questions)

    for i, q in enumerate(questions, start=1):
        question = q["question"]
        expect_refuse = i >= 16  # last 5 are out-of-scope
        logger = TraceLogger(path=str(log_dir / f"test_{idx:03d}_q{i:02d}.json"))
        engine.logger = logger
        result = engine.ask(question, show_details=False)
        logger.set("expected", "refused" if expect_refuse else "answered")
        logger.save()

        if expect_refuse:
            ok = result.is_refused
            verdict = "REFUSE-OK" if ok else "SHOULD-HAVE-REFUSED"
        else:
            ok = (not result.is_refused) and (
                result.overall_score is not None
                and result.overall_score >= 0.77
            )
            verdict = "ANSWER-OK" if ok else "WEAK-OR-REFUSED"

        if ok:
            correct += 1
        score_str = (
            f"{result.overall_score:.3f} ({result.overall_level.value})"
            if result.overall_score is not None
            else "n/a (refused)"
        )
        print(
            f"[{i:02d}] {verdict}  |  gate={result.gate_decision}  "
            f"|  score={score_str}  |  {question[:64]}..."
        )

    print("=" * 70)
    print(f"RESULT: {correct}/{total} passed ({100*correct/total:.0f}%)")
    print(f"[test] traces saved under {log_dir}")


if __name__ == "__main__":
    main()
