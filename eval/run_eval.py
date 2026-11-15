"""Run the evaluation dataset through the pipeline and print a report.
Usage:
    python -m eval.run_eval                 # one pass
    python -m eval.run_eval --repeat 3      # also measure consistency
    python -m eval.run_eval --no-filter     # test the MODEL's own injection
                                            # resistance
    python -m eval.run_eval --gates         # exit code 1 if any quality gate
                                            # fails (for CI)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from app.rag import answer_question
from eval import metrics
from eval.judge import judge