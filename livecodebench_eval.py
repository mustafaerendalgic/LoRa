#!/usr/bin/env python3
"""
LiveCodeBench Evaluation Pipeline

Author: naholav

Evaluates LoRA fine-tuned models on LiveCodeBench benchmark.
Supports filtering by difficulty (easy, medium, hard) and model type.

Usage:
    # Evaluate all models on all difficulties
    python livecodebench_eval.py

    # Evaluate specific model type
    python livecodebench_eval.py --model_type deep_think

    # Evaluate specific difficulty
    python livecodebench_eval.py --difficulty medium

    # Evaluate specific checkpoint step
    python livecodebench_eval.py --steps 500 600

    # Include base model comparison
    python livecodebench_eval.py --include_base

Output:
    results/livecodebench/detailed/{model_name}_{difficulty}.jsonl  <-- DETAILED LOGS
    results/livecodebench/generations/{model_name}_{difficulty}.json
    results/livecodebench/evaluations/{model_name}_{difficulty}_results.json
    results/livecodebench/summary.json
"""

import json
import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add common to path
sys.path.insert(0, str(Path(__file__).parent))
from common.model_loader import load_base_model, load_lora_checkpoint, generate_code
from common.code_postprocess import postprocess_generated_code
from common.code_executor import execute_code_subprocess, evaluate_solution


# =============================================================================
# Configuration - MODIFY THESE PATHS FOR YOUR SETUP
# =============================================================================

@dataclass
class Config:
    # Base model (HuggingFace model name)
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # =========================================================================
    # IMPORTANT: Update this path to your checkpoint directory
    # =========================================================================
    checkpoint_base_dir: str = "./models"  # <-- YOUR_CHECKPOINT_PATH

    # Model types to evaluate (must match your training setup)
    model_types: tuple = ("deep_instruction", "diverse_instruction")

    # Checkpoint steps to evaluate
    checkpoint_steps: tuple = (300, 400, 500, 600, 700, 800)

    # LiveCodeBench settings
    livecodebench_version: str = "release_v5"  # Latest version with 880 problems

    # NVIDIA-style date filtering (YYMM format)
    # 2408 = August 2024, 2502 = February 2025
    date_range_start: str = "2408"  # YYMM
    date_range_end: str = "2502"    # YYMM

    # Generation settings
    max_new_tokens: int = 8192
    temperature: float = 0.0  # Greedy decoding for reproducibility
    top_p: float = 1.0
    num_samples: int = 1  # For pass@1
    num_runs: int = 1     # NVIDIA uses 64, we use 1 for speed

    # System prompts for different model types
    system_prompts: dict = None

    # Output directories
    output_base_dir: str = "./results/livecodebench"

    def __post_init__(self):
        # MUST match training system prompts exactly!
        self.system_prompts = {
            "think": "You are an expert programmer. Use <think> tags for reasoning before writing code.",
            "instruction": "You are an expert Python programmer. Please read the problem carefully before writing any Python code."
        }


CONFIG = Config()


# =============================================================================
# Difficulty Mapping
# =============================================================================

DIFFICULTY_MAP = {
    "easy": ["easy", "Easy", "EASY", "simple", "Simple"],
    "medium": ["medium", "Medium", "MEDIUM", "moderate", "Moderate"],
    "hard": ["hard", "Hard", "HARD", "difficult", "Difficult"],
}


def categorize_difficulty(difficulty_str: str) -> str:
    """
    Categorize a difficulty string into easy/medium/hard.
    """
    if difficulty_str is None:
        return "unknown"

    diff_lower = str(difficulty_str).lower().strip()

    if diff_lower in ["easy", "simple"]:
        return "easy"
    elif diff_lower in ["medium", "moderate"]:
        return "medium"
    elif diff_lower in ["hard", "difficult"]:
        return "hard"

    # Codeforces rating-based
    try:
        rating = int(difficulty_str)
        if rating <= 1200:
            return "easy"
        elif rating <= 1800:
            return "medium"
        else:
            return "hard"
    except (ValueError, TypeError):
        pass

    # AtCoder problem difficulty
    if diff_lower in ["a", "b"]:
        return "easy"
    elif diff_lower in ["c", "d"]:
        return "medium"
    elif diff_lower in ["e", "f", "g", "h"]:
        return "hard"

    return "unknown"


# =============================================================================
# Dataset Loading
# =============================================================================

def load_livecodebench(
    version: str = "release_v5",
    difficulty: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    platform: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load LiveCodeBench dataset from HuggingFace with NVIDIA-style filtering.
    """
    print(f"\nLoading LiveCodeBench ({version})...")

    try:
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag=version,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading code_generation_lite, trying code_generation: {e}")
        dataset = load_dataset(
            "livecodebench/code_generation",
            version_tag=version,
            trust_remote_code=True
        )

    # Get the test split
    if "test" in dataset:
        problems = list(dataset["test"])
    elif "train" in dataset:
        problems = list(dataset["train"])
    else:
        problems = list(dataset[list(dataset.keys())[0]])

    print(f"Loaded {len(problems)} problems total")

    if problems:
        print(f"\nDataset fields: {list(problems[0].keys())}")

    # NVIDIA-style date filtering
    if date_start or date_end:
        original_count = len(problems)
        filtered = []
        for p in problems:
            date = p.get('contest_date', '')
            if date and len(date) >= 7:
                year_month = date[:7].replace('-', '')
                yymm = year_month[2:6]

                in_range = True
                if date_start and yymm < date_start:
                    in_range = False
                if date_end and yymm > date_end:
                    in_range = False

                if in_range:
                    filtered.append(p)

        problems = filtered
        print(f"Date filter ({date_start or 'any'} to {date_end or 'any'}): {len(problems)} problems (from {original_count})")

    # Platform filtering
    if platform and platform != "all":
        original_count = len(problems)
        problems = [p for p in problems if p.get("platform", "").lower() == platform.lower()]
        print(f"Platform filter ({platform}): {len(problems)} problems (from {original_count})")

    # Filter by difficulty
    if difficulty and difficulty != "all":
        original_count = len(problems)
        problems = [
            p for p in problems
            if categorize_difficulty(p.get("difficulty", "")) == difficulty
        ]
        print(f"Difficulty filter ({difficulty}): {len(problems)} problems (from {original_count})")

    # Show platform distribution
    platforms = {}
    for p in problems:
        plat = p.get('platform', 'unknown')
        platforms[plat] = platforms.get(plat, 0) + 1
    print(f"Platform distribution: {platforms}")

    return problems


# =============================================================================
# Model Discovery
# =============================================================================

def discover_checkpoints(
    base_dir: str,
    model_types: Optional[List[str]] = None,
    steps: Optional[List[int]] = None
) -> Dict[str, List[str]]:
    """
    Discover all available checkpoints.
    """
    base_path = Path(base_dir)
    checkpoints = defaultdict(list)

    if not base_path.exists():
        print(f"WARNING: Checkpoint directory not found: {base_dir}")
        return dict(checkpoints)

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_type = model_dir.name

        if model_types and model_type not in model_types:
            continue

        checkpoint_dir = model_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue

        for ckpt in sorted(checkpoint_dir.iterdir()):
            if not ckpt.is_dir() or not ckpt.name.startswith("checkpoint-"):
                continue

            try:
                step = int(ckpt.name.split("-")[2])
            except (IndexError, ValueError):
                continue

            if steps and step not in steps:
                continue

            checkpoints[model_type].append(str(ckpt))

    return dict(checkpoints)


# =============================================================================
# Solution Generation
# =============================================================================

def build_prompt(problem: Dict[str, Any]) -> str:
    """
    Build the user prompt for code generation.
    """
    description = (
        problem.get("question_content") or
        problem.get("problem_description") or
        problem.get("prompt") or
        problem.get("description") or
        str(problem)
    )

    return description


def extract_test_cases(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract test cases from a problem.
    """
    test_cases = []

    io_fields = ["public_test_cases", "private_test_cases", "input_output",
                 "public_input_output", "test_cases", "examples"]

    for field in io_fields:
        if field not in problem or problem[field] is None:
            continue

        io = problem[field]

        try:
            if isinstance(io, str):
                io = json.loads(io)

            if isinstance(io, list):
                for tc in io:
                    if isinstance(tc, dict):
                        inp = tc.get("input", "")
                        out = tc.get("output", tc.get("expected", ""))
                        if inp or out:
                            test_cases.append({
                                "input": str(inp) if inp is not None else "",
                                "output": str(out) if out is not None else ""
                            })

            elif isinstance(io, dict):
                inputs = io.get("inputs", io.get("input", []))
                outputs = io.get("outputs", io.get("output", []))

                if isinstance(inputs, list) and isinstance(outputs, list):
                    for inp, out in zip(inputs, outputs):
                        test_cases.append({
                            "input": str(inp) if inp is not None else "",
                            "output": str(out) if out is not None else ""
                        })

        except (json.JSONDecodeError, TypeError, KeyError):
            continue

        if test_cases:
            break

    return test_cases


def generate_and_evaluate(
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    system_prompt: str,
    config: Config,
    output_file: str,
    detailed_file: str,
    model_name: str,
    difficulty: str
) -> Dict[str, Any]:
    """
    Generate solutions and evaluate them, saving detailed results to JSONL.
    """
    results = []
    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "error": 0,
        "no_tests": 0
    }

    # Check for existing results (resume support)
    existing_ids = set()
    if os.path.exists(detailed_file):
        try:
            with open(detailed_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        existing_ids.add(entry["question_id"])
            print(f"Resuming from {len(existing_ids)} existing solutions")
        except (json.JSONDecodeError, KeyError):
            pass

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    problems_to_process = [p for p in problems if (
        p.get("question_id") or p.get("task_id") or p.get("id")
    ) not in existing_ids]

    print(f"\nGenerating and evaluating {len(problems_to_process)} problems...")
    print(f"(Skipping {len(existing_ids)} already processed)")

    with open(detailed_file, 'a', buffering=1) as detail_f:
        for problem in tqdm(problems_to_process, desc=f"Evaluating ({model_name})"):
            question_id = (
                problem.get("question_id") or
                problem.get("task_id") or
                problem.get("id") or
                str(hash(str(problem)))
            )

            stats["total"] += 1

            problem_content = (
                problem.get("question_content") or
                problem.get("problem_description") or
                problem.get("prompt") or
                problem.get("description") or
                ""
            )

            prompt = build_prompt(problem)

            detailed_entry = {
                "question_id": question_id,
                "difficulty": difficulty,
                "problem_title": problem.get("question_title", problem.get("title", "")),
                "problem_prompt": problem_content[:5000],
                "model_output_raw": "",
                "extracted_code": "",
                "passed": False,
                "test_results": None,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

            try:
                generated_texts = generate_code(
                    model, tokenizer, prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    num_return_sequences=config.num_samples,
                    system_prompt=system_prompt
                )

                raw_output = generated_texts[0] if generated_texts else ""
                extracted_code = postprocess_generated_code(raw_output)

                detailed_entry["model_output_raw"] = raw_output
                detailed_entry["extracted_code"] = extracted_code

                results.append({
                    "question_id": question_id,
                    "code_list": [extracted_code]
                })

                test_cases = extract_test_cases(problem)

                if test_cases:
                    eval_result = evaluate_solution(
                        extracted_code,
                        test_cases,
                        timeout_per_case=10.0,
                        memory_limit_mb=512
                    )

                    detailed_entry["test_results"] = {
                        "total": eval_result["total"],
                        "passed": eval_result["passed"],
                        "failed": eval_result["failed"],
                        "timeout": eval_result["timeout"],
                        "error": eval_result["error"],
                        "pass_rate": eval_result["pass_rate"]
                    }

                    if eval_result["all_passed"]:
                        detailed_entry["passed"] = True
                        stats["passed"] += 1
                    else:
                        stats["failed"] += 1
                else:
                    detailed_entry["test_results"] = {"note": "No test cases available"}
                    stats["no_tests"] += 1

            except Exception as e:
                detailed_entry["error"] = str(e)
                detailed_entry["model_output_raw"] = f"# Generation failed: {e}"
                detailed_entry["extracted_code"] = f"# Error: {e}"
                stats["error"] += 1

                results.append({
                    "question_id": question_id,
                    "code_list": [f"# Generation failed: {e}"]
                })

            json_line = json.dumps(detailed_entry, ensure_ascii=False) + '\n'
            detail_f.write(json_line)
            detail_f.flush()
            os.fsync(detail_f.fileno())

            status = "PASS" if detailed_entry["passed"] else "FAIL"
            print(f"  [{stats['total']}] {question_id}: {status}")

            if stats["total"] % 5 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    evaluated = stats["passed"] + stats["failed"]
    stats["pass_at_1"] = stats["passed"] / evaluated if evaluated > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Results for {model_name} on {difficulty}:")
    print(f"  Total: {stats['total']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Errors: {stats['error']}")
    print(f"  No tests: {stats['no_tests']}")
    print(f"  Pass@1: {stats['pass_at_1']*100:.2f}%")
    print(f"{'='*60}")

    return stats


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_checkpoint(
    checkpoint_path: Optional[str],
    model_type: str,
    problems: List[Dict[str, Any]],
    difficulty: str,
    config: Config,
    is_base_model: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single checkpoint on the given problems.
    """
    if is_base_model:
        model_name = "base_model"
    else:
        model_name = f"{model_type}_{Path(checkpoint_path).name}"

    print("\n" + "="*80)
    print(f"EVALUATING: {model_name} on {difficulty} problems")
    print("="*80)

    gen_dir = Path(config.output_base_dir) / "generations"
    eval_dir = Path(config.output_base_dir) / "evaluations"
    detailed_dir = Path(config.output_base_dir) / "detailed"
    gen_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir.mkdir(parents=True, exist_ok=True)

    gen_file = gen_dir / f"{model_name}_{difficulty}.json"
    eval_file = eval_dir / f"{model_name}_{difficulty}_results.json"
    detailed_file = detailed_dir / f"{model_name}_{difficulty}.jsonl"

    if "think" in model_type:
        system_prompt = config.system_prompts["think"]
    else:
        system_prompt = config.system_prompts["instruction"]

    print(f"\nLoading model...")
    if is_base_model:
        model, tokenizer = load_base_model(
            config.base_model,
            use_flash_attention_2=True
        )
    else:
        model, tokenizer = load_lora_checkpoint(
            checkpoint_path,
            base_model_name=config.base_model,
            use_flash_attention_2=True
        )

    stats = generate_and_evaluate(
        model, tokenizer, problems, system_prompt, config,
        str(gen_file), str(detailed_file), model_name, difficulty
    )

    del model
    del tokenizer
    torch.cuda.empty_cache()

    results = {
        "model_name": model_name,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "difficulty": difficulty,
        "num_problems": len(problems),
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "pass_at_1": stats.get("pass_at_1", 0.0),
        "detailed_log": str(detailed_file)
    }

    with open(eval_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - Detailed JSONL: {detailed_file}")
    print(f"  - Summary JSON: {eval_file}")
    print(f"  - LiveCodeBench format: {gen_file}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench Evaluation Pipeline")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["deep_think", "deep_instruction", "diverse_think", "diverse_instruction", "all"],
        default="all",
        help="Model type to evaluate"
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Checkpoint steps to evaluate (e.g., --steps 500 600 700)"
    )

    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Include base model (without LoRA) in evaluation"
    )

    parser.add_argument(
        "--version",
        type=str,
        default="release_v5",
        help="LiveCodeBench version (default: release_v5)"
    )

    parser.add_argument(
        "--date_start",
        type=str,
        default="2408",
        help="Start date in YYMM format (default: 2408 = Aug 2024)"
    )

    parser.add_argument(
        "--date_end",
        type=str,
        default="2502",
        help="End date in YYMM format (default: 2502 = Feb 2025)"
    )

    parser.add_argument(
        "--platform",
        type=str,
        choices=["atcoder", "leetcode", "codeforces", "all"],
        default="all",
        help="Platform to evaluate (default: all)"
    )

    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty level to evaluate (default: all)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/livecodebench",
        help="Output directory for results"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./models",
        help="Directory containing model checkpoints"
    )

    args = parser.parse_args()

    # Update config
    config = CONFIG
    config.livecodebench_version = args.version
    config.output_base_dir = args.output_dir
    config.date_range_start = args.date_start
    config.date_range_end = args.date_end
    config.checkpoint_base_dir = args.checkpoint_dir

    print("="*80)
    print("LIVECODEBENCH EVALUATION PIPELINE")
    print("Author: naholav")
    print("="*80)
    print(f"Base model: {config.base_model}")
    print(f"Checkpoint directory: {config.checkpoint_base_dir}")
    print(f"LiveCodeBench version: {config.livecodebench_version}")
    print(f"Date range: {args.date_start} - {args.date_end}")
    print(f"Platform filter: {args.platform}")
    print(f"Difficulty filter: {args.difficulty}")
    print(f"Model type filter: {args.model_type}")
    print(f"Step filter: {args.steps or 'all'}")
    print(f"Include base model: {args.include_base}")
    print(f"Output directory: {config.output_base_dir}")
    print("="*80)

    if args.model_type == "all":
        model_types = list(config.model_types)
    else:
        model_types = [args.model_type]

    checkpoints = discover_checkpoints(
        config.checkpoint_base_dir,
        model_types=model_types,
        steps=args.steps
    )

    total_checkpoints = sum(len(v) for v in checkpoints.values())
    print(f"\nDiscovered {total_checkpoints} checkpoints:")
    for model_type, ckpts in checkpoints.items():
        print(f"  {model_type}: {len(ckpts)} checkpoints")

    problems = load_livecodebench(
        version=config.livecodebench_version,
        difficulty=args.difficulty if args.difficulty != "all" else None,
        date_start=args.date_start,
        date_end=args.date_end,
        platform=args.platform if args.platform != "all" else None
    )

    if not problems:
        print("\nERROR: No problems found with the given filters!")
        return

    print(f"\nTotal problems to evaluate: {len(problems)}")

    all_results = []
    eval_name = f"{args.date_start}-{args.date_end}"
    if args.platform != "all":
        eval_name += f"_{args.platform}"

    print(f"\n{'='*80}")
    print(f"EVALUATING {len(problems)} PROBLEMS")
    print("="*80)

    if args.include_base:
        result = evaluate_checkpoint(
            None, "base", problems, eval_name, config,
            is_base_model=True
        )
        all_results.append(result)

    for model_type, ckpt_paths in checkpoints.items():
        for ckpt_path in ckpt_paths:
            result = evaluate_checkpoint(
                ckpt_path, model_type, problems, eval_name, config
            )
            all_results.append(result)

    summary_file = Path(config.output_base_dir) / "summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": config.base_model,
            "livecodebench_version": config.livecodebench_version,
            "date_range": f"{args.date_start}-{args.date_end}",
            "platform": args.platform,
            "difficulty": args.difficulty,
            "model_types": model_types,
            "steps": args.steps,
            "num_problems": len(problems)
        },
        "results": all_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {config.output_base_dir}")
    print(f"Summary file: {summary_file}")

    print(f"\n{'Model':<50} {'Pass@1':<10} {'Problems':<10}")
    print("-"*70)
    for result in all_results:
        model_name = result.get("model_name", "unknown")
        pass_at_1 = result.get("pass_at_1", 0.0)
        num_problems = result.get("num_problems", 0)
        print(f"{model_name:<50} {pass_at_1*100:.1f}%{'':<5} {num_problems:<10}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
