#!/usr/bin/env python3
"""Validate the curated corpus used in the main paper."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeip.config import REFERENCE_LANG, TARGET_LANGS  # noqa: E402

ALGORITHMS = [
    "binary_search",
    "bubble_sort",
    "merge_sort",
    "quicksort",
    "sieve_of_eratosthenes",
    "matrix_multiply",
    "edit_distance",
    "longest_common_subsequence",
    "knapsack_01",
    "dfs_graph",
    "dijkstra",
    "topological_sort",
    "nqueens",
    "gcd_extended",
    "heap_sort",
    "insertion_sort",
    "bfs_graph",
    "counting_sort",
    "kmp_search",
    "tower_of_hanoi",
    "longest_increasing_subsequence",
    "binary_tree_inorder",
]

BOILERPLATE_PATTERNS = {
    "Python": [
        (r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', "if __name__ block"),
        (r'^def main\s*\(', "main() function"),
        (r'\bprint\s*\(', "print() statement"),
    ],
    "C++": [
        (r'int\s+main\s*\(', "main() function"),
        (r'\bcout\s*<<', "cout output"),
        (r'\bprintf\s*\(', "printf output"),
    ],
    "Java": [
        (r'public\s+static\s+void\s+main\s*\(', "main() method"),
        (r'System\.out', "System.out output"),
    ],
    "Go": [
        (r'^func\s+main\s*\(', "main() function"),
        (r'fmt\.Print', "fmt.Print output"),
    ],
    "Rust": [
        (r'^fn\s+main\s*\(', "main() function"),
        (r'println!\s*\(', "println! output"),
        (r'eprintln!\s*\(', "eprintln! output"),
    ],
    "JavaScript": [(r'console\.log\s*\(', "console.log output")],
    "TypeScript": [(r'console\.log\s*\(', "console.log output")],
    "Swift": [(r'(?<!\w)print\s*\((?!.*repeating)', "print() output")],
    "C#": [
        (r'static\s+void\s+Main\s*\(', "Main() method"),
        (r'Console\.Write', "Console.Write output"),
    ],
    "PHP": [
        (r'\becho\s+', "echo output"),
        (r'\bvar_dump\s*\(', "var_dump output"),
        (r'\bprint_r\s*\(', "print_r output"),
    ],
    "Ruby": [(r'\bputs\s+', "puts output")],
    "Lua": [
        (r'\bprint\s*\(', "print() output"),
        (r'\bio\.write\s*\(', "io.write output"),
    ],
    "R": [
        (r'\bcat\s*\(', "cat() output"),
        (r'\bprint\s*\(', "print() output"),
    ],
    "Julia": [(r'\bprintln\s*\(', "println() output")],
    "Kotlin": [
        (r'\bprintln\s*\(', "println() output"),
        (r'fun\s+main\s*\(', "main() function"),
    ],
    "Scala": [
        (r'\bprintln\s*\(', "println() output"),
        (r'def\s+main\s*\(', "main() method"),
    ],
    "D": [
        (r'\bwriteln\s*\(', "writeln output"),
        (r'\bwritefln\s*\(', "writefln output"),
        (r'void\s+main\s*\(', "main() function"),
    ],
    "Perl": [
        (r'\bprint\s+', "print output"),
        (r'\bsay\s+', "say output"),
    ],
    "Bash": [],
    "Racket": [
        (r'\(display\s+', "display output"),
        (r'\(printf\s+', "printf output"),
        (r'\(displayln\s+', "displayln output"),
    ],
}


def load_corpus(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_boilerplate(code: str, lang: str) -> list[tuple[str, str]]:
    issues = []
    for line in code.split("\n"):
        for pattern, desc in BOILERPLATE_PATTERNS.get(lang, []):
            if re.search(pattern, line):
                issues.append((desc, line.strip()[:80]))
    return issues


def validate_clean_corpus(corpus: dict) -> dict:
    results = {}
    for algo in ALGORITHMS:
        results[algo] = {}
        if algo not in corpus:
            for lang in TARGET_LANGS:
                results[algo][lang] = [("MISSING", f"Algorithm '{algo}' not in corpus")]
            continue

        for lang in sorted(TARGET_LANGS):
            issues = []
            if lang not in corpus[algo]:
                issues.append(("MISSING", f"No {lang} implementation"))
            else:
                code = corpus[algo][lang]
                issues.extend(check_boilerplate(code, lang))
                if not code.strip():
                    issues.append(("EMPTY", "Implementation is empty"))
            results[algo][lang] = issues
    return results


def print_report(results: dict, *, summary_only: bool = False) -> int:
    total_issues = 0
    issue_langs = set()
    issue_algos = set()

    for algo in ALGORITHMS:
        for lang in sorted(TARGET_LANGS):
            issues = results.get(algo, {}).get(lang, [])
            total_issues += len(issues)
            if issues:
                issue_langs.add(lang)
                issue_algos.add(algo)

    print("\n" + "=" * 80)
    print("  CURATED CORPUS VALIDATION")
    print("=" * 80)
    print(f"  Algorithms checked: {len(ALGORITHMS)}")
    print(f"  Reference language: {REFERENCE_LANG}")
    print(f"  Languages checked:  {len(TARGET_LANGS)}")
    print(f"  Total issues:       {total_issues}")
    print(f"  Algorithms with issues: {len(issue_algos)}/{len(ALGORITHMS)}")
    print(f"  Languages with issues:  {len(issue_langs)}/{len(TARGET_LANGS)}")

    if summary_only or total_issues == 0:
        return total_issues

    for algo in ALGORITHMS:
        algo_issues = {
            lang: issues
            for lang, issues in results.get(algo, {}).items()
            if issues
        }
        if not algo_issues:
            continue
        print(f"\n  --- {algo} ---")
        for lang, issues in sorted(algo_issues.items()):
            for kind, detail in issues:
                print(f"    {lang:>12s}  [{kind}] {detail}")
    return total_issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the curated corpus")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "curated_corpus.json",
        help="Path to curated_corpus.json",
    )
    parser.add_argument("--summary", action="store_true", help="Only print summary counts")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: {args.corpus} not found")
        print("Provide --corpus or place curated_corpus.json under data/.")
        return 1

    corpus = load_corpus(args.corpus)
    results = validate_clean_corpus(corpus)
    return 1 if print_report(results, summary_only=args.summary) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
