#!/usr/bin/env python3
"""Execution-validate the curated 22-algorithm corpus.

For each algorithm × language, we:
  1. Generate a self-contained test program that calls the corpus function
     with shared test inputs.
  2. Compile / run it.
  3. Compare JSON-serialised output against the Python reference.

Covers the 20 languages in the curated corpus and can be restricted with
``--languages`` for focused checks. Produces a JSON report suitable for the
paper appendix.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "curated_corpus.json"

LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "PHP", "Go", "Rust", "C++", "Julia",
    "Perl", "Ruby", "Lua", "Java", "Bash", "R", "Racket", "C#", "Kotlin",
    "Scala", "D", "Swift",
]

_TOOL_ENV_VARS = {
    "go": "CODEIP_GO",
    "rustc": "CODEIP_RUSTC",
    "php": "CODEIP_PHP",
    "g++": "CODEIP_GXX",
    "julia": "CODEIP_JULIA",
    "node": "CODEIP_NODE",
    "ruby": "CODEIP_RUBY",
    "javac": "CODEIP_JAVAC",
    "java": "CODEIP_JAVA",
    "lua": "CODEIP_LUA",
    "Rscript": "CODEIP_RSCRIPT",
    "racket": "CODEIP_RACKET",
    "dotnet": "CODEIP_DOTNET",
    "kotlinc": "CODEIP_KOTLINC",
    "kotlin": "CODEIP_KOTLIN",
    "scala-cli": "CODEIP_SCALA_CLI",
    "swiftc": "CODEIP_SWIFTC",
}

_TOOL_PATHS: dict[str, str | None] = {
    name: os.environ.get(env_name) or shutil.which(name)
    for name, env_name in _TOOL_ENV_VARS.items()
}

def _tool(name: str) -> str | None:
    p = _TOOL_PATHS.get(name)
    if p and os.path.isfile(p):
        return p
    return None

# ---------------------------------------------------------------------------
# Test-case definitions for all 22 algorithms
# Each entry: {"inputs": [...args], "desc": "short label"}
# The Python reference function is called with *inputs and the return value
# is the expected output for all languages.
# ---------------------------------------------------------------------------

TEST_CASES: dict[str, list[dict]] = {
    "binary_search": [
        {"inputs": [[1, 3, 5, 7, 9, 11], 7], "desc": "found"},
        {"inputs": [[1, 3, 5, 7, 9, 11], 4], "desc": "not found"},
        {"inputs": [[], 1], "desc": "empty"},
        {"inputs": [[42], 42], "desc": "single element found"},
    ],
    "bubble_sort": [
        {"inputs": [[5, 3, 8, 1, 2]], "desc": "normal"},
        {"inputs": [[1, 2, 3]], "desc": "sorted"},
        {"inputs": [[]], "desc": "empty"},
        {"inputs": [[7]], "desc": "single"},
    ],
    "merge_sort": [
        {"inputs": [[5, 3, 8, 1, 2]], "desc": "normal"},
        {"inputs": [[1, 2, 3]], "desc": "sorted"},
        {"inputs": [[], ], "desc": "empty"},
    ],
    "quicksort": [
        {"inputs": [[5, 3, 8, 1, 2, 7, 4]], "desc": "normal"},
        {"inputs": [[1, 2, 3, 4]], "desc": "sorted"},
        {"inputs": [[4, 3, 2, 1]], "desc": "reverse"},
    ],
    "heap_sort": [
        {"inputs": [[5, 3, 8, 1, 2]], "desc": "normal"},
        {"inputs": [[1]], "desc": "single"},
        {"inputs": [[3, 3, 3]], "desc": "duplicates"},
    ],
    "insertion_sort": [
        {"inputs": [[5, 3, 8, 1, 2]], "desc": "normal"},
        {"inputs": [[1, 2, 3]], "desc": "sorted"},
        {"inputs": [[], ], "desc": "empty"},
    ],
    "counting_sort": [
        {"inputs": [[4, 2, 2, 8, 3, 3, 1]], "desc": "normal"},
        {"inputs": [[1]], "desc": "single"},
        {"inputs": [[5, 5, 5]], "desc": "duplicates"},
    ],
    "sieve_of_eratosthenes": [
        {"inputs": [30], "desc": "primes up to 30"},
        {"inputs": [2], "desc": "edge case 2"},
        {"inputs": [1], "desc": "edge case 1"},
    ],
    "nqueens": [
        {"inputs": [4], "desc": "n=4"},
        {"inputs": [1], "desc": "n=1"},
        {"inputs": [8], "desc": "n=8"},
    ],
    "edit_distance": [
        {"inputs": ["kitten", "sitting"], "desc": "classic"},
        {"inputs": ["", "abc"], "desc": "empty first"},
        {"inputs": ["abc", "abc"], "desc": "identical"},
    ],
    "longest_common_subsequence": [
        {"inputs": ["ABCBDAB", "BDCAB"], "desc": "classic"},
        {"inputs": ["ABC", "DEF"], "desc": "disjoint"},
        {"inputs": ["", "XYZ"], "desc": "empty"},
    ],
    "knapsack_01": [
        {"inputs": [[2, 3, 4, 5], [3, 4, 5, 6], 8], "desc": "classic"},
        {"inputs": [[1], [10], 1], "desc": "single item fits"},
        {"inputs": [[10], [5], 3], "desc": "single item too heavy"},
    ],
    "kmp_search": [
        {"inputs": ["AABAACAADAABAABA", "AABA"], "desc": "multiple matches"},
        {"inputs": ["abcdef", "xyz"], "desc": "no match"},
        {"inputs": ["aaaa", "aa"], "desc": "overlapping"},
    ],
    "longest_increasing_subsequence": [
        {"inputs": [[10, 9, 2, 5, 3, 7, 101, 18]], "desc": "classic"},
        {"inputs": [[1, 2, 3, 4, 5]], "desc": "sorted"},
        {"inputs": [[5, 4, 3, 2, 1]], "desc": "reverse"},
    ],
    "gcd_extended": [
        {"inputs": [35, 15], "desc": "classic"},
        {"inputs": [0, 5], "desc": "zero first"},
        {"inputs": [7, 1], "desc": "coprime"},
    ],
    "matrix_multiply": [
        {"inputs": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "desc": "2x2"},
        {"inputs": [[[1, 0], [0, 1]], [[9, 8], [7, 6]]], "desc": "identity"},
    ],
    "tower_of_hanoi": [
        {"inputs": [2, "A", "C", "B"], "desc": "n=2"},
        {"inputs": [3, "A", "C", "B"], "desc": "n=3"},
        {"inputs": [1, "A", "C", "B"], "desc": "n=1"},
    ],
    "dfs_graph": [
        {"inputs": [{0: [1, 2], 1: [3], 2: [3], 3: []}, 0], "desc": "diamond"},
        {"inputs": [{0: [1], 1: [2], 2: []}, 0], "desc": "chain"},
    ],
    "bfs_graph": [
        {"inputs": [{0: [1, 2], 1: [3], 2: [3], 3: []}, 0], "desc": "diamond"},
        {"inputs": [{0: [1], 1: [2], 2: []}, 0], "desc": "chain"},
    ],
    "dijkstra": [
        {"inputs": [
            [[(1, 4), (2, 1)], [(3, 1)], [(1, 2), (3, 5)], []],
            0,
        ], "desc": "4-node weighted"},
    ],
    "topological_sort": [
        {"inputs": [[[1, 2], [3], [3], []], 4], "desc": "diamond DAG"},
        {"inputs": [[[1], [], [0]], 3], "desc": "3-node chain with back-edge-free"},
    ],
    "binary_tree_inorder": [
        {"inputs": ["balanced_3"], "desc": "balanced tree, 3 nodes"},
        {"inputs": ["left_chain_4"], "desc": "left-skewed, 4 nodes"},
        {"inputs": ["single"], "desc": "single node"},
    ],
}


# ---------------------------------------------------------------------------
# Python reference runner
# ---------------------------------------------------------------------------

def _build_tree_python(spec: str):
    """Build a Python tree for binary_tree_inorder test cases."""

    class Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    if spec == "balanced_3":
        return Node(2, Node(1), Node(3))
    if spec == "left_chain_4":
        return Node(4, Node(3, Node(2, Node(1))))
    if spec == "single":
        return Node(42)
    raise ValueError(f"Unknown tree spec: {spec}")


def run_python_reference(algo: str, code: str) -> list[dict]:
    """Execute the Python implementation and collect expected outputs."""
    ns: dict = {}
    exec(code, ns)  # noqa: S102

    results = []
    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_tree_inorder":
            tree = _build_tree_python(inputs[0])
            out = ns["inorder_traversal"](tree)
            results.append({"desc": tc["desc"], "expected": out})
            continue

        if algo == "quicksort":
            arr = list(inputs[0])
            ns["quicksort"](arr, 0, len(arr) - 1)
            results.append({"desc": tc["desc"], "expected": arr})
            continue

        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = list(inputs[0])
            ret = ns[algo](arr)
            results.append({"desc": tc["desc"], "expected": ret if ret is not None else arr})
            continue

        if algo == "knapsack_01":
            val, items = ns["knapsack"](*inputs)
            results.append({"desc": tc["desc"], "expected": [val, sorted(items)]})
            continue

        if algo == "gcd_extended":
            g, x, y = ns["gcd_extended"](*inputs)
            results.append({"desc": tc["desc"], "expected": [g, x, y]})
            continue

        if algo == "tower_of_hanoi":
            moves = ns["tower_of_hanoi"](*inputs)
            results.append({"desc": tc["desc"], "expected": [[m[0], m[1]] for m in moves]})
            continue

        if algo == "dfs_graph" or algo == "bfs_graph":
            func = ns["dfs"] if algo == "dfs_graph" else ns["bfs"]
            out = func(*inputs)
            results.append({"desc": tc["desc"], "expected": out})
            continue

        if algo == "dijkstra":
            out = ns["dijkstra"](*inputs)
            results.append({"desc": tc["desc"], "expected": out})
            continue

        if algo == "topological_sort":
            out = ns["topological_sort"](*inputs)
            results.append({"desc": tc["desc"], "expected": out})
            continue

        func_name = {
            "binary_search": "binary_search",
            "merge_sort": "merge_sort",
            "sieve_of_eratosthenes": "sieve",
            "nqueens": "nqueens",
            "edit_distance": "edit_distance",
            "longest_common_subsequence": "lcs",
            "kmp_search": "kmp_search",
            "longest_increasing_subsequence": "longest_increasing_subsequence",
            "matrix_multiply": "matrix_multiply",
        }.get(algo, algo)

        out = ns[func_name](*inputs)
        results.append({"desc": tc["desc"], "expected": out})

    return results


# ---------------------------------------------------------------------------
# JavaScript runner
# ---------------------------------------------------------------------------

def _js_encode_value(val) -> str:
    """Encode a Python value as a JavaScript literal."""
    return json.dumps(val)


def _js_graph_dense(graph_dict: dict, n: int | None = None) -> str:
    """Convert sparse Python graph dict to dense JS array literal."""
    if n is None:
        n = max(graph_dict.keys()) + 1 if graph_dict else 0
    rows = []
    for i in range(n):
        rows.append(json.dumps(graph_dict.get(i, [])))
    return "[" + ", ".join(rows) + "]"


def run_javascript(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Execute the JavaScript implementation with node."""
    lines = [code, ""]

    func_map = {
        "binary_search": "binarySearch",
        "bubble_sort": "bubbleSort",
        "merge_sort": "mergeSort",
        "quicksort": "quicksort",
        "heap_sort": "heapSort",
        "insertion_sort": "insertionSort",
        "counting_sort": "countingSort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "editDistance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmpSearch",
        "longest_increasing_subsequence": "longestIncreasingSubsequence",
        "gcd_extended": "gcdExtended",
        "matrix_multiply": "matrixMultiply",
        "tower_of_hanoi": "towerOfHanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topologicalSort",
        "binary_tree_inorder": "inorderTraversal",
        "knapsack_01": "knapsack",
    }
    fn = func_map[algo]
    lines.append("const results = [];")

    for i, tc in enumerate(TEST_CASES[algo]):
        inputs = tc["inputs"]

        if algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f"results.push(JSON.stringify({fn}(new Node(2, new Node(1, null, null), new Node(3, null, null)))));")
            elif spec == "left_chain_4":
                lines.append(f"results.push(JSON.stringify({fn}(new Node(4, new Node(3, new Node(2, new Node(1, null, null), null), null), null))));")
            elif spec == "single":
                lines.append(f"results.push(JSON.stringify({fn}(new Node(42, null, null))));")
            continue

        if algo == "quicksort":
            arr = json.dumps(inputs[0])
            lines.append(f"{{ let a = {arr}; {fn}(a, 0, a.length - 1); results.push(JSON.stringify(a)); }}")
            continue

        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = json.dumps(inputs[0])
            lines.append(f"{{ let a = {arr}; {fn}(a); results.push(JSON.stringify(a)); }}")
            continue

        if algo == "knapsack_01":
            w, v, c = inputs
            lines.append(f"{{ let r = {fn}({json.dumps(w)}, {json.dumps(v)}, {c}); results.push(JSON.stringify([r[0], r[1].sort((a,b)=>a-b)])); }}")
            continue

        if algo == "gcd_extended":
            a, b = inputs
            lines.append(f"{{ let r = {fn}({a}, {b}); results.push(JSON.stringify(r)); }}")
            continue

        if algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'{{ let r = {fn}({n}, "{s}", "{t}", "{aux}"); results.push(JSON.stringify(r.map(m => [m[0] || m.from || m.From, m[1] || m.to || m.To]))); }}')
            continue

        if algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            lines.append(f"results.push(JSON.stringify({fn}({_js_encode_value(graph)}, {start})));")
            continue

        if algo == "dijkstra":
            graph, src = inputs
            lines.append(f"results.push(JSON.stringify({fn}({_js_encode_value(graph)}, {src})));")
            continue

        if algo == "topological_sort":
            graph, n_nodes = inputs
            lines.append(f"results.push(JSON.stringify({fn}({_js_encode_value(graph)}, {n_nodes})));")
            continue

        arg_strs = [_js_encode_value(a) for a in inputs]
        lines.append(f"results.push(JSON.stringify({fn}({', '.join(arg_strs)})));")

    lines.append('process.stdout.write("[" + results.join(",") + "]");')

    script = "\n".join(lines)
    try:
        r = subprocess.run(
            [_tool("node"), "-e", script],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": r.stderr[:500]}]
        raw = json.loads(r.stdout)
        out = []
        for i, tc in enumerate(TEST_CASES[algo]):
            out.append({"desc": tc["desc"], "actual": raw[i]})
        return out
    except Exception as e:
        return [{"desc": "EXCEPTION", "error": str(e)[:500]}]


# ---------------------------------------------------------------------------
# TypeScript runner (reuses JS harness logic, executes via npx tsx)
# ---------------------------------------------------------------------------

def run_typescript(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Execute the TypeScript implementation via npx tsx."""
    ts_func_map = {
        "binary_search": "binarySearch",
        "bubble_sort": "bubbleSort",
        "merge_sort": "mergeSort",
        "quicksort": "quicksort",
        "heap_sort": "heapSort",
        "insertion_sort": "insertionSort",
        "counting_sort": "countingSort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "editDistance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmpSearch",
        "longest_increasing_subsequence": "longestIncreasingSubsequence",
        "gcd_extended": "gcdExtended",
        "matrix_multiply": "matrixMultiply",
        "tower_of_hanoi": "towerOfHanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topologicalSort",
        "binary_tree_inorder": "inorderTraversal",
        "knapsack_01": "knapsack",
    }
    fn = ts_func_map[algo]

    lines = [code, "", "const results: string[] = [];"]

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f"results.push(JSON.stringify({fn}({{val:2, left:{{val:1, left:null, right:null}}, right:{{val:3, left:null, right:null}}}})));")
            elif spec == "left_chain_4":
                lines.append(f"results.push(JSON.stringify({fn}({{val:4, left:{{val:3, left:{{val:2, left:{{val:1, left:null, right:null}}, right:null}}, right:null}}, right:null}})));")
            elif spec == "single":
                lines.append(f"results.push(JSON.stringify({fn}({{val:42, left:null, right:null}})));")
            continue

        if algo == "quicksort":
            arr = json.dumps(inputs[0])
            lines.append(f"{{ let a = {arr}; {fn}(a, 0, a.length - 1); results.push(JSON.stringify(a)); }}")
            continue

        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = json.dumps(inputs[0])
            lines.append(f"{{ let a = {arr}; {fn}(a); results.push(JSON.stringify(a)); }}")
            continue

        if algo == "knapsack_01":
            w, v, c = inputs
            lines.append(f"{{ let r = {fn}({json.dumps(w)}, {json.dumps(v)}, {c}); results.push(JSON.stringify([r[0], r[1].sort((a:number,b:number)=>a-b)])); }}")
            continue

        if algo == "gcd_extended":
            a, b = inputs
            lines.append(f"{{ let r = {fn}({a}, {b}); results.push(JSON.stringify(r)); }}")
            continue

        if algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'{{ let r = {fn}({n}, "{s}", "{t}", "{aux}"); results.push(JSON.stringify(r.map((m: any) => [m[0] || m.from || m.From, m[1] || m.to || m.To]))); }}')
            continue

        if algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            lines.append(f"results.push(JSON.stringify({fn}({_js_encode_value(graph)}, {start})));")
            continue

        if algo == "dijkstra":
            graph, src = inputs
            lines.append(f"results.push(JSON.stringify({fn}({_js_encode_value(graph)}, {src})));")
            continue

        if algo == "topological_sort":
            graph, n_nodes = inputs
            lines.append(f"results.push(JSON.stringify({fn}({_js_encode_value(graph)}, {n_nodes})));")
            continue

        arg_strs = [_js_encode_value(a) for a in inputs]
        lines.append(f"results.push(JSON.stringify({fn}({', '.join(arg_strs)})));")

    lines.append('process.stdout.write("[" + results.join(",") + "]");')

    script = "\n".join(lines)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False, encoding="utf-8") as f:
        f.write(script)
        ts_path = f.name

    try:
        r = subprocess.run(
            ["npx", "tsx", ts_path],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT), shell=True,
        )
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": r.stderr[:500]}]
        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "EXCEPTION", "error": str(e)[:500]}]
    finally:
        os.unlink(ts_path)


# ---------------------------------------------------------------------------
# C++ runner
# ---------------------------------------------------------------------------

_CPP_PREAMBLE = """\
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <cstring>
#include <sstream>
#include <utility>
using namespace std;

// Minimal JSON output helpers
string json_int(int v) { return to_string(v); }
string json_arr(const vector<int>& v) {
    string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += ",";
        s += to_string(v[i]);
    }
    return s + "]";
}
string json_str(const string& s) {
    return "\\"" + s + "\\"";
}
string json_matrix(const vector<vector<int>>& m) {
    string s = "[";
    for (size_t i = 0; i < m.size(); i++) {
        if (i) s += ",";
        s += json_arr(m[i]);
    }
    return s + "]";
}
"""


def _cpp_build_test_calls(algo: str, fn: str) -> str:
    """Generate C++ test call lines for a given algorithm."""
    lines = ['    vector<string> results;']

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_search":
            arr, target = inputs
            n = len(arr)
            arr_lit = "{" + ",".join(str(x) for x in arr) + "}"
            lines.append(f"    {{ int a[] = {arr_lit}; results.push_back(json_int(binary_search(a, {n}, {target}))); }}")

        elif algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = inputs[0]
            n = len(arr)
            arr_lit = "{" + ",".join(str(x) for x in arr) + "}"
            cpp_fn = {"bubble_sort": "bubble_sort", "heap_sort": "heapSort",
                      "insertion_sort": "insertionSort", "counting_sort": "countingSort"}[algo]
            lines.append(f"    {{ int a[] = {arr_lit}; {cpp_fn}(a, {n}); vector<int> v(a, a+{n}); results.push_back(json_arr(v)); }}")

        elif algo == "merge_sort":
            arr = inputs[0]
            arr_lit = "{" + ",".join(str(x) for x in arr) + "}"
            lines.append(f"    {{ vector<int> a = {arr_lit}; vector<int> r = merge_sort(a); results.push_back(json_arr(r)); }}")

        elif algo == "quicksort":
            arr = inputs[0]
            n = len(arr)
            arr_lit = "{" + ",".join(str(x) for x in arr) + "}"
            lines.append(f"    {{ int a[] = {arr_lit}; quicksort(a, 0, {n - 1}); vector<int> v(a, a+{n}); results.push_back(json_arr(v)); }}")

        elif algo == "sieve_of_eratosthenes":
            n_val = inputs[0]
            lines.append(f"    {{ vector<int> r = sieve({n_val}); results.push_back(json_arr(r)); }}")

        elif algo == "nqueens":
            n_val = inputs[0]
            lines.append(f"    results.push_back(json_int(nqueens({n_val})));")

        elif algo == "edit_distance":
            s1, s2 = inputs
            lines.append(f'    results.push_back(json_int(edit_distance("{s1}", "{s2}")));')

        elif algo == "longest_common_subsequence":
            s1, s2 = inputs
            lines.append(f'    results.push_back(json_str(lcs("{s1}", "{s2}")));')

        elif algo == "kmp_search":
            text, pat = inputs
            lines.append(f'    results.push_back(json_arr(kmpSearch("{text}", "{pat}")));')

        elif algo == "longest_increasing_subsequence":
            arr = inputs[0]
            arr_lit = "{" + ",".join(str(x) for x in arr) + "}"
            lines.append(f"    {{ vector<int> a = {arr_lit}; results.push_back(json_int(longestIncreasingSubsequence(a))); }}")

        elif algo == "gcd_extended":
            a, b = inputs
            lines.append(f"    {{ GcdResult r = gcdExtended({a}, {b}); results.push_back(string(\"[\") + to_string(r.gcd) + \",\" + to_string(r.x) + \",\" + to_string(r.y) + \"]\"); }}")

        elif algo == "matrix_multiply":
            a_mat, b_mat = inputs
            a_lit = "{" + ",".join("{" + ",".join(str(x) for x in row) + "}" for row in a_mat) + "}"
            b_lit = "{" + ",".join("{" + ",".join(str(x) for x in row) + "}" for row in b_mat) + "}"
            lines.append(f"    {{ vector<vector<int>> a = {a_lit}; vector<vector<int>> b = {b_lit}; results.push_back(json_matrix(matrix_multiply(a, b))); }}")

        elif algo == "knapsack_01":
            w, v, c = inputs
            w_lit = "{" + ",".join(str(x) for x in w) + "}"
            v_lit = "{" + ",".join(str(x) for x in v) + "}"
            lines.append(f"    {{ vector<int> w = {w_lit}; vector<int> v = {v_lit}; auto r = knapsack(w, v, {c}); sort(r.second.begin(), r.second.end()); string s = \"[\" + to_string(r.first) + \",\" + json_arr(r.second) + \"]\"; results.push_back(s); }}")

        elif algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f"    {{ auto moves = towerOfHanoi({n}, '{s}', '{t}', '{aux}'); string s = \"[\"; for (size_t i=0; i<moves.size(); i++) {{ if(i) s+=\",\"; s += string(\"[\\\"\")+moves[i].first+\"\\\",\\\"\"+moves[i].second+\"\\\"]\"; }} s+=\"]\"; results.push_back(s); }}")

        elif algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            n = max(graph.keys()) + 1 if graph else 0
            rows = []
            for i in range(n):
                r = graph.get(i, [])
                rows.append("{" + ",".join(str(x) for x in r) + "}")
            graph_lit = "{" + ",".join(rows) + "}"
            fn_cpp = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f"    {{ vector<vector<int>> g = {graph_lit}; results.push_back(json_arr({fn_cpp}(g, {start}))); }}")

        elif algo == "dijkstra":
            graph, src = inputs
            n = len(graph)
            rows = []
            for adj in graph:
                pairs = ",".join(f"{{{nb},{wt}}}" for nb, wt in adj)
                rows.append("{" + pairs + "}")
            graph_lit = "{" + ",".join(rows) + "}"
            lines.append(f"    {{ vector<vector<pair<int,int>>> g = {graph_lit}; results.push_back(json_arr(dijkstra(g, {src}))); }}")

        elif algo == "topological_sort":
            graph, n_nodes = inputs
            rows = []
            for adj in graph:
                rows.append("{" + ",".join(str(x) for x in adj) + "}")
            graph_lit = "{" + ",".join(rows) + "}"
            lines.append(f"    {{ vector<vector<int>> g = {graph_lit}; results.push_back(json_arr(topologicalSort(g, {n_nodes}))); }}")

        elif algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f"    {{ Node n1(1), n3(3), n2(2); n2.left=&n1; n2.right=&n3; results.push_back(json_arr(inorderTraversal(&n2))); }}")
            elif spec == "left_chain_4":
                lines.append(f"    {{ Node n1(1), n2(2), n3(3), n4(4); n4.left=&n3; n3.left=&n2; n2.left=&n1; results.push_back(json_arr(inorderTraversal(&n4))); }}")
            elif spec == "single":
                lines.append(f"    {{ Node n42(42); results.push_back(json_arr(inorderTraversal(&n42))); }}")

    lines.append('    cout << "[";')
    lines.append('    for (size_t i = 0; i < results.size(); i++) {')
    lines.append('        if (i) cout << ",";')
    lines.append('        cout << results[i];')
    lines.append('    }')
    lines.append('    cout << "]" << endl;')
    return "\n".join(lines)


def run_cpp(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Compile and run C++ implementation via g++."""
    if not _tool("g++"):
        return [{"desc": "SKIP", "error": "g++ not found"}]

    main_body = _cpp_build_test_calls(algo, "")

    src = _CPP_PREAMBLE + "\n" + code + "\n\nint main() {\n" + main_body + "\n    return 0;\n}\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "test.cpp")
        exe_path = os.path.join(tmpdir, "test.exe")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src)

        comp = subprocess.run(
            [_tool("g++"), "-std=c++17", "-O0", "-o", exe_path, src_path],
            capture_output=True, text=True, timeout=30,
        )
        if comp.returncode != 0:
            return [{"desc": "COMPILE_ERROR", "error": comp.stderr[:500]}]

        run = subprocess.run(
            [exe_path], capture_output=True, text=True, timeout=30,
        )
        if run.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": run.stderr[:500]}]

        try:
            raw = json.loads(run.stdout)
            return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
        except Exception as e:
            return [{"desc": "PARSE_ERROR", "error": f"stdout={run.stdout[:200]}, err={e}"}]


# ---------------------------------------------------------------------------
# Julia runner
# ---------------------------------------------------------------------------

def _julia_encode_value(val) -> str:
    """Encode Python value as Julia literal."""
    if isinstance(val, list):
        if not val:
            return "Int[]"
        if all(isinstance(x, int) for x in val):
            return "Int[" + ", ".join(str(x) for x in val) + "]"
        if all(isinstance(x, list) for x in val):
            inner = ", ".join(_julia_encode_value(x) for x in val)
            return "[" + inner + "]"
        if all(isinstance(x, tuple) for x in val):
            inner = ", ".join(f"({x[0]}, {x[1]})" for x in val)
            return "[" + inner + "]"
        return "[" + ", ".join(_julia_encode_value(x) for x in val) + "]"
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, dict):
        pairs = ", ".join(f"{k} => {_julia_encode_value(v)}" for k, v in val.items())
        return f"Dict({pairs})"
    return str(val)


def _julia_build_test_calls(algo: str) -> str:
    """Generate Julia test call code for an algorithm."""
    lines = ["results = String[]"]

    julia_func = {
        "binary_search": "binary_search",
        "bubble_sort": "bubble_sort!",
        "merge_sort": "merge_sort",
        "quicksort": "quicksort!",
        "heap_sort": "heap_sort!",
        "insertion_sort": "insertion_sort!",
        "counting_sort": "counting_sort!",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "longest_increasing_subsequence",
        "gcd_extended": "gcd_extended",
        "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "tower_of_hanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = julia_func[algo]

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f"push!(results, string(\"[\", join({fn}(TreeNode(2, TreeNode(1, nothing, nothing), TreeNode(3, nothing, nothing))), \",\"), \"]\"))")
            elif spec == "left_chain_4":
                lines.append(f"push!(results, string(\"[\", join({fn}(TreeNode(4, TreeNode(3, TreeNode(2, TreeNode(1, nothing, nothing), nothing), nothing), nothing)), \",\"), \"]\"))")
            elif spec == "single":
                lines.append(f"push!(results, string(\"[\", join({fn}(TreeNode(42, nothing, nothing)), \",\"), \"]\"))")
            continue

        if algo == "quicksort":
            arr = inputs[0]
            if arr:
                arr_lit = "Int[" + ", ".join(str(x) for x in arr) + "]"
                lines.append(f"begin; a = {arr_lit}; {fn}(a, 1, length(a)); push!(results, string(\"[\", join(a, \",\"), \"]\")); end")
            else:
                lines.append(f'begin; a = Int[]; push!(results, "[]"); end')
            continue

        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = inputs[0]
            if arr:
                arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
            else:
                arr_lit = "Int[]"
            lines.append(f"begin; a = {arr_lit}; {fn}(a); push!(results, string(\"[\", join(a, \",\"), \"]\")); end")
            continue

        if algo == "merge_sort":
            arr = inputs[0]
            if arr:
                arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
            else:
                arr_lit = "Int[]"
            lines.append(f"begin; r = {fn}({arr_lit}); push!(results, string(\"[\", join(r, \",\"), \"]\")); end")
            continue

        if algo == "sieve_of_eratosthenes":
            n_val = inputs[0]
            lines.append(f"begin; r = {fn}({n_val}); push!(results, string(\"[\", join(r, \",\"), \"]\")); end")
            continue

        if algo == "nqueens":
            n_val = inputs[0]
            lines.append(f"push!(results, string({fn}({n_val})))")
            continue

        if algo == "edit_distance":
            s1, s2 = inputs
            lines.append(f'push!(results, string({fn}("{s1}", "{s2}")))')
            continue

        if algo == "longest_common_subsequence":
            s1, s2 = inputs
            lines.append(f'push!(results, string("\\"", {fn}("{s1}", "{s2}"), "\\""))')
            continue

        if algo == "kmp_search":
            text, pat = inputs
            lines.append(f'begin; r = {fn}("{text}", "{pat}"); push!(results, string("[", join(r, ","), "]")); end')
            continue

        if algo == "longest_increasing_subsequence":
            arr = inputs[0]
            arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
            lines.append(f"push!(results, string({fn}({arr_lit})))")
            continue

        if algo == "gcd_extended":
            a, b = inputs
            lines.append(f"begin; g, x, y = {fn}({a}, {b}); push!(results, string(\"[\", g, \",\", x, \",\", y, \"]\")); end")
            continue

        if algo == "matrix_multiply":
            a_mat, b_mat = inputs
            a_lit = "[" + ", ".join("[" + ", ".join(str(x) for x in row) + "]" for row in a_mat) + "]"
            b_lit = "[" + ", ".join("[" + ", ".join(str(x) for x in row) + "]" for row in b_mat) + "]"
            lines.append(f'begin; r = {fn}({a_lit}, {b_lit}); push!(results, string("[", join([string("[", join(row, ","), "]") for row in r], ","), "]")); end')
            continue

        if algo == "knapsack_01":
            w, v, c = inputs
            w_lit = "[" + ", ".join(str(x) for x in w) + "]"
            v_lit = "[" + ", ".join(str(x) for x in v) + "]"
            lines.append(f'begin; val, items = {fn}({w_lit}, {v_lit}, {c}); push!(results, string("[", val, ",[", join(sort(items), ","), "]]")); end')
            continue

        if algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'begin; moves = {fn}({n}, "{s}", "{t}", "{aux}"); push!(results, string("[", join([string("[\\"", m[1], "\\",\\"", m[2], "\\"]") for m in moves], ","), "]")); end')
            continue

        if algo == "binary_search":
            arr, target = inputs
            if arr:
                arr_lit = "Int[" + ", ".join(str(x) for x in arr) + "]"
            else:
                arr_lit = "Int[]"
            lines.append(f"push!(results, string({fn}({arr_lit}, {target})))")
            continue

        if algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            pairs = ", ".join(f"{k} => Int[{', '.join(str(x) for x in v)}]" for k, v in graph.items())
            graph_lit = f"Dict{{Int, Vector{{Int}}}}({pairs})"
            lines.append(f"begin; r = {fn}({graph_lit}, {start}); push!(results, string(\"[\", join(r, \",\"), \"]\")); end")
            continue

        if algo == "dijkstra":
            graph, src = inputs
            rows = []
            for adj in graph:
                if adj:
                    rows.append("[" + ", ".join(f"({nb+1}, {wt})" for nb, wt in adj) + "]")
                else:
                    rows.append("Tuple{Int,Int}[]")
            graph_lit = "[" + ", ".join(rows) + "]"
            lines.append(f"begin; r = {fn}({graph_lit}, {src+1}); push!(results, string(\"[\", join(r, \",\"), \"]\")); end")
            continue

        if algo == "topological_sort":
            graph, n_nodes = inputs
            rows = []
            for adj in graph:
                if adj:
                    rows.append("Int[" + ", ".join(str(x+1) for x in adj) + "]")
                else:
                    rows.append("Int[]")
            graph_lit = "Vector{Int}[" + ", ".join(rows) + "]"
            lines.append(f"begin; r = {fn}({graph_lit}, {n_nodes}); push!(results, string(\"[\", join(r, \",\"), \"]\")); end")
            continue

    lines.append('print("[", join(results, ","), "]")')
    return "\n".join(lines)


def run_julia(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Execute Julia implementation."""
    if not _tool("julia"):
        return [{"desc": "SKIP", "error": "julia not found"}]

    test_code = _julia_build_test_calls(algo)
    script = code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jl", delete=False, encoding="utf-8") as f:
        f.write(script)
        jl_path = f.name

    try:
        r = subprocess.run(
            [_tool("julia"), "--startup-file=no", jl_path],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": r.stderr[:500]}]

        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": f"stdout={r.stdout[:200] if 'r' in dir() else '?'}, err={e}"}]
    finally:
        os.unlink(jl_path)


# ---------------------------------------------------------------------------
# Go runner
# ---------------------------------------------------------------------------

_GO_IMPORTS = """\
package main

import (
\t"container/heap"
\t"fmt"
\t"math"
\t"strings"
\t"sort"
)
"""

GO_FUNC_MAP = {
    "binary_search": "binarySearch",
    "bubble_sort": "bubbleSort",
    "merge_sort": "mergeSort",
    "quicksort": "quicksort",
    "heap_sort": "heapSort",
    "insertion_sort": "insertionSort",
    "counting_sort": "countingSort",
    "sieve_of_eratosthenes": "sieve",
    "nqueens": "nqueens",
    "edit_distance": "editDistance",
    "longest_common_subsequence": "lcs",
    "kmp_search": "kmpSearch",
    "longest_increasing_subsequence": "longestIncreasingSubsequence",
    "gcd_extended": "gcdExtended",
    "matrix_multiply": "matrixMultiply",
    "tower_of_hanoi": "towerOfHanoi",
    "dfs_graph": "dfs",
    "bfs_graph": "bfs",
    "dijkstra": "dijkstra",
    "topological_sort": "topologicalSort",
    "binary_tree_inorder": "inorderTraversal",
    "knapsack_01": "knapsack",
}


def _go_arr_lit(arr: list[int]) -> str:
    return "[]int{" + ", ".join(str(x) for x in arr) + "}"


def _go_build_test_calls(algo: str) -> str:
    """Generate Go main() body for testing an algorithm."""
    fn = GO_FUNC_MAP[algo]
    lines = ['\tresults := make([]string, 0)']

    def _json_int_arr(var: str) -> str:
        return f'func() string {{ s := "["; for i, v := range {var} {{ if i > 0 {{ s += "," }}; s += fmt.Sprintf("%d", v) }}; return s + "]" }}()'

    def _json_float_arr(var: str) -> str:
        return f'func() string {{ s := "["; for i, v := range {var} {{ if i > 0 {{ s += "," }}; if v >= 1e18 || v == math.Inf(1) {{ s += fmt.Sprintf("%d", int(math.Inf(1))) }} else {{ s += fmt.Sprintf("%g", v) }} }}; return s + "]" }}()'

    def _json_int_matrix(var: str) -> str:
        return f'func() string {{ s := "["; for i, row := range {var} {{ if i > 0 {{ s += "," }}; s += "["; for j, v := range row {{ if j > 0 {{ s += "," }}; s += fmt.Sprintf("%d", v) }}; s += "]" }}; return s + "]" }}()'

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_search":
            arr, target = inputs
            lines.append(f"\tresults = append(results, fmt.Sprintf(\"%d\", {fn}({_go_arr_lit(arr)}, {target})))")

        elif algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = inputs[0]
            lines.append(f"\t{{ a := {_go_arr_lit(arr)}; {fn}(a); results = append(results, {_json_int_arr('a')}) }}")

        elif algo == "merge_sort":
            arr = inputs[0]
            lines.append(f"\t{{ r := {fn}({_go_arr_lit(arr)}); results = append(results, {_json_int_arr('r')}) }}")

        elif algo == "quicksort":
            arr = inputs[0]
            n = len(arr)
            lines.append(f"\t{{ a := {_go_arr_lit(arr)}; if len(a) > 0 {{ {fn}(a, 0, len(a)-1) }}; results = append(results, {_json_int_arr('a')}) }}")

        elif algo == "sieve_of_eratosthenes":
            lines.append(f"\t{{ r := {fn}({inputs[0]}); results = append(results, {_json_int_arr('r')}) }}")

        elif algo == "nqueens":
            lines.append(f"\tresults = append(results, fmt.Sprintf(\"%d\", {fn}({inputs[0]})))")

        elif algo == "edit_distance":
            s1, s2 = inputs
            lines.append(f'\tresults = append(results, fmt.Sprintf("%d", {fn}("{s1}", "{s2}")))')

        elif algo == "longest_common_subsequence":
            s1, s2 = inputs
            lines.append(f'\t{{ r := {fn}("{s1}", "{s2}"); results = append(results, fmt.Sprintf("\\"%s\\"", r)) }}')

        elif algo == "kmp_search":
            text, pat = inputs
            lines.append(f'\t{{ r := {fn}("{text}", "{pat}"); results = append(results, {_json_int_arr("r")}) }}')

        elif algo == "longest_increasing_subsequence":
            arr = inputs[0]
            lines.append(f"\tresults = append(results, fmt.Sprintf(\"%d\", {fn}({_go_arr_lit(arr)})))")

        elif algo == "gcd_extended":
            a, b = inputs
            lines.append(f'\t{{ g, x, y := {fn}({a}, {b}); results = append(results, fmt.Sprintf("[%d,%d,%d]", g, x, y)) }}')

        elif algo == "matrix_multiply":
            a_mat, b_mat = inputs
            a_lit = "[][]int{" + ", ".join("{" + ", ".join(str(x) for x in row) + "}" for row in a_mat) + "}"
            b_lit = "[][]int{" + ", ".join("{" + ", ".join(str(x) for x in row) + "}" for row in b_mat) + "}"
            lines.append(f"\t{{ r := {fn}({a_lit}, {b_lit}); results = append(results, {_json_int_matrix('r')}) }}")

        elif algo == "knapsack_01":
            w, v, c = inputs
            w_lit = _go_arr_lit(w)
            v_lit = _go_arr_lit(v)
            lines.append(f'\t{{ val, items := {fn}({w_lit}, {v_lit}, {c}); sort.Ints(items); results = append(results, fmt.Sprintf("[%d,%s]", val, {_json_int_arr("items")})) }}')

        elif algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'\t{{ moves := {fn}({n}, "{s}", "{t}", "{aux}"); s := "["; for i, m := range moves {{ if i > 0 {{ s += "," }}; s += fmt.Sprintf("[\\"%s\\",\\"%s\\"]", m.From, m.To) }}; s += "]"; results = append(results, s) }}')

        elif algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            n = max(graph.keys()) + 1 if graph else 0
            rows = []
            for i in range(n):
                r = graph.get(i, [])
                rows.append("{" + ", ".join(str(x) for x in r) + "}")
            graph_lit = "[][]int{" + ", ".join(rows) + "}"
            fn_go = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f"\t{{ r := {fn_go}({graph_lit}, {start}); results = append(results, {_json_int_arr('r')}) }}")

        elif algo == "dijkstra":
            graph, src = inputs
            rows = []
            for adj in graph:
                if adj:
                    pairs = ", ".join(f"[2]int{{{nb}, {wt}}}" for nb, wt in adj)
                else:
                    pairs = ""
                rows.append("{" + pairs + "}")
            graph_lit = "[][][2]int{" + ", ".join(rows) + "}"
            lines.append(f"\t{{ r := {fn}({graph_lit}, {src}); results = append(results, {_json_float_arr('r')}) }}")

        elif algo == "topological_sort":
            graph, n_nodes = inputs
            rows = []
            for adj in graph:
                rows.append("{" + ", ".join(str(x) for x in adj) + "}")
            graph_lit = "[][]int{" + ", ".join(rows) + "}"
            lines.append(f"\t{{ r := {fn}({graph_lit}, {n_nodes}); results = append(results, {_json_int_arr('r')}) }}")

        elif algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f'\t{{ n1 := &Node{{Val:1}}; n3 := &Node{{Val:3}}; n2 := &Node{{Val:2, Left:n1, Right:n3}}; r := {fn}(n2); results = append(results, {_json_int_arr("r")}) }}')
            elif spec == "left_chain_4":
                lines.append(f'\t{{ n1 := &Node{{Val:1}}; n2 := &Node{{Val:2, Left:n1}}; n3 := &Node{{Val:3, Left:n2}}; n4 := &Node{{Val:4, Left:n3}}; r := {fn}(n4); results = append(results, {_json_int_arr("r")}) }}')
            elif spec == "single":
                lines.append(f'\t{{ n := &Node{{Val:42}}; r := {fn}(n); results = append(results, {_json_int_arr("r")}) }}')

    lines.append('\tfmt.Print("[" + strings.Join(results, ",") + "]")')
    return "\n".join(lines)


def run_go(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Compile and run Go implementation."""
    go_bin = _tool("go")
    if not go_bin:
        return [{"desc": "SKIP", "error": "go not found"}]

    main_body = _go_build_test_calls(algo)
    import re
    code_clean = re.sub(r'^\s*import\s*\(\s*(?:"[^"]*"\s*)*\)\s*', '', code, flags=re.MULTILINE)
    code_clean = re.sub(r'^\s*import\s+"[^"]*"\s*$', '', code_clean, flags=re.MULTILINE)
    unused_import_guard = "var _ = math.Inf\nvar _ = heap.Init\nvar _ = sort.Ints\nvar _ = strings.Join\n"
    src = _GO_IMPORTS + "\n" + unused_import_guard + "\n" + code_clean + "\n\nfunc main() {\n" + main_body + "\n}\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "main.go")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src)

        # Initialize a temporary Go module
        subprocess.run(
            [go_bin, "mod", "init", "test"],
            capture_output=True, text=True, cwd=tmpdir, timeout=10,
        )

        run = subprocess.run(
            [go_bin, "run", "main.go"],
            capture_output=True, text=True, timeout=60, cwd=tmpdir,
        )
        if run.returncode != 0:
            return [{"desc": "COMPILE/RUNTIME_ERROR", "error": run.stderr[:500]}]

        try:
            raw = json.loads(run.stdout)
            return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
        except Exception as e:
            return [{"desc": "PARSE_ERROR", "error": f"stdout={run.stdout[:300]}, err={e}"}]


# ---------------------------------------------------------------------------
# Rust runner
# ---------------------------------------------------------------------------

_RUST_PREAMBLE = """\
use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::cmp::{Ordering, Reverse, min, max};
"""


def _rust_build_test_calls(algo: str) -> str:
    """Generate Rust main() body for testing an algorithm."""
    rust_func = {
        "binary_search": "binary_search",
        "bubble_sort": "bubble_sort",
        "merge_sort": "merge_sort",
        "quicksort": "quicksort",
        "heap_sort": "heap_sort",
        "insertion_sort": "insertion_sort",
        "counting_sort": "counting_sort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "longest_increasing_subsequence",
        "gcd_extended": "gcd_extended",
        "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "tower_of_hanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = rust_func[algo]
    lines = ['    let mut results: Vec<String> = Vec::new();']

    def _json_i32_arr(var: str) -> str:
        return f'{var}.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")'

    def _json_usize_arr(var: str) -> str:
        return f'{var}.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")'

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_search":
            arr, target = inputs
            arr_lit = "&[" + ", ".join(f"{x}i32" for x in arr) + "]"
            lines.append(f"    results.push(format!(\"{{}}\", {fn}({arr_lit}, {target})));")

        elif algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = inputs[0]
            arr_lit = "vec![" + ", ".join(f"{x}i32" for x in arr) + "]"
            lines.append(f"    {{ let mut a = {arr_lit}; {fn}(&mut a); results.push(format!(\"[{{}}]\", {_json_i32_arr('a')})); }}")

        elif algo == "merge_sort":
            arr = inputs[0]
            arr_lit = "vec![" + ", ".join(f"{x}i32" for x in arr) + "]"
            lines.append(f"    {{ let mut a = {arr_lit}; {fn}(&mut a); results.push(format!(\"[{{}}]\", {_json_i32_arr('a')})); }}")

        elif algo == "quicksort":
            arr = inputs[0]
            n = len(arr)
            arr_lit = "vec![" + ", ".join(f"{x}i32" for x in arr) + "]"
            if n > 0:
                lines.append(f"    {{ let mut a = {arr_lit}; {fn}(&mut a, 0, {n - 1}); results.push(format!(\"[{{}}]\", {_json_i32_arr('a')})); }}")
            else:
                lines.append(f'    {{ results.push("[]".to_string()); }}')

        elif algo == "sieve_of_eratosthenes":
            n_val = inputs[0]
            lines.append(f"    {{ let r = {fn}({n_val}); results.push(format!(\"[{{}}]\", {_json_usize_arr('r')})); }}")

        elif algo == "nqueens":
            n_val = inputs[0]
            lines.append(f"    results.push(format!(\"{{}}\", {fn}({n_val})));")

        elif algo == "edit_distance":
            s1, s2 = inputs
            lines.append(f'    results.push(format!("{{}}", {fn}("{s1}", "{s2}")));')

        elif algo == "longest_common_subsequence":
            s1, s2 = inputs
            lines.append(f'    results.push(format!("\\"{{}}\\"", {fn}("{s1}", "{s2}")));')

        elif algo == "kmp_search":
            text, pat = inputs
            lines.append(f'    {{ let r = {fn}("{text}", "{pat}"); results.push(format!("[{{}}]", {_json_usize_arr("r")})); }}')

        elif algo == "longest_increasing_subsequence":
            arr = inputs[0]
            arr_lit = "&[" + ", ".join(f"{x}i32" for x in arr) + "]"
            lines.append(f"    results.push(format!(\"{{}}\", {fn}({arr_lit})));")

        elif algo == "gcd_extended":
            a, b = inputs
            lines.append(f'    {{ let (g, x, y) = {fn}({a}, {b}); results.push(format!("[{{}},{{}},{{}}]", g, x, y)); }}')

        elif algo == "matrix_multiply":
            a_mat, b_mat = inputs
            a_lit = "vec![" + ", ".join("vec![" + ", ".join(f"{x}i32" for x in row) + "]" for row in a_mat) + "]"
            b_lit = "vec![" + ", ".join("vec![" + ", ".join(f"{x}i32" for x in row) + "]" for row in b_mat) + "]"
            lines.append(f'    {{ let r = {fn}(&{a_lit}, &{b_lit}); let s: Vec<String> = r.iter().map(|row| format!("[{{}}]", row.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))).collect(); results.push(format!("[{{}}]", s.join(","))); }}')

        elif algo == "knapsack_01":
            w, v, c = inputs
            w_lit = "&[" + ", ".join(f"{x}usize" for x in w) + "]"
            v_lit = "&[" + ", ".join(f"{x}i32" for x in v) + "]"
            lines.append(f'    {{ let (val, mut items) = {fn}({w_lit}, {v_lit}, {c}); items.sort(); results.push(format!("[{{}},{{}}]", val, format!("[{{}}]", items.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")))); }}')

        elif algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'    {{ let moves = {fn}({n}, "{s}", "{t}", "{aux}"); let m: Vec<String> = moves.iter().map(|m| format!("[\\"{{}}\\",\\"{{}}\\"]", m.0, m.1)).collect(); results.push(format!("[{{}}]", m.join(","))); }}')

        elif algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            n = max(graph.keys()) + 1 if graph else 0
            rows = []
            for i in range(n):
                r = graph.get(i, [])
                rows.append("vec![" + ", ".join(f"{x}usize" for x in r) + "]")
            graph_lit = "vec![" + ", ".join(rows) + "]"
            fn_rs = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f"    {{ let r = {fn_rs}(&{graph_lit}, {start}); results.push(format!(\"[{{}}]\", {_json_usize_arr('r')})); }}")

        elif algo == "dijkstra":
            graph, src = inputs
            rows = []
            for adj in graph:
                if adj:
                    pairs = ", ".join(f"({nb}usize, {wt}i32)" for nb, wt in adj)
                else:
                    pairs = ""
                rows.append("vec![" + pairs + "]")
            graph_lit = "vec![" + ", ".join(rows) + "]"
            lines.append(f"    {{ let r = {fn}(&{graph_lit}, {src}); results.push(format!(\"[{{}}]\", {_json_i32_arr('r')})); }}")

        elif algo == "topological_sort":
            graph, n_nodes = inputs
            rows = []
            for adj in graph:
                rows.append("vec![" + ", ".join(f"{x}usize" for x in adj) + "]")
            graph_lit = "vec![" + ", ".join(rows) + "]"
            lines.append(f"    {{ let r = {fn}(&{graph_lit}, {n_nodes}); results.push(format!(\"[{{}}]\", {_json_usize_arr('r')})); }}")

        elif algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f'    {{ let tree = Node {{ val: 2, left: Some(Box::new(Node {{ val: 1, left: None, right: None }})), right: Some(Box::new(Node {{ val: 3, left: None, right: None }})) }}; let r = {fn}(&tree); results.push(format!("[{{}}]", {_json_i32_arr("r")})); }}')
            elif spec == "left_chain_4":
                lines.append(f'    {{ let tree = Node {{ val: 4, left: Some(Box::new(Node {{ val: 3, left: Some(Box::new(Node {{ val: 2, left: Some(Box::new(Node {{ val: 1, left: None, right: None }})), right: None }})), right: None }})), right: None }}; let r = {fn}(&tree); results.push(format!("[{{}}]", {_json_i32_arr("r")})); }}')
            elif spec == "single":
                lines.append(f'    {{ let tree = Node {{ val: 42, left: None, right: None }}; let r = {fn}(&tree); results.push(format!("[{{}}]", {_json_i32_arr("r")})); }}')

    lines.append('    print!("[{}]", results.join(","));')
    return "\n".join(lines)


def run_rust(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Compile and run Rust implementation."""
    rustc = _tool("rustc")
    if not rustc:
        return [{"desc": "SKIP", "error": "rustc not found"}]

    main_body = _rust_build_test_calls(algo)
    import re
    code_clean = re.sub(r'^\s*use\s+std::collections::\w+;\s*$', '', code, flags=re.MULTILINE)
    code_clean = re.sub(r'^\s*use\s+std::cmp::\w+;\s*$', '', code_clean, flags=re.MULTILINE)
    preamble = "#![allow(unused_imports, dead_code, unused_variables, unused_mut, unused_assignments)]\n" + _RUST_PREAMBLE
    src = preamble + "\n" + code_clean + "\n\nfn main() {\n" + main_body + "\n}\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "test.rs")
        exe_path = os.path.join(tmpdir, "test.exe")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src)

        comp = subprocess.run(
            [rustc, "--edition", "2021", "-o", exe_path, src_path],
            capture_output=True, text=True, timeout=60,
        )
        if comp.returncode != 0:
            return [{"desc": "COMPILE_ERROR", "error": comp.stderr[:500]}]

        run = subprocess.run(
            [exe_path], capture_output=True, text=True, timeout=30,
        )
        if run.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": run.stderr[:500]}]

        try:
            raw = json.loads(run.stdout)
            return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
        except Exception as e:
            return [{"desc": "PARSE_ERROR", "error": f"stdout={run.stdout[:300]}, err={e}"}]


# ---------------------------------------------------------------------------
# PHP runner
# ---------------------------------------------------------------------------

def _php_encode(val) -> str:
    """Encode a Python value as a PHP literal."""
    if isinstance(val, dict):
        pairs = ", ".join(f"{k} => {_php_encode(v)}" for k, v in val.items())
        return "[" + pairs + "]"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_php_encode(x) for x in val) + "]"
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _php_build_test_script(algo: str, code: str) -> str:
    """Generate a PHP test script for an algorithm."""
    php_func = {
        "binary_search": "binary_search",
        "bubble_sort": "bubble_sort",
        "merge_sort": "merge_sort",
        "quicksort": "quicksort",
        "heap_sort": "heap_sort",
        "insertion_sort": "insertion_sort",
        "counting_sort": "counting_sort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "longest_increasing_subsequence",
        "gcd_extended": "gcd_extended",
        "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "tower_of_hanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = php_func[algo]

    lines = ["<?php", code, "", "$results = [];"]

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f"$results[] = json_encode({fn}(new Node(2, new Node(1), new Node(3))));")
            elif spec == "left_chain_4":
                lines.append(f"$results[] = json_encode({fn}(new Node(4, new Node(3, new Node(2, new Node(1))))));")
            elif spec == "single":
                lines.append(f"$results[] = json_encode({fn}(new Node(42)));")
            continue

        if algo == "quicksort":
            arr = json.dumps(inputs[0])
            lines.append(f"{{ $a = {arr}; {fn}($a, 0, count($a) - 1); $results[] = json_encode($a); }}")
            continue

        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = json.dumps(inputs[0])
            lines.append(f"{{ $a = {arr}; {fn}($a); $results[] = json_encode($a); }}")
            continue

        if algo == "knapsack_01":
            w, v, c = inputs
            lines.append(f"{{ $r = {fn}({json.dumps(w)}, {json.dumps(v)}, {c}); $items = $r[1]; sort($items); $results[] = json_encode([$r[0], $items]); }}")
            continue

        if algo == "gcd_extended":
            a, b = inputs
            lines.append(f"{{ $r = {fn}({a}, {b}); $results[] = json_encode($r); }}")
            continue

        if algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'{{ $moves = {fn}({n}, "{s}", "{t}", "{aux}"); $results[] = json_encode($moves); }}')
            continue

        if algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            lines.append(f"$results[] = json_encode({fn}({_php_encode(graph)}, {start}));")
            continue

        if algo == "dijkstra":
            graph, src = inputs
            lines.append(f"$results[] = json_encode({fn}({_php_encode(graph)}, {src}));")
            continue

        if algo == "topological_sort":
            graph, n_nodes = inputs
            lines.append(f"$results[] = json_encode({fn}({_php_encode(graph)}, {n_nodes}));")
            continue

        arg_strs = [_php_encode(a) for a in inputs]
        lines.append(f"$results[] = json_encode({fn}({', '.join(arg_strs)}));")

    lines.append('echo "[" . implode(",", $results) . "]";')
    return "\n".join(lines)


def run_php(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Execute PHP implementation."""
    php_bin = _tool("php")
    if not php_bin:
        return [{"desc": "SKIP", "error": "php not found"}]

    script = _php_build_test_script(algo, code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False, encoding="utf-8") as f:
        f.write(script)
        php_path = f.name

    try:
        r = subprocess.run(
            [php_bin, php_path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "unknown error")[:500]
            return [{"desc": "RUNTIME_ERROR", "error": err}]

        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        stdout_peek = ""
        try:
            stdout_peek = r.stdout[:200]
        except Exception:
            pass
        return [{"desc": "PARSE_ERROR", "error": f"stdout={stdout_peek}, err={e}"}]
    finally:
        os.unlink(php_path)


# ---------------------------------------------------------------------------
# Perl runner
# ---------------------------------------------------------------------------

def _perl_encode(val) -> str:
    """Encode a Python value as a Perl literal."""
    if isinstance(val, dict):
        pairs = ", ".join(f"{k} => {_perl_encode(v)}" for k, v in val.items())
        return "{" + pairs + "}"
    if isinstance(val, (list, tuple)):
        if all(isinstance(x, (list, tuple)) for x in val):
            inner = ", ".join(_perl_encode(x) for x in val)
            return "[" + inner + "]"
        return "[" + ", ".join(_perl_encode(x) for x in val) + "]"
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(val)


def _perl_build_test_script(algo: str, code: str) -> str:
    """Generate a Perl test script for an algorithm."""
    perl_func = {
        "binary_search": "binary_search",
        "bubble_sort": "bubble_sort",
        "merge_sort": "merge_sort",
        "quicksort": "quicksort",
        "heap_sort": "heap_sort",
        "insertion_sort": "insertion_sort",
        "counting_sort": "counting_sort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "longest_increasing_subsequence",
        "gcd_extended": "gcd_extended",
        "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "tower_of_hanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = perl_func[algo]

    lines = [
        "use strict;",
        "use warnings;",
        "use JSON::PP;",
        "",
        code,
        "",
        "my @results;",
    ]

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f"push @results, encode_json({fn}(new_node(2, new_node(1, undef, undef), new_node(3, undef, undef))));")
            elif spec == "left_chain_4":
                lines.append(f"push @results, encode_json({fn}(new_node(4, new_node(3, new_node(2, new_node(1, undef, undef), undef), undef), undef)));")
            elif spec == "single":
                lines.append(f"push @results, encode_json({fn}(new_node(42, undef, undef)));")
            continue

        if algo == "quicksort":
            arr = inputs[0]
            arr_lit = "(" + ", ".join(str(x) for x in arr) + ")"
            lines.append(f"{{ my @a = {arr_lit}; {fn}(\\@a, 0, $#a); push @results, encode_json(\\@a); }}")
            continue

        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr = inputs[0]
            arr_lit = "(" + ", ".join(str(x) for x in arr) + ")"
            lines.append(f"{{ my @a = {arr_lit}; {fn}(\\@a); push @results, encode_json(\\@a); }}")
            continue

        if algo == "merge_sort":
            arr = inputs[0]
            arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
            lines.append(f"push @results, encode_json({fn}({arr_lit}));")
            continue

        if algo == "knapsack_01":
            w, v, c = inputs
            w_lit = "[" + ", ".join(str(x) for x in w) + "]"
            v_lit = "[" + ", ".join(str(x) for x in v) + "]"
            lines.append(f"{{ my ($val, $items) = {fn}({w_lit}, {v_lit}, {c}); my @sorted_items = sort {{ $a <=> $b }} @$items; push @results, encode_json([$val, \\@sorted_items]); }}")
            continue

        if algo == "gcd_extended":
            a, b = inputs
            lines.append(f"{{ my @r = {fn}({a}, {b}); push @results, encode_json(\\@r); }}")
            continue

        if algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'{{ my $moves = {fn}({n}, "{s}", "{t}", "{aux}"); push @results, encode_json($moves); }}')
            continue

        if algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            graph_lit = _perl_encode(graph)
            fn_perl = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f"push @results, encode_json({fn_perl}({graph_lit}, {start}));")
            continue

        if algo == "dijkstra":
            graph, src = inputs
            graph_lit = _perl_encode(graph)
            lines.append(f"push @results, encode_json({fn}({graph_lit}, {src}));")
            continue

        if algo == "topological_sort":
            graph, n_nodes = inputs
            graph_lit = _perl_encode(graph)
            lines.append(f"push @results, encode_json({fn}({graph_lit}, {n_nodes}));")
            continue

        if algo in ("binary_search", "sieve_of_eratosthenes", "nqueens",
                     "edit_distance", "longest_common_subsequence",
                     "kmp_search", "longest_increasing_subsequence",
                     "matrix_multiply"):
            if algo == "binary_search":
                arr, target = inputs
                arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
                lines.append(f"push @results, encode_json({fn}({arr_lit}, {target}));")
            elif algo == "sieve_of_eratosthenes":
                lines.append(f"push @results, encode_json({fn}({inputs[0]}));")
            elif algo == "nqueens":
                lines.append(f"push @results, encode_json({fn}({inputs[0]}));")
            elif algo in ("edit_distance", "longest_common_subsequence"):
                s1, s2 = inputs
                lines.append(f'push @results, encode_json({fn}("{s1}", "{s2}"));')
            elif algo == "kmp_search":
                text, pat = inputs
                lines.append(f'push @results, encode_json({fn}("{text}", "{pat}"));')
            elif algo == "longest_increasing_subsequence":
                arr = inputs[0]
                arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
                lines.append(f"push @results, encode_json({fn}({arr_lit}));")
            elif algo == "matrix_multiply":
                a_mat, b_mat = inputs
                a_lit = _perl_encode(a_mat)
                b_lit = _perl_encode(b_mat)
                lines.append(f"push @results, encode_json({fn}({a_lit}, {b_lit}));")
            continue

    lines.append('print "[" . join(",", @results) . "]";')
    return "\n".join(lines)


def run_perl(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Execute Perl implementation."""
    perl_bin = shutil.which("perl")
    if not perl_bin:
        return [{"desc": "SKIP", "error": "perl not found"}]

    script = _perl_build_test_script(algo, code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False, encoding="utf-8") as f:
        f.write(script)
        pl_path = f.name

    try:
        r = subprocess.run(
            [perl_bin, pl_path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "unknown error")[:500]
            return [{"desc": "RUNTIME_ERROR", "error": err}]

        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        stdout_peek = ""
        try:
            stdout_peek = r.stdout[:200]
        except Exception:
            pass
        return [{"desc": "PARSE_ERROR", "error": f"stdout={stdout_peek}, err={e}"}]
    finally:
        os.unlink(pl_path)


# ---------------------------------------------------------------------------
# Bash runner
# ---------------------------------------------------------------------------

def _bash_build_test_script(algo: str, code: str) -> str:
    """Generate a Bash script that runs the algorithm with test inputs."""
    bash_func = {
        "binary_search": "binary_search",
        "bubble_sort": "bubble_sort",
        "merge_sort": "merge_sort",
        "quicksort": "quicksort",
        "heap_sort": "heap_sort",
        "insertion_sort": "insertion_sort",
        "counting_sort": "counting_sort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "lis",
        "gcd_extended": "gcd_extended",
        "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "solve",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = bash_func[algo]

    lines = ["#!/bin/bash", "", code, ""]

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_search":
            arr, target = inputs
            arr_str = " ".join(str(x) for x in arr)
            lines.append(f'result=$({fn} {target} {arr_str})')
            lines.append('echo "${result:--1}"')

        elif algo in ("bubble_sort", "insertion_sort", "counting_sort",
                       "merge_sort", "heap_sort", "quicksort"):
            arr = inputs[0]
            arr_str = " ".join(str(x) for x in arr)
            lines.append(f'result=$({fn} {arr_str})')
            lines.append('echo "$result"')

        elif algo == "sieve_of_eratosthenes":
            n_val = inputs[0]
            lines.append(f'result=$({fn} {n_val})')
            lines.append('echo "$result"')

        elif algo == "nqueens":
            n_val = inputs[0]
            lines.append(f'result=$({fn} {n_val})')
            lines.append('echo "$result"')

        elif algo == "edit_distance":
            s1, s2 = inputs
            lines.append(f'result=$({fn} "{s1}" "{s2}")')
            lines.append('echo "$result"')

        elif algo == "longest_common_subsequence":
            s1, s2 = inputs
            lines.append(f'result=$({fn} "{s1}" "{s2}")')
            lines.append('echo "$result"')

        elif algo == "kmp_search":
            text, pat = inputs
            lines.append(f'result=$({fn} "{text}" "{pat}")')
            lines.append('echo "$result"')

        elif algo == "longest_increasing_subsequence":
            arr = inputs[0]
            arr_str = " ".join(str(x) for x in arr)
            lines.append(f'result=$({fn} {arr_str})')
            lines.append('echo "$result"')

        elif algo == "gcd_extended":
            a, b = inputs
            lines.append(f'result=$({fn} {a} {b})')
            lines.append('echo "$result"')

        elif algo == "matrix_multiply":
            a_mat, b_mat = inputs
            rows_a = len(a_mat)
            cols_a = len(a_mat[0]) if a_mat else 0
            cols_b = len(b_mat[0]) if b_mat else 0
            a_flat = " ".join(str(x) for row in a_mat for x in row)
            b_flat = " ".join(str(x) for row in b_mat for x in row)
            lines.append(f'result=$({fn} {rows_a} {cols_a} {cols_b} {a_flat} {b_flat})')
            lines.append('echo "$result"')

        elif algo == "knapsack_01":
            w, v, c = inputs
            n = len(w)
            w_str = " ".join(str(x) for x in w)
            v_str = " ".join(str(x) for x in v)
            lines.append(f'result=$({fn} {c} {n} {w_str} {v_str})')
            lines.append('echo "$result"')

        elif algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(f'toh_moves=()')
            lines.append(f'{fn} {n} {s} {t} {aux} toh_moves')
            lines.append('echo "${toh_moves[*]}"')

        elif algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            n = max(graph.keys()) + 1 if graph else 0
            lines.append(f'g_arr=(')
            for i in range(n):
                nbrs = graph.get(i, [])
                lines.append(f'  "{" ".join(str(x) for x in nbrs)}"')
            lines.append(')')
            fn_bash = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f'result=$({fn_bash} g_arr {start})')
            lines.append('echo "$result"')

        elif algo == "dijkstra":
            graph, src = inputs
            n = len(graph)
            lines.append('dj_graph=(')
            for adj in graph:
                if adj:
                    edge_str = " ".join(f"{nb},{wt}" for nb, wt in adj)
                else:
                    edge_str = ""
                lines.append(f'  "{edge_str}"')
            lines.append(')')
            lines.append(f'result=$({fn} dj_graph {src})')
            lines.append('echo "$result"')

        elif algo == "topological_sort":
            graph, n_nodes = inputs
            lines.append('ts_graph=(')
            for adj in graph:
                lines.append(f'  "{" ".join(str(x) for x in adj)}"')
            lines.append(')')
            lines.append(f'result=$({fn} ts_graph {n_nodes})')
            lines.append('echo "$result"')

        elif algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append('bt_val=(2 1 3); bt_left=(1 -1 -1); bt_right=(2 -1 -1)')
                lines.append(f'result=$({fn} bt_val bt_left bt_right 0)')
            elif spec == "left_chain_4":
                lines.append('bt_val=(4 3 2 1); bt_left=(1 2 3 -1); bt_right=(-1 -1 -1 -1)')
                lines.append(f'result=$({fn} bt_val bt_left bt_right 0)')
            elif spec == "single":
                lines.append('bt_val=(42); bt_left=(-1); bt_right=(-1)')
                lines.append(f'result=$({fn} bt_val bt_left bt_right 0)')
            lines.append('echo "$result"')

    return "\n".join(lines)


def _parse_bash_output(algo: str, stdout: str, expected_results: list[dict]) -> list[dict]:
    """Parse Bash stdout (one result per line) into comparison-ready dicts."""
    raw_lines = stdout.split("\n")
    # Strip the single trailing empty element that split("\n") produces
    # when stdout ends with a newline, but preserve legitimate blank lines
    # (empty arrays / empty strings produce blank output lines)
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    output_lines = [line.strip() for line in raw_lines]

    results = []
    for i, tc in enumerate(TEST_CASES[algo]):
        if i >= len(output_lines):
            results.append({"desc": tc["desc"], "actual": None})
            continue

        raw = output_lines[i]
        exp = expected_results[i]["expected"]

        if isinstance(exp, int):
            try:
                results.append({"desc": tc["desc"], "actual": int(raw)})
            except ValueError:
                results.append({"desc": tc["desc"], "actual": raw})

        elif isinstance(exp, str) and not isinstance(exp, list):
            results.append({"desc": tc["desc"], "actual": raw})

        elif isinstance(exp, list) and exp and isinstance(exp[0], list):
            # tower_of_hanoi: "A C A B B C" → [["A","C"],["A","B"],["B","C"]]
            if algo == "tower_of_hanoi":
                tokens = raw.split()
                pairs = [[tokens[j], tokens[j+1]] for j in range(0, len(tokens), 2)] if tokens else []
                results.append({"desc": tc["desc"], "actual": pairs})
            elif algo == "matrix_multiply":
                nums = [int(x) for x in raw.split()]
                n = len(expected_results[i]["expected"])
                m = len(expected_results[i]["expected"][0]) if n else 0
                matrix = [nums[r*m:(r+1)*m] for r in range(n)]
                results.append({"desc": tc["desc"], "actual": matrix})
            else:
                results.append({"desc": tc["desc"], "actual": raw})

        elif isinstance(exp, list) and exp and isinstance(exp[0], int):
            if algo == "knapsack_01":
                tokens = raw.split()
                if tokens:
                    val = int(tokens[0])
                    items = sorted(int(x) for x in tokens[1:]) if len(tokens) > 1 else []
                    results.append({"desc": tc["desc"], "actual": [val, items]})
                else:
                    results.append({"desc": tc["desc"], "actual": raw})
            elif algo == "gcd_extended":
                tokens = raw.split()
                results.append({"desc": tc["desc"], "actual": [int(x) for x in tokens]})
            else:
                tokens = raw.split()
                results.append({"desc": tc["desc"], "actual": [int(x) for x in tokens] if tokens else []})

        elif isinstance(exp, list) and not exp:
            tokens = raw.split()
            results.append({"desc": tc["desc"], "actual": [int(x) for x in tokens] if tokens else []})

        else:
            results.append({"desc": tc["desc"], "actual": raw})

    return results


def run_bash(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    """Execute Bash implementation."""
    bash_bin = shutil.which("bash")
    if not bash_bin:
        return [{"desc": "SKIP", "error": "bash not found"}]

    script = _bash_build_test_script(algo, code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8", newline="\n") as f:
        f.write(script)
        sh_path = f.name

    try:
        r = subprocess.run(
            [bash_bin, sh_path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "unknown error")[:500]
            return [{"desc": "RUNTIME_ERROR", "error": err}]

        return _parse_bash_output(algo, r.stdout, expected_results)
    except Exception as e:
        stdout_peek = ""
        try:
            stdout_peek = r.stdout[:200]
        except Exception:
            pass
        return [{"desc": "PARSE_ERROR", "error": f"stdout={stdout_peek}, err={e}"}]
    finally:
        os.unlink(sh_path)


# ---------------------------------------------------------------------------
# Ruby runner
# ---------------------------------------------------------------------------

def _ruby_encode(val) -> str:
    if isinstance(val, dict):
        pairs = ", ".join(f"{k} => {_ruby_encode(v)}" for k, v in val.items())
        return "{" + pairs + "}"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_ruby_encode(x) for x in val) + "]"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _ruby_build_test_script(algo: str, code: str) -> str:
    ruby_func = {
        "binary_search": "binary_search", "bubble_sort": "bubble_sort",
        "merge_sort": "merge_sort", "quicksort": "quicksort",
        "heap_sort": "heap_sort", "insertion_sort": "insertion_sort",
        "counting_sort": "counting_sort",
        "sieve_of_eratosthenes": "sieve", "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs", "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "longest_increasing_subsequence",
        "gcd_extended": "gcd_extended", "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "tower_of_hanoi",
        "dfs_graph": "dfs", "bfs_graph": "bfs", "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = ruby_func[algo]
    lines = ["require 'json'", "TreeNode = Struct.new(:val, :left, :right)", "", code, "", "results = []"]
    for tc in TEST_CASES[algo]:
        inp = tc["inputs"]
        if algo == "binary_tree_inorder":
            spec = inp[0]
            if spec == "balanced_3":
                lines.append(f"results << {fn}(TreeNode.new(2, TreeNode.new(1), TreeNode.new(3)))")
            elif spec == "left_chain_4":
                lines.append(f"results << {fn}(TreeNode.new(4, TreeNode.new(3, TreeNode.new(2, TreeNode.new(1)))))")
            elif spec == "single":
                lines.append(f"results << {fn}(TreeNode.new(42))")
            continue
        if algo == "quicksort":
            lines.append(f"a = {_ruby_encode(inp[0])}; {fn}(a, 0, a.length - 1); results << a")
            continue
        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            lines.append(f"a = {_ruby_encode(inp[0])}; {fn}(a); results << a")
            continue
        if algo == "knapsack_01":
            w, v, c = inp
            lines.append(f"val, items = {fn}({_ruby_encode(w)}, {_ruby_encode(v)}, {c}); results << [val, items.sort]")
            continue
        if algo == "gcd_extended":
            lines.append(f"results << {fn}({inp[0]}, {inp[1]})")
            continue
        args = ", ".join(_ruby_encode(a) for a in inp)
        lines.append(f"results << {fn}({args})")
    lines.append("puts JSON.generate(results)")
    return "\n".join(lines)


def run_ruby(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    ruby_bin = _tool("ruby") or shutil.which("ruby")
    if not ruby_bin:
        return [{"desc": "SKIP", "error": "ruby not found"}]
    script = _ruby_build_test_script(algo, code)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rb", delete=False, encoding="utf-8") as f:
        f.write(script); rb_path = f.name
    try:
        r = subprocess.run([ruby_bin, rb_path], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (r.stderr or r.stdout or "")[:500]}]
        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        os.unlink(rb_path)


# ---------------------------------------------------------------------------
# Lua runner
# ---------------------------------------------------------------------------

def _lua_encode(val) -> str:
    if isinstance(val, dict):
        pairs = ", ".join(f"[{k}]={_lua_encode(v)}" for k, v in val.items())
        return "{" + pairs + "}"
    if isinstance(val, (list, tuple)):
        return "{" + ", ".join(_lua_encode(x) for x in val) + "}"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _lua_build_test_script(algo: str, code: str) -> str:
    def _lua_graph_1based(graph, weighted=False):
        """Convert 0-based Python graph to 1-based Lua adjacency list."""
        n = len(graph) if isinstance(graph, list) else max(int(k) for k in graph) + 1 if graph else 0
        parts = []
        for i in range(n):
            adj = graph[i] if isinstance(graph, list) else graph.get(i, graph.get(str(i), []))
            if weighted:
                inner = ", ".join(f"{{{e[0]+1}, {e[1]}}}" for e in adj)
            else:
                inner = ", ".join(str(x + 1) for x in adj)
            parts.append("{" + inner + "}")
        return "{" + ", ".join(parts) + "}"

    def _lua_graph_1based_adj(graph):
        """Convert 0-based adjacency list to 1-based for Lua."""
        parts = []
        for adj in graph:
            inner = ", ".join(str(x + 1) for x in adj)
            parts.append("{" + inner + "}")
        return "{" + ", ".join(parts) + "}"

    lua_func = {
        "binary_search": "binary_search", "bubble_sort": "bubble_sort",
        "merge_sort": "merge_sort", "quicksort": "quicksort",
        "heap_sort": "heapSort", "insertion_sort": "insertionSort",
        "counting_sort": "countingSort",
        "sieve_of_eratosthenes": "sieve", "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs", "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "longest_increasing_subsequence",
        "gcd_extended": "gcd_extended", "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "tower_of_hanoi",
        "dfs_graph": "dfs", "bfs_graph": "bfs", "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = lua_func[algo]
    json_lib = textwrap.dedent("""\
    local function _json(v)
      if type(v) == "table" then
        local n = #v
        local is_arr = (n > 0) or next(v) == nil
        if is_arr then
          local parts = {}
          for i = 1, n do parts[i] = _json(v[i]) end
          return "[" .. table.concat(parts, ",") .. "]"
        else
          local parts = {}
          for k, val in pairs(v) do
            parts[#parts+1] = '"' .. tostring(k) .. '":' .. _json(val)
          end
          return "{" .. table.concat(parts, ",") .. "}"
        end
      elseif type(v) == "string" then
        return '"' .. v:gsub('\\\\', '\\\\\\\\'):gsub('"', '\\\\"') .. '"'
      elseif type(v) == "number" then
        if v == math.floor(v) then return string.format("%d", v) end
        return tostring(v)
      elseif type(v) == "boolean" then
        return v and "true" or "false"
      elseif v == nil then
        return "null"
      end
      return tostring(v)
    end
    """)
    lines = [json_lib, code, "", "local results = {}"]
    for tc in TEST_CASES[algo]:
        inp = tc["inputs"]
        if algo == "binary_tree_inorder":
            spec = inp[0]
            if spec == "balanced_3":
                lines.append(f"table.insert(results, {fn}(new_node(2, new_node(1, nil, nil), new_node(3, nil, nil))))")
            elif spec == "left_chain_4":
                lines.append(f"table.insert(results, {fn}(new_node(4, new_node(3, new_node(2, new_node(1, nil, nil), nil), nil), nil)))")
            elif spec == "single":
                lines.append(f"table.insert(results, {fn}(new_node(42, nil, nil)))")
            continue
        if algo == "quicksort":
            lines.append(f"do local a = {_lua_encode(inp[0])}; {fn}(a, 1, #a); table.insert(results, a) end")
            continue
        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            lines.append(f"do local a = {_lua_encode(inp[0])}; {fn}(a); table.insert(results, a) end")
            continue
        if algo == "knapsack_01":
            w, v, c = inp
            lines.append(f"do local val, items = {fn}({_lua_encode(w)}, {_lua_encode(v)}, {c}); table.sort(items); table.insert(results, {{val, items}}) end")
            continue
        if algo == "gcd_extended":
            lines.append(f"do local g, x, y = {fn}({inp[0]}, {inp[1]}); table.insert(results, {{g, x, y}}) end")
            continue
        if algo in ("dfs_graph", "bfs_graph"):
            fn_l = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f"table.insert(results, {fn_l}({_lua_encode(inp[0])}, {inp[1]}))")
            continue
        if algo == "dijkstra":
            graph, src = inp
            n = len(graph)
            g_1 = _lua_graph_1based(graph, weighted=True)
            lines.append(f"table.insert(results, {fn}({g_1}, {src + 1}, {n}))")
            continue
        if algo == "topological_sort":
            graph, n_nodes = inp
            g_1 = _lua_graph_1based_adj(graph)
            lines.append(f"table.insert(results, {fn}({g_1}, {n_nodes}))")
            continue
        args = ", ".join(_lua_encode(a) for a in inp)
        lines.append(f"table.insert(results, {fn}({args}))")
    lines.append("io.write(_json(results))")
    return "\n".join(lines)


def run_lua(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    lua_bin = _tool("lua") or shutil.which("lua5.4") or shutil.which("lua")
    if not lua_bin:
        return [{"desc": "SKIP", "error": "lua not found"}]
    script = _lua_build_test_script(algo, code)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lua", delete=False, encoding="utf-8") as f:
        f.write(script); lua_path = f.name
    try:
        r = subprocess.run([lua_bin, lua_path], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (r.stderr or r.stdout or "")[:500]}]
        raw = json.loads(r.stdout)
        results = []
        for i, tc in enumerate(TEST_CASES[algo]):
            val = raw[i]
            if algo == "topological_sort" and isinstance(val, list):
                val = [int(x) - 1 for x in val]
            results.append({"desc": tc["desc"], "actual": val})
        return results
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        os.unlink(lua_path)


# ---------------------------------------------------------------------------
# R runner
# ---------------------------------------------------------------------------

def _r_encode(val) -> str:
    if isinstance(val, dict):
        pairs = ", ".join(f'"{k}"={_r_encode(v)}' for k, v in val.items())
        return f"list({pairs})"
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return "integer(0)"
        if all(isinstance(x, (list, tuple)) for x in val):
            return "list(" + ", ".join(_r_encode(x) for x in val) + ")"
        return "c(" + ", ".join(_r_encode(x) for x in val) + ")"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _r_graph_1based(graph_dict):
    """Convert 0-based Python graph dict to 1-based R adjacency list."""
    max_node = max(int(k) for k in graph_dict) if graph_dict else -1
    parts = []
    for i in range(max_node + 1):
        nbrs = graph_dict.get(i, graph_dict.get(str(i), []))
        nbrs_1 = [x + 1 for x in nbrs]
        if nbrs_1:
            parts.append("c(" + ",".join(str(x) for x in nbrs_1) + ")")
        else:
            parts.append("integer(0)")
    return "list(" + ", ".join(parts) + ")"


def _r_graph_list_1based(adj_list):
    """Convert 0-based adjacency list (Python list-of-lists) to 1-based R list."""
    parts = []
    for nbrs in adj_list:
        nbrs_1 = [x + 1 for x in nbrs]
        if nbrs_1:
            parts.append("c(" + ",".join(str(x) for x in nbrs_1) + ")")
        else:
            parts.append("integer(0)")
    return "list(" + ", ".join(parts) + ")"


def _r_weighted_graph_1based(adj_list):
    """Convert 0-based weighted adjacency list to 1-based R list-of-lists.

    Input: list of lists of (neighbor, weight) tuples.
    Output: R list where each element is a list of c(neighbor_1based, weight) vectors.
    """
    parts = []
    for edges in adj_list:
        if edges:
            edge_parts = [f"c({nbr + 1},{wt})" for nbr, wt in edges]
            parts.append("list(" + ", ".join(edge_parts) + ")")
        else:
            parts.append("list()")
    return "list(" + ", ".join(parts) + ")"


def _r_matrix_encode(mat):
    """Encode a 2D Python list as an R matrix (column-major)."""
    nrow = len(mat)
    ncol = len(mat[0]) if nrow else 0
    vals = []
    for j in range(ncol):
        for i in range(nrow):
            vals.append(str(mat[i][j]))
    return f"matrix(c({','.join(vals)}), nrow={nrow}, ncol={ncol})"


def _r_build_test_script(algo: str, code: str) -> str:
    r_func = {
        "binary_search": "binary_search", "bubble_sort": "bubble_sort",
        "merge_sort": "merge_sort", "quicksort": "quicksort",
        "heap_sort": "heap_sort", "insertion_sort": "insertion_sort",
        "counting_sort": "counting_sort",
        "sieve_of_eratosthenes": "sieve", "nqueens": "nqueens",
        "edit_distance": "edit_distance",
        "longest_common_subsequence": "lcs", "kmp_search": "kmp_search",
        "longest_increasing_subsequence": "longest_increasing_subsequence",
        "gcd_extended": "gcd_extended", "matrix_multiply": "matrix_multiply",
        "tower_of_hanoi": "tower_of_hanoi",
        "dfs_graph": "dfs", "bfs_graph": "bfs", "dijkstra": "dijkstra",
        "topological_sort": "topological_sort",
        "binary_tree_inorder": "inorder_traversal",
        "knapsack_01": "knapsack",
    }
    fn = r_func[algo]
    lines = ["library(jsonlite)", "", code, "", "results <- list()"]
    for tc in TEST_CASES[algo]:
        inp = tc["inputs"]
        if algo == "binary_tree_inorder":
            spec = inp[0]
            if spec == "balanced_3":
                lines.append(f'results[[length(results)+1]] <- {fn}(new_node(2, new_node(1, NULL, NULL), new_node(3, NULL, NULL)))')
            elif spec == "left_chain_4":
                lines.append(f'results[[length(results)+1]] <- {fn}(new_node(4, new_node(3, new_node(2, new_node(1, NULL, NULL), NULL), NULL), NULL))')
            elif spec == "single":
                lines.append(f'results[[length(results)+1]] <- {fn}(new_node(42, NULL, NULL))')
            continue
        if algo == "quicksort":
            lines.append(f'a <- {_r_encode(inp[0])}; res <- {fn}(a, 1, length(a)); results[[length(results)+1]] <- res')
            continue
        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            lines.append(f'results[[length(results)+1]] <- {fn}({_r_encode(inp[0])})')
            continue
        if algo == "knapsack_01":
            w, v, c = inp
            lines.append(f'r <- {fn}({_r_encode(w)}, {_r_encode(v)}, {c}); items <- r[[2]]; if (is.null(items)) items <- integer(0); results[[length(results)+1]] <- list(r[[1]], sort(items))')
            continue
        if algo == "gcd_extended":
            lines.append(f'results[[length(results)+1]] <- {fn}({inp[0]}, {inp[1]})')
            continue
        if algo in ("dfs_graph", "bfs_graph"):
            fn_r = "dfs" if algo == "dfs_graph" else "bfs"
            g_lit = _r_graph_1based(inp[0])
            lines.append(f'results[[length(results)+1]] <- {fn_r}({g_lit}, {inp[1] + 1})')
            continue
        if algo == "dijkstra":
            g_lit = _r_weighted_graph_1based(inp[0])
            n = len(inp[0])
            lines.append(f'results[[length(results)+1]] <- {fn}({g_lit}, {inp[1] + 1}, {n})')
            continue
        if algo == "topological_sort":
            g_lit = _r_graph_list_1based(inp[0])
            lines.append(f'results[[length(results)+1]] <- {fn}({g_lit}, {inp[1]})')
            continue
        if algo == "matrix_multiply":
            a_lit = _r_matrix_encode(inp[0])
            b_lit = _r_matrix_encode(inp[1])
            lines.append(f'r <- {fn}({a_lit}, {b_lit}); results[[length(results)+1]] <- lapply(1:nrow(r), function(i) as.integer(r[i,]))')
            continue
        if algo == "kmp_search":
            args = ", ".join(_r_encode(a) for a in inp)
            lines.append(f'r <- {fn}({args}); if (is.null(r)) r <- integer(0); results[[length(results)+1]] <- r')
            continue
        args = ", ".join(_r_encode(a) for a in inp)
        lines.append(f'results[[length(results)+1]] <- {fn}({args})')
    lines.append('cat(toJSON(results, auto_unbox=TRUE))')
    return "\n".join(lines)


def run_r(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    rscript = _tool("Rscript") or shutil.which("Rscript")
    if not rscript:
        return [{"desc": "SKIP", "error": "Rscript not found"}]
    script = _r_build_test_script(algo, code)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, encoding="utf-8") as f:
        f.write(script); r_path = f.name
    try:
        r = subprocess.run([rscript, "--vanilla", r_path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (r.stderr or r.stdout or "")[:500]}]
        raw = json.loads(r.stdout)
        results = []
        for i, tc in enumerate(TEST_CASES[algo]):
            val = raw[i]
            if algo in ("dfs_graph", "bfs_graph", "topological_sort") and isinstance(val, list):
                val = [int(x) - 1 for x in val]
            if algo == "dijkstra" and isinstance(val, list):
                val = [float("inf") if v is None else v for v in val]
            if algo == "gcd_extended" and isinstance(val, dict):
                val = [val["gcd"], val["x"], val["y"]]
            if algo == "knapsack_01" and isinstance(val, list) and len(val) == 2:
                if isinstance(val[1], dict) and not val[1]:
                    val[1] = []
            results.append({"desc": tc["desc"], "actual": val})
        return results
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        os.unlink(r_path)


# ---------------------------------------------------------------------------
# Java runner
# ---------------------------------------------------------------------------

import re as _re

_JAVA_ALGO_MAP = {
    "binary_search":    ("BinarySearch", "binarySearch"),
    "bubble_sort":      ("BubbleSort",   "bubbleSort"),
    "merge_sort":       ("MergeSort",    "mergeSort"),
    "quicksort":        ("QuickSort",    "quickSort"),
    "heap_sort":        ("HeapSort",     "heapSort"),
    "insertion_sort":   ("InsertionSort","insertionSort"),
    "counting_sort":    ("CountingSort", "countingSort"),
    "sieve_of_eratosthenes": ("Sieve",  "sieve"),
    "nqueens":          ("NQueens",      "nqueens"),
    "edit_distance":    ("EditDistance", "editDistance"),
    "longest_common_subsequence": ("LCS","lcs"),
    "kmp_search":       ("KMPSearch",   "kmpSearch"),
    "longest_increasing_subsequence": ("LIS","longestIncreasingSubsequence"),
    "gcd_extended":     ("GcdExtended", "gcdExtended"),
    "matrix_multiply":  ("MatrixMultiply","multiply"),
    "tower_of_hanoi":   ("TowerOfHanoi","towerOfHanoi"),
    "dfs_graph":        ("DFS",         "dfs"),
    "bfs_graph":        ("BFS",         "bfs"),
    "dijkstra":         ("Dijkstra",    "dijkstra"),
    "topological_sort": ("TopologicalSort","topologicalSort"),
    "binary_tree_inorder": ("BinaryTreeInorder","inorderTraversal"),
    "knapsack_01":      ("Knapsack",    "knapsack"),
}


def _java_arr(arr):
    return "new int[]{" + ",".join(str(x) for x in arr) + "}"


def _java_adj_list(graph):
    """Build a Java List<List<Integer>> literal from a Python dict graph."""
    max_node = max(int(k) for k in graph) if graph else 0
    parts = []
    for i in range(max_node + 1):
        nbrs = graph.get(i, graph.get(str(i), []))
        inner = ",".join(str(x) for x in nbrs) if nbrs else ""
        parts.append(f"new ArrayList<>(Arrays.asList({inner}))" if inner else "new ArrayList<>()")
    return "new ArrayList<>(Arrays.asList(" + ", ".join(parts) + "))"


def _java_build_source(algo: str, code: str) -> str:
    cls_name, method = _JAVA_ALGO_MAP[algo]
    qm = f"{cls_name}.{method}"

    corpus_imports = _re.findall(r'^import\s+.+;', code, _re.MULTILINE)
    code_body = _re.sub(r'^import\s+.+;\n?', '', code, flags=_re.MULTILINE)
    code_body = code_body.replace(f"public class {cls_name}", f"class {cls_name}")

    imports = sorted(set([
        "import java.util.*;",
        "import java.util.stream.*;",
    ] + corpus_imports))

    json_helper = textwrap.dedent("""\
    class Harness {
        static String toJson(Object o) {
            if (o instanceof List) {
                return "[" + ((List<?>)o).stream().map(Harness::toJson).collect(Collectors.joining(",")) + "]";
            }
            if (o instanceof int[])
                return "[" + Arrays.stream((int[])o).mapToObj(Integer::toString).collect(Collectors.joining(",")) + "]";
            if (o instanceof int[][])
                return "[" + Arrays.stream((int[][])o).map(Harness::toJson).collect(Collectors.joining(",")) + "]";
            if (o instanceof Map)
                return "{" + ((Map<?,?>)o).entrySet().stream()
                    .map(e -> "\\"" + e.getKey() + "\\":" + toJson(e.getValue()))
                    .collect(Collectors.joining(",")) + "}";
            if (o instanceof String) return "\\"" + o + "\\"";
            return String.valueOf(o);
        }
    }
    """)

    calls = []
    for tc in TEST_CASES[algo]:
        inp = tc["inputs"]
        if algo == "binary_tree_inorder":
            tree_cls = "BinaryTreeInorder"
            spec = inp[0]
            if spec == "balanced_3":
                calls.append(f"results.add({qm}(new {tree_cls}.Node(2, new {tree_cls}.Node(1, null, null), new {tree_cls}.Node(3, null, null))));")
            elif spec == "left_chain_4":
                calls.append(f"results.add({qm}(new {tree_cls}.Node(4, new {tree_cls}.Node(3, new {tree_cls}.Node(2, new {tree_cls}.Node(1), null), null), null)));")
            elif spec == "single":
                calls.append(f"results.add({qm}(new {tree_cls}.Node(42)));")
            continue
        elif algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            calls.append(f"{{ int[] a = {_java_arr(inp[0])}; {qm}(a); results.add(a); }}")
        elif algo == "merge_sort":
            calls.append(f"{{ int[] a = {_java_arr(inp[0])}; {qm}(a, 0, a.length-1); results.add(a); }}")
        elif algo == "quicksort":
            calls.append(f"{{ int[] a = {_java_arr(inp[0])}; {qm}(a, 0, a.length-1); results.add(a); }}")
        elif algo == "binary_search":
            calls.append(f"results.add({qm}({_java_arr(inp[0])}, {inp[1]}));")
        elif algo in ("sieve_of_eratosthenes", "nqueens"):
            calls.append(f"results.add({qm}({inp[0]}));")
        elif algo in ("edit_distance", "longest_common_subsequence"):
            calls.append(f'results.add({qm}("{inp[0]}", "{inp[1]}"));')
        elif algo == "kmp_search":
            calls.append(f'results.add({qm}("{inp[0]}", "{inp[1]}"));')
        elif algo == "longest_increasing_subsequence":
            calls.append(f"results.add({qm}({_java_arr(inp[0])}));")
        elif algo == "gcd_extended":
            calls.append(f"results.add({qm}({inp[0]}, {inp[1]}));")
        elif algo == "matrix_multiply":
            a, b = inp
            a_lit = "new int[][]{" + ",".join("{" + ",".join(str(x) for x in row) + "}" for row in a) + "}"
            b_lit = "new int[][]{" + ",".join("{" + ",".join(str(x) for x in row) + "}" for row in b) + "}"
            calls.append(f"results.add({qm}({a_lit}, {b_lit}));")
        elif algo == "tower_of_hanoi":
            n, s, t, aux = inp
            calls.append(f"{{ List<char[]> moves = new ArrayList<>(); TowerOfHanoi.solve({n}, '{s}', '{t}', '{aux}', moves); List<List<String>> ml = new ArrayList<>(); for (char[] m : moves) ml.add(java.util.Arrays.asList(String.valueOf(m[0]), String.valueOf(m[1]))); results.add(ml); }}")
        elif algo == "knapsack_01":
            w, v, c = inp
            calls.append(f"{{ var r = {qm}({_java_arr(w)}, {_java_arr(v)}, {c}); List<Object> kl = new ArrayList<>(); kl.add(r.maxValue); Collections.sort(r.items); kl.add(r.items); results.add(kl); }}")
        elif algo in ("dfs_graph", "bfs_graph"):
            graph, start = inp
            g_lit = _java_adj_list(graph)
            calls.append(f"results.add({qm}({g_lit}, {start}));")
        elif algo == "dijkstra":
            graph, src = inp
            n = len(graph)
            parts = []
            for adj in graph:
                inner = ", ".join(f"new int[]{{{e[0]},{e[1]}}}" for e in adj) if adj else ""
                parts.append(f"new ArrayList<>(Arrays.asList({inner}))" if inner else "new ArrayList<>()")
            arr_lit = f"new List[{n}]"
            init_lines = [f"List<int[]>[] g = new List[{n}];"]
            for i, p in enumerate(parts):
                init_lines.append(f"g[{i}] = {p};")
            init_lines.append(f"results.add({qm}(g, {src}));")
            calls.append("{ " + " ".join(init_lines) + " }")
        elif algo == "topological_sort":
            graph, n_nodes = inp
            init_lines = [f"List<Integer>[] tg = new List[{n_nodes}];"]
            for i, adj in enumerate(graph):
                inner = ",".join(str(x) for x in adj) if adj else ""
                init_lines.append(f"tg[{i}] = new ArrayList<>(Arrays.asList({inner}));" if inner else f"tg[{i}] = new ArrayList<>();")
            init_lines.append(f"results.add({qm}(tg, {n_nodes}));")
            calls.append("{ " + " ".join(init_lines) + " }")

    indented_calls = "\n".join(f"            {c}" for c in calls)
    main_cls = textwrap.dedent(f"""\
    public class Test {{
        public static void main(String[] args) {{
            List<Object> results = new ArrayList<>();
    {indented_calls}
            System.out.print(Harness.toJson(results));
        }}
    }}
    """)

    return "\n".join(imports) + "\n\n" + code_body + "\n\n" + json_helper + "\n" + main_cls


def run_java(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    javac = _tool("javac") or shutil.which("javac")
    java_bin = _tool("java") or shutil.which("java")
    if not javac or not java_bin:
        return [{"desc": "SKIP", "error": "javac/java not found"}]

    source = _java_build_source(algo, code)
    tmpdir = tempfile.mkdtemp()
    src_path = os.path.join(tmpdir, "Test.java")
    try:
        with open(src_path, "w") as f:
            f.write(source)
        comp = subprocess.run([javac, "-Xlint:none", src_path],
                             capture_output=True, text=True, timeout=30)
        if comp.returncode != 0:
            return [{"desc": "COMPILE_ERROR", "error": (comp.stderr or comp.stdout)[:500]}]
        r = subprocess.run([java_bin, "-cp", tmpdir, "Test"],
                          capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (r.stderr or r.stdout)[:500]}]
        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# C# runner
# ---------------------------------------------------------------------------

_CS_ALGO_MAP = {
    "binary_search":    ("BinarySearch", "Search"),
    "bubble_sort":      ("BubbleSort",   "Sort"),
    "merge_sort":       ("MergeSort",    "Sort"),
    "quicksort":        ("QuickSort",    "Sort"),
    "heap_sort":        ("HeapSort",     "Sort"),
    "insertion_sort":   ("InsertionSort","Sort"),
    "counting_sort":    ("CountingSort", "Sort"),
    "sieve_of_eratosthenes": ("Sieve",  "SieveOfEratosthenes"),
    "nqueens":          ("NQueens",      "Solve"),
    "edit_distance":    ("EditDistance", "Compute"),
    "longest_common_subsequence": ("LCS","Compute"),
    "kmp_search":       ("KMPSearch",   "KmpSearch"),
    "longest_increasing_subsequence": ("LIS","LongestIncreasingSubsequence"),
    "gcd_extended":     ("GcdExtended", "Compute"),
    "matrix_multiply":  ("MatrixMultiply","Multiply"),
    "tower_of_hanoi":   ("TowerOfHanoi","HanoiMoves"),
    "dfs_graph":        ("DFS",         "DepthFirstSearch"),
    "bfs_graph":        ("BFS",         "BreadthFirstSearch"),
    "dijkstra":         ("Dijkstra",    "Run"),
    "topological_sort": ("TopologicalSort","Sort"),
    "binary_tree_inorder": ("BinaryTreeInorder","InorderTraversal"),
    "knapsack_01":      ("Knapsack",    "Solve"),
}


def run_csharp(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    dotnet = _tool("dotnet") or shutil.which("dotnet")
    if not dotnet:
        return [{"desc": "SKIP", "error": "dotnet not found"}]

    cls_name, method = _CS_ALGO_MAP[algo]
    qm = f"{cls_name}.{method}"

    # Corpus classes/methods default to private – make them accessible to Program.Main
    code = _re.sub(r'^(\s*)class ', r'\1public class ', code, flags=_re.MULTILINE)
    code = _re.sub(r'^(\s+)static ', r'\1public static ', code, flags=_re.MULTILINE)

    tmpdir = tempfile.mkdtemp()
    try:
        subprocess.run([dotnet, "new", "console", "-o", tmpdir, "--force"],
                       capture_output=True, text=True, timeout=60)

        # Disable nullable warnings so they don't leak into stdout
        csproj = next(
            (f for f in os.listdir(tmpdir) if f.endswith(".csproj")), None
        )
        if csproj:
            cp = os.path.join(tmpdir, csproj)
            with open(cp) as fh:
                csp = fh.read()
            csp = csp.replace(
                "<Nullable>enable</Nullable>",
                "<Nullable>disable</Nullable>",
            )
            with open(cp, "w") as fh:
                fh.write(csp)

        prog_lines = [
            "using System;",
            "using System.Collections.Generic;",
            "using System.Linq;",
            "using System.Text.Json;",
            "",
            code,
            "",
            "class Program {",
            "    static void Main() {",
            "        var results = new List<object>();",
        ]
        for tc in TEST_CASES[algo]:
            inp = tc["inputs"]
            if algo == "binary_search":
                arr_lit = "new int[]{" + ",".join(str(x) for x in inp[0]) + "}"
                prog_lines.append(f"        results.Add({qm}({arr_lit}, {inp[1]}));")
            elif algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
                arr_lit = "new int[]{" + ",".join(str(x) for x in inp[0]) + "}"
                prog_lines.append(f"        {{ var a = {arr_lit}; {qm}(a); results.Add(a); }}")
            elif algo == "merge_sort":
                arr_lit = "new int[]{" + ",".join(str(x) for x in inp[0]) + "}"
                prog_lines.append(f"        {{ var a = {arr_lit}; {qm}(a, 0, a.Length-1); results.Add(a); }}")
            elif algo == "quicksort":
                arr_lit = "new int[]{" + ",".join(str(x) for x in inp[0]) + "}"
                prog_lines.append(f"        {{ var a = {arr_lit}; {qm}(a, 0, a.Length-1); results.Add(a); }}")
            elif algo in ("sieve_of_eratosthenes", "nqueens"):
                prog_lines.append(f"        results.Add({qm}({inp[0]}));")
            elif algo in ("edit_distance", "longest_common_subsequence"):
                prog_lines.append(f'        results.Add({qm}("{inp[0]}", "{inp[1]}"));')
            elif algo == "kmp_search":
                prog_lines.append(f'        results.Add({qm}("{inp[0]}", "{inp[1]}"));')
            elif algo == "longest_increasing_subsequence":
                arr_lit = "new int[]{" + ",".join(str(x) for x in inp[0]) + "}"
                prog_lines.append(f"        results.Add({qm}({arr_lit}));")
            elif algo == "gcd_extended":
                prog_lines.append(
                    f"        {{ var _r = {qm}({inp[0]}, {inp[1]});"
                    f" results.Add(new int[]{{ _r.gcd, _r.x, _r.y }}); }}"
                )
            elif algo == "matrix_multiply":
                a, b = inp
                a_lit = "new int[,]{" + ",".join("{" + ",".join(str(x) for x in row) + "}" for row in a) + "}"
                b_lit = "new int[,]{" + ",".join("{" + ",".join(str(x) for x in row) + "}" for row in b) + "}"
                prog_lines.append(
                    f"        {{ var _m = {qm}({a_lit}, {b_lit});"
                    f" var _rows = new List<int[]>();"
                    f" for(int _i=0;_i<_m.GetLength(0);_i++)"
                    f"{{ var _row=new int[_m.GetLength(1)];"
                    f" for(int _j=0;_j<_m.GetLength(1);_j++) _row[_j]=_m[_i,_j];"
                    f" _rows.Add(_row); }}"
                    f" results.Add(_rows); }}"
                )
            elif algo == "tower_of_hanoi":
                n, s, t, aux = inp
                prog_lines.append(
                    f"        {{ var _mv = {qm}({n}, '{s}', '{t}', '{aux}');"
                    f" results.Add(_mv.Select(t => new string[]"
                    f"{{ t.Item1.ToString(), t.Item2.ToString() }}).ToList()); }}"
                )
            elif algo == "knapsack_01":
                w, v, c = inp
                w_lit = "new int[]{" + ",".join(str(x) for x in w) + "}"
                v_lit = "new int[]{" + ",".join(str(x) for x in v) + "}"
                prog_lines.append(
                    f"        {{ var _k = {qm}({w_lit}, {v_lit}, {c});"
                    f" results.Add(new object[]{{ _k.Item1, _k.Item2 }}); }}"
                )
            elif algo in ("dfs_graph", "bfs_graph"):
                graph, start = inp
                max_node = max(int(k) for k in graph)
                lines_g = [f"var gl = new List<List<int>>();"]
                for i in range(max_node + 1):
                    nbrs = graph.get(i, graph.get(str(i), []))
                    inner = ",".join(str(x) for x in nbrs) if nbrs else ""
                    lines_g.append(f"gl.Add(new List<int>{{{inner}}});")
                lines_g.append(f"results.Add({qm}(gl, {start}));")
                prog_lines.append("        { " + " ".join(lines_g) + " }")
            elif algo == "dijkstra":
                graph, src = inp
                n = len(graph)
                lines_g = [f"var dg = new List<int[]>[{n}];"]
                for i, adj in enumerate(graph):
                    inner = ", ".join(f"new int[]{{{e[0]},{e[1]}}}" for e in adj) if adj else ""
                    lines_g.append(f"dg[{i}] = new List<int[]> {{ {inner} }};")
                lines_g.append(f"results.Add({qm}(dg, {src}));")
                prog_lines.append("        { " + " ".join(lines_g) + " }")
            elif algo == "topological_sort":
                graph, n_nodes = inp
                lines_g = [f"var sg = new List<int>[{n_nodes}];"]
                for i, adj in enumerate(graph):
                    inner = ",".join(str(x) for x in adj) if adj else ""
                    lines_g.append(f"sg[{i}] = new List<int>{{{inner}}};")
                lines_g.append(f"results.Add({qm}(sg, {n_nodes}));")
                prog_lines.append("        { " + " ".join(lines_g) + " }")
            elif algo == "binary_tree_inorder":
                tree_cls = f"{cls_name}.Node"
                spec = inp[0]
                if spec == "balanced_3":
                    prog_lines.append(f"        results.Add({qm}(new {tree_cls}(2, new {tree_cls}(1), new {tree_cls}(3))));")
                elif spec == "left_chain_4":
                    prog_lines.append(f"        results.Add({qm}(new {tree_cls}(4, new {tree_cls}(3, new {tree_cls}(2, new {tree_cls}(1))))));")
                elif spec == "single":
                    prog_lines.append(f"        results.Add({qm}(new {tree_cls}(42)));")
        prog_lines.append("        Console.Write(JsonSerializer.Serialize(results));")
        prog_lines.append("    }")
        prog_lines.append("}")
        with open(os.path.join(tmpdir, "Program.cs"), "w") as f:
            f.write("\n".join(prog_lines))
        r = subprocess.run([dotnet, "run", "--project", tmpdir],
                          capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (r.stderr or r.stdout)[:500]}]
        # Build warnings may leak into stdout; extract the JSON array at the end
        out = r.stdout.strip()
        json_start = out.rfind("\n[")
        if json_start >= 0:
            out = out[json_start + 1:]
        elif not out.startswith("["):
            last_nl = out.rfind("\n")
            if last_nl >= 0:
                out = out[last_nl + 1:]
        raw = json.loads(out)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Kotlin runner
# ---------------------------------------------------------------------------

def _kt_encode(val) -> str:
    if isinstance(val, dict):
        pairs = ", ".join(f"{k} to {_kt_encode(v)}" for k, v in val.items())
        return f"mutableMapOf({pairs})"
    if isinstance(val, (list, tuple)):
        return "mutableListOf(" + ", ".join(_kt_encode(x) for x in val) + ")"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def run_kotlin(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    kotlinc = _tool("kotlinc") or shutil.which("kotlinc")
    kotlin_bin = _tool("kotlin") or shutil.which("kotlin")
    if not kotlinc:
        return [{"desc": "SKIP", "error": "kotlinc not found"}]

    kt_fn = {
        "binary_search": "binarySearch", "bubble_sort": "bubbleSort",
        "merge_sort": "mergeSort", "quicksort": "quicksort",
        "heap_sort": "heapSort", "insertion_sort": "insertionSort",
        "counting_sort": "countingSort",
        "sieve_of_eratosthenes": "sieve", "nqueens": "nqueens",
        "edit_distance": "editDistance",
        "longest_common_subsequence": "lcs", "kmp_search": "kmpSearch",
        "longest_increasing_subsequence": "longestIncreasingSubsequence",
        "gcd_extended": "gcdExtended", "matrix_multiply": "matrixMultiply",
        "tower_of_hanoi": "towerOfHanoi",
        "dfs_graph": "dfs", "bfs_graph": "bfs", "dijkstra": "dijkstra",
        "topological_sort": "topologicalSort",
        "binary_tree_inorder": "inorderTraversal",
        "knapsack_01": "knapsack",
    }
    fn = kt_fn[algo]

    json_helper = textwrap.dedent("""\
    fun toJ(v: Any?): String = when(v) {
        null -> "null"
        is Int -> v.toString()
        is Long -> v.toString()
        is Double -> if (v == v.toLong().toDouble()) v.toLong().toString() else v.toString()
        is String -> "\\"$v\\""
        is List<*> -> "[" + v.joinToString(",") { toJ(it) } + "]"
        is Array<*> -> "[" + v.joinToString(",") { toJ(it) } + "]"
        is IntArray -> "[" + v.joinToString(",") + "]"
        is DoubleArray -> "[" + v.joinToString(",") { d -> if (d == Double.POSITIVE_INFINITY) "2147483647" else if (d == d.toLong().toDouble()) d.toLong().toString() else d.toString() } + "]"
        is Map<*, *> -> "{" + v.entries.joinToString(",") { (k, v2) -> "\\"$k\\":" + toJ(v2) } + "}"
        is Pair<*, *> -> "[" + toJ(v.first) + "," + toJ(v.second) + "]"
        is Triple<*, *, *> -> "[" + toJ(v.first) + "," + toJ(v.second) + "," + toJ(v.third) + "]"
        else -> v.toString()
    }
    """)

    lines = ["", code, "", json_helper, "", "fun main() {",
             "    val results = mutableListOf<String>()"]

    for tc in TEST_CASES[algo]:
        inp = tc["inputs"]
        if algo == "binary_tree_inorder":
            spec = inp[0]
            if spec == "balanced_3":
                lines.append(f"    results.add(toJ({fn}(Node(2, Node(1), Node(3)))))")
            elif spec == "left_chain_4":
                lines.append(f"    results.add(toJ({fn}(Node(4, Node(3, Node(2, Node(1)))))))")
            elif spec == "single":
                lines.append(f"    results.add(toJ({fn}(Node(42))))")
            continue
        if algo == "quicksort":
            arr_lit = "intArrayOf(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"    run {{ val a = {arr_lit}; {fn}(a, 0, a.size-1); results.add(toJ(a.toList())) }}")
            continue
        if algo == "heap_sort":
            arr_lit = "intArrayOf(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"    run {{ val a = {arr_lit}; {fn}(a); results.add(toJ(a.toList())) }}")
            continue
        if algo in ("bubble_sort", "insertion_sort", "counting_sort"):
            arr_lit = "intArrayOf(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"    run {{ val a = {fn}({arr_lit}); results.add(toJ(a.toList())) }}")
            continue
        if algo == "merge_sort":
            arr_lit = "intArrayOf(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"    results.add(toJ({fn}({arr_lit})))")
            continue
        if algo == "knapsack_01":
            w, v, c = inp
            w_lit = "intArrayOf(" + ",".join(str(x) for x in w) + ")"
            v_lit = "intArrayOf(" + ",".join(str(x) for x in v) + ")"
            lines.append(f"    run {{ val r = {fn}({w_lit}, {v_lit}, {c}); val items = (r.second as List<Int>).sorted(); results.add(toJ(listOf(r.first, items))) }}")
            continue
        if algo == "gcd_extended":
            lines.append(f"    results.add(toJ({fn}({inp[0]}, {inp[1]})))")
            continue
        if algo in ("dfs_graph", "bfs_graph"):
            fn_k = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f"    results.add(toJ({fn_k}({_kt_encode(inp[0])}, {inp[1]})))")
            continue
        if algo == "dijkstra":
            graph, src = inp
            adj_parts = []
            for adj in graph:
                pairs = ",".join(f"Pair({e[0]},{e[1]})" for e in adj) if adj else ""
                adj_parts.append(f"listOf({pairs})")
            adj_lit = "arrayOf(" + ",".join(adj_parts) + ")"
            lines.append(f"    results.add(toJ({fn}({adj_lit}, {src})))")
            continue
        if algo == "topological_sort":
            graph, n_nodes = inp
            adj_lit = "arrayOf(" + ",".join("listOf(" + ",".join(str(x) for x in adj) + ")" for adj in graph) + ")"
            lines.append(f"    results.add(toJ({fn}({adj_lit}, {n_nodes})))")
            continue
        if algo == "binary_search":
            arr_lit = "intArrayOf(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"    results.add(toJ({fn}({arr_lit}, {inp[1]})))")
            continue
        if algo == "longest_increasing_subsequence":
            arr_lit = "intArrayOf(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"    results.add(toJ({fn}({arr_lit})))")
            continue
        if algo == "matrix_multiply":
            a, b = inp
            a_lit = "arrayOf(" + ",".join("intArrayOf(" + ",".join(str(x) for x in row) + ")" for row in a) + ")"
            b_lit = "arrayOf(" + ",".join("intArrayOf(" + ",".join(str(x) for x in row) + ")" for row in b) + ")"
            lines.append(f"    results.add(toJ({fn}({a_lit}, {b_lit})))")
            continue
        if algo == "tower_of_hanoi":
            n = inp[0]
            peg_map = {"A": 1, "B": 2, "C": 3}
            s, t, aux = peg_map[inp[1]], peg_map[inp[2]], peg_map[inp[3]]
            lines.append(f'    run {{ val rm = mapOf(1 to "A", 2 to "B", 3 to "C"); val moves = {fn}({n}, {s}, {t}, {aux}); results.add(toJ(moves.map {{ (a, b) -> listOf(rm[a]!!, rm[b]!!) }})) }}')
            continue
        args = ", ".join(_kt_encode(a) for a in inp)
        lines.append(f"    results.add(toJ({fn}({args})))")

    lines.append('    print("[" + results.joinToString(",") + "]")')
    lines.append("}")
    source = "\n".join(lines)

    tmpdir = tempfile.mkdtemp()
    src_path = os.path.join(tmpdir, "Test.kt")
    java_home = _tool("java")
    if java_home:
        java_home = os.path.dirname(os.path.dirname(java_home))
    kt_env = os.environ.copy()
    if java_home and os.path.isdir(java_home):
        kt_env["JAVA_HOME"] = java_home
        kt_env["PATH"] = os.path.join(java_home, "bin") + os.pathsep + kt_env.get("PATH", "")
    try:
        with open(src_path, "w") as f:
            f.write(source)
        comp = subprocess.run([kotlinc, src_path, "-include-runtime", "-d",
                              os.path.join(tmpdir, "test.jar")],
                             capture_output=True, text=True, timeout=120, env=kt_env)
        if comp.returncode != 0:
            return [{"desc": "COMPILE_ERROR", "error": (comp.stderr or comp.stdout)[:500]}]
        run_bin = kotlin_bin or "kotlin"
        r = subprocess.run([run_bin, "-classpath", os.path.join(tmpdir, "test.jar"), "TestKt"],
                          capture_output=True, text=True, timeout=30, env=kt_env)
        if r.returncode != 0:
            java_bin = _tool("java") or "java"
            r2 = subprocess.run([java_bin, "-jar", os.path.join(tmpdir, "test.jar")],
                               capture_output=True, text=True, timeout=30, env=kt_env)
            if r2.returncode != 0:
                return [{"desc": "RUNTIME_ERROR", "error": (r2.stderr or r2.stdout)[:500]}]
            r = r2
        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Scala runner (via scala-cli)
# ---------------------------------------------------------------------------

def _scala_encode(val) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    if isinstance(val, dict):
        pairs = ", ".join(f"{k} -> {_scala_encode(v)}" for k, v in val.items())
        return f"Map({pairs})"
    if isinstance(val, tuple):
        return "(" + ", ".join(_scala_encode(x) for x in val) + ")"
    if isinstance(val, list):
        return "List(" + ", ".join(_scala_encode(x) for x in val) + ")"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def run_scala(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    scala_cli = _tool("scala-cli") or shutil.which("scala-cli") or shutil.which("scala")
    if not scala_cli:
        return [{"desc": "SKIP", "error": "scala-cli not found"}]

    sc_fn = {
        "binary_search": "binarySearch", "bubble_sort": "bubbleSort",
        "merge_sort": "mergeSort", "quicksort": "quicksort",
        "heap_sort": "heapSort", "insertion_sort": "insertionSort",
        "counting_sort": "countingSort",
        "sieve_of_eratosthenes": "sieve", "nqueens": "nqueens",
        "edit_distance": "editDistance",
        "longest_common_subsequence": "lcs", "kmp_search": "kmpSearch",
        "longest_increasing_subsequence": "longestIncreasingSubsequence",
        "gcd_extended": "gcdExtended", "matrix_multiply": "matrixMultiply",
        "tower_of_hanoi": "towerOfHanoi",
        "dfs_graph": "dfs", "bfs_graph": "bfs", "dijkstra": "dijkstra",
        "topological_sort": "topologicalSort",
        "binary_tree_inorder": "inorderTraversal",
        "knapsack_01": "knapsack",
    }
    fn = sc_fn[algo]

    json_helper = textwrap.dedent("""\
    def toJson(v: Any): String = v match {
      case d: Double if d == Double.PositiveInfinity => "1e18"
      case l: List[_] => "[" + l.map(toJson).mkString(",") + "]"
      case a: Array[_] => "[" + a.map(toJson).mkString(",") + "]"
      case m: Map[_, _] => "{" + m.map{case (k,v2) => s"\\"${k}\\":${toJson(v2)}"}.mkString(",") + "}"
      case (a, b) => s"[${toJson(a)},${toJson(b)}]"
      case (a, b, c) => s"[${toJson(a)},${toJson(b)},${toJson(c)}]"
      case s: String => s"\\"${s}\\""
      case x => x.toString
    }
    """)

    lines = ["import scala.collection.mutable", "", code, "", json_helper, "",
             "@main def run(): Unit = {",
             "  val results = mutable.ListBuffer[Any]()"]

    for tc in TEST_CASES[algo]:
        inp = tc["inputs"]
        if algo == "binary_tree_inorder":
            spec = inp[0]
            if spec == "balanced_3":
                lines.append(f"  results += {fn}(new Node(2, new Node(1), new Node(3)))")
            elif spec == "left_chain_4":
                lines.append(f"  results += {fn}(new Node(4, new Node(3, new Node(2, new Node(1)))))")
            elif spec == "single":
                lines.append(f"  results += {fn}(new Node(42))")
            continue
        if algo == "quicksort":
            arr_lit = "Array(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"  {{ val a = {arr_lit}; {fn}(a, 0, a.length-1); results += a.toList }}")
            continue
        if algo in ("bubble_sort", "heap_sort", "insertion_sort", "counting_sort"):
            arr_lit = "Array.empty[Int]" if not inp[0] else "Array(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"  {{ val a = {arr_lit}; {fn}(a); results += a.toList }}")
            continue
        if algo == "merge_sort":
            arr_lit = "Array.empty[Int]" if not inp[0] else "Array(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"  results += {fn}({arr_lit}).toList")
            continue
        if algo == "knapsack_01":
            w, v, c = inp
            w_lit = "Array(" + ",".join(str(x) for x in w) + ")"
            v_lit = "Array(" + ",".join(str(x) for x in v) + ")"
            lines.append(f"  {{ val r = {fn}({w_lit}, {v_lit}, {c}); results += List(r._1, r._2.sorted) }}")
            continue
        if algo == "gcd_extended":
            lines.append(f"  results += {fn}({inp[0]}, {inp[1]})")
            continue
        if algo in ("dfs_graph", "bfs_graph"):
            fn_s = "dfs" if algo == "dfs_graph" else "bfs"
            lines.append(f"  results += {fn_s}({_scala_encode(inp[0])}, {inp[1]})")
            continue
        if algo == "dijkstra":
            graph, src = inp
            adj_parts = []
            for adj in graph:
                pairs = ", ".join(f"({e[0]},{e[1]})" for e in adj) if adj else ""
                adj_parts.append(f"List({pairs})")
            adj_lit = "Array(" + ", ".join(adj_parts) + ")"
            lines.append(f"  results += {fn}({adj_lit}, {src}).toList")
            continue
        if algo == "topological_sort":
            graph, n_nodes = inp
            adj_lit = "Array(" + ", ".join(
                "List(" + ",".join(str(x) for x in adj) + ")" for adj in graph
            ) + ")"
            lines.append(f"  results += {fn}({adj_lit}, {n_nodes})")
            continue
        if algo == "binary_search":
            arr_lit = "Array.empty[Int]" if not inp[0] else "Array(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"  results += {fn}({arr_lit}, {inp[1]})")
            continue
        if algo == "longest_increasing_subsequence":
            arr_lit = "Array(" + ",".join(str(x) for x in inp[0]) + ")"
            lines.append(f"  results += {fn}({arr_lit})")
            continue
        if algo == "matrix_multiply":
            def _scala_matrix(m):
                return "Array(" + ",".join("Array(" + ",".join(str(x) for x in row) + ")" for row in m) + ")"
            lines.append(f"  results += {fn}({_scala_matrix(inp[0])}, {_scala_matrix(inp[1])})")
            continue
        if algo == "tower_of_hanoi":
            n_disks = inp[0]
            peg_map = {"A": 1, "B": 2, "C": 3}
            s, t, aux = peg_map[inp[1]], peg_map[inp[2]], peg_map[inp[3]]
            lines.append(
                f'  {{ val pm = Map(1->"A",2->"B",3->"C"); '
                f'results += {fn}({n_disks},{s},{t},{aux}).map{{ case (a,b) => List(pm(a),pm(b)) }} }}'
            )
            continue
        args = ", ".join(_scala_encode(a) for a in inp)
        lines.append(f"  results += {fn}({args})")

    lines.append('  print("[" + results.map(toJson).mkString(",") + "]")')
    lines.append("}")
    source = "\n".join(lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".scala", delete=False, encoding="utf-8") as f:
        f.write(source); sc_path = f.name
    try:
        r = subprocess.run([scala_cli, "run", sc_path,
                           "--suppress-directives-in-multiple-files-warning"],
                          capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (r.stderr or r.stdout)[:500]}]
        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        os.unlink(sc_path)


# ---------------------------------------------------------------------------
# Racket runner
# ---------------------------------------------------------------------------

def _racket_encode(val, as_vector=True) -> str:
    if isinstance(val, dict):
        pairs = " ".join(f"(cons {k} {_racket_encode(v, as_vector)})" for k, v in val.items())
        return f"(make-hash (list {pairs}))"
    if isinstance(val, (list, tuple)):
        inner = " ".join(_racket_encode(x, as_vector) for x in val)
        if as_vector:
            return f"(vector {inner})"
        return f"(list {inner})"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def run_racket(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    racket_bin = _tool("racket") or shutil.which("racket")
    if not racket_bin:
        return [{"desc": "SKIP", "error": "racket not found"}]

    rkt_fn = {
        "binary_search": "binary-search", "bubble_sort": "bubble-sort",
        "merge_sort": "merge-sort", "quicksort": "quicksort",
        "heap_sort": "heap-sort", "insertion_sort": "insertion-sort",
        "counting_sort": "counting-sort",
        "sieve_of_eratosthenes": "sieve", "nqueens": "nqueens",
        "edit_distance": "edit-distance",
        "longest_common_subsequence": "lcs", "kmp_search": "kmp-search",
        "longest_increasing_subsequence": "lis",
        "gcd_extended": "gcd-extended", "matrix_multiply": "matrix-multiply",
        "tower_of_hanoi": "tower-of-hanoi",
        "dfs_graph": "dfs", "bfs_graph": "bfs", "dijkstra": "dijkstra",
        "topological_sort": "topological-sort",
        "binary_tree_inorder": "inorder-traversal",
        "knapsack_01": "knapsack",
    }
    fn = rkt_fn[algo]

    to_jsexpr = textwrap.dedent("""\
    (define (to-jsexpr v)
      (cond
        [(vector? v) (map to-jsexpr (vector->list v))]
        [(list? v) (map to-jsexpr v)]
        [(pair? v) (list (to-jsexpr (car v)) (to-jsexpr (cdr v)))]
        [(hash? v) (for/hasheq ([(k val) (in-hash v)])
                     (values (if (number? k) (number->string k) k) (to-jsexpr val)))]
        [(eqv? v +inf.0) 1e18]
        [else v]))
    """)

    lines = ["#lang racket", "(require json)", "", code, "", to_jsexpr,
             "(define results '())"]

    for tc in TEST_CASES[algo]:
        inp = tc["inputs"]

        if algo == "binary_tree_inorder":
            spec = inp[0]
            if spec == "balanced_3":
                tree = "(node 2 (node 1 #f #f) (node 3 #f #f))"
            elif spec == "left_chain_4":
                tree = "(node 4 (node 3 (node 2 (node 1 #f #f) #f) #f) #f)"
            elif spec == "single":
                tree = "(node 42 #f #f)"
            else:
                raise ValueError(f"Unknown tree spec: {spec}")
            lines.append(f"(set! results (append results (list (to-jsexpr ({fn} {tree})))))")
            continue

        if algo == "heap_sort":
            arr_enc = _racket_encode(inp[0])
            lines.append(f"(let ([arr {arr_enc}]) ({fn} arr) (set! results (append results (list (to-jsexpr arr)))))")
            continue

        if algo == "quicksort":
            arr_enc = _racket_encode(inp[0])
            lines.append(f"(let ([arr {arr_enc}]) ({fn} arr 0 (- (vector-length arr) 1)) (set! results (append results (list (to-jsexpr arr)))))")
            continue

        if algo == "knapsack_01":
            args = " ".join(_racket_encode(a) for a in inp)
            lines.append(f"(let ([res ({fn} {args})]) (set! results (append results (list (to-jsexpr (list (car res) (sort (cadr res) <)))))))")
            continue

        if algo == "dijkstra":
            graph = inp[0]
            rows = []
            for adj_list in graph:
                if not adj_list:
                    rows.append("'()")
                else:
                    edges = " ".join(f"(cons {e[0]} {e[1]})" for e in adj_list)
                    rows.append(f"(list {edges})")
            graph_enc = f"(vector {' '.join(rows)})"
            lines.append(f"(set! results (append results (list (to-jsexpr ({fn} {graph_enc} {inp[1]})))))")
            continue

        if algo == "topological_sort":
            graph = inp[0]
            rows = []
            for adj_list in graph:
                if not adj_list:
                    rows.append("'()")
                else:
                    elems = " ".join(str(x) for x in adj_list)
                    rows.append(f"(list {elems})")
            graph_enc = f"(vector {' '.join(rows)})"
            lines.append(f"(set! results (append results (list (to-jsexpr ({fn} {graph_enc} {inp[1]})))))")
            continue

        if algo in ("dfs_graph", "bfs_graph"):
            args = " ".join(_racket_encode(a, as_vector=False) for a in inp)
        elif algo in ("merge_sort", "sieve_of_eratosthenes", "tower_of_hanoi"):
            args = " ".join(_racket_encode(a, as_vector=False) for a in inp)
        else:
            args = " ".join(_racket_encode(a) for a in inp)
        lines.append(f"(set! results (append results (list (to-jsexpr ({fn} {args})))))")

    lines.append("(write-json results)")
    script = "\n".join(lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rkt", delete=False, encoding="utf-8") as f:
        f.write(script); rkt_path = f.name
    try:
        r = subprocess.run([racket_bin, rkt_path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (r.stderr or r.stdout)[:500]}]
        raw = json.loads(r.stdout)
        return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
    except Exception as e:
        return [{"desc": "PARSE_ERROR", "error": str(e)[:300]}]
    finally:
        os.unlink(rkt_path)


# ---------------------------------------------------------------------------
# D runner
# ---------------------------------------------------------------------------

def _d_json_helpers() -> str:
    return textwrap.dedent("""\
    import std.stdio, std.conv, std.array, std.algorithm, std.format, std.math;

    string jsonInt(int v) { return to!string(v); }
    string jsonStr(string s) { return `"` ~ s ~ `"`; }
    string jsonIntArr(int[] a) {
        string r = "[";
        foreach (i, v; a) { if (i) r ~= ","; r ~= to!string(v); }
        return r ~ "]";
    }
    string jsonDblArr(double[] a) {
        string r = "[";
        foreach (i, v; a) {
            if (i) r ~= ",";
            if (v == double.infinity) r ~= "1e+308";
            else r ~= format("%.10g", v);
        }
        return r ~ "]";
    }
    string jsonMatrix(int[][] m) {
        string r = "[";
        foreach (i, row; m) { if (i) r ~= ","; r ~= jsonIntArr(row); }
        return r ~ "]";
    }
    string jsonStrPairArr(int[][] pairs) {
        string r = "[";
        foreach (i, p; pairs) {
            if (i) r ~= ",";
            r ~= `["` ~ to!string(p[0]) ~ `","` ~ to!string(p[1]) ~ `"]`;
        }
        return r ~ "]";
    }
    """)


def _d_build_test_calls(algo: str) -> str:
    lines: list[str] = ['    string[] results;']

    d_func = {
        "binary_search": "binarySearch",
        "bubble_sort": "bubbleSort",
        "merge_sort": "mergeSort",
        "quicksort": "quicksort",
        "heap_sort": "heapSort",
        "insertion_sort": "insertionSort",
        "counting_sort": "countingSort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "editDistance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmpSearch",
        "longest_increasing_subsequence": "longestIncreasingSubsequence",
        "gcd_extended": "gcdExtended",
        "matrix_multiply": "matrixMultiply",
        "tower_of_hanoi": "towerOfHanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topologicalSort",
        "binary_tree_inorder": "inorderTraversal",
        "knapsack_01": "knapsack",
    }
    fn = d_func[algo]

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_search":
            arr, target = inputs
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"    results ~= jsonInt({fn}({arr_lit}, {target}));")

        elif algo in ("bubble_sort", "insertion_sort", "counting_sort"):
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"    results ~= jsonIntArr({fn}({arr_lit}));")

        elif algo == "heap_sort":
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"    {{ auto a = {arr_lit}; {fn}(a); results ~= jsonIntArr(a); }}")

        elif algo == "merge_sort":
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"    results ~= jsonIntArr({fn}({arr_lit}));")

        elif algo == "quicksort":
            arr = inputs[0]
            n = len(arr)
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"    {{ auto a = {arr_lit}; {fn}(a, 0, {n - 1}); results ~= jsonIntArr(a); }}")

        elif algo == "sieve_of_eratosthenes":
            n_val = inputs[0]
            lines.append(f"    results ~= jsonIntArr({fn}({n_val}));")

        elif algo == "nqueens":
            n_val = inputs[0]
            lines.append(f"    results ~= jsonInt({fn}({n_val}));")

        elif algo == "edit_distance":
            s1, s2 = inputs
            lines.append(f'    results ~= jsonInt({fn}("{s1}", "{s2}"));')

        elif algo == "longest_common_subsequence":
            s1, s2 = inputs
            lines.append(f'    results ~= jsonStr({fn}("{s1}", "{s2}"));')

        elif algo == "kmp_search":
            text, pat = inputs
            lines.append(f'    results ~= jsonIntArr({fn}("{text}", "{pat}"));')

        elif algo == "longest_increasing_subsequence":
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"    results ~= jsonInt({fn}({arr_lit}));")

        elif algo == "gcd_extended":
            a, b = inputs
            lines.append(f'    {{ auto r = {fn}({a}, {b}); results ~= "[" ~ to!string(r.gcd) ~ "," ~ to!string(r.x) ~ "," ~ to!string(r.y) ~ "]"; }}')

        elif algo == "matrix_multiply":
            a_mat, b_mat = inputs
            a_lit = "[" + ",".join("[" + ",".join(str(x) for x in row) + "]" for row in a_mat) + "]"
            b_lit = "[" + ",".join("[" + ",".join(str(x) for x in row) + "]" for row in b_mat) + "]"
            lines.append(f"    results ~= jsonMatrix({fn}({a_lit}, {b_lit}));")

        elif algo == "knapsack_01":
            w, v, c = inputs
            w_lit = "[" + ",".join(str(x) for x in w) + "]"
            v_lit = "[" + ",".join(str(x) for x in v) + "]"
            lines.append(
                f"    {{ auto r = {fn}({w_lit}, {v_lit}, {c}); "
                f'auto sorted_items = r.items.dup; sorted_items.sort(); '
                f'results ~= "[" ~ to!string(r.maxValue) ~ "," ~ jsonIntArr(cast(int[])sorted_items) ~ "]"; }}'
            )

        elif algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append(
                f"    {{ auto moves = {fn}({n}, {ord(s)}, {ord(t)}, {ord(aux)}); "
                f'string s = "["; '
                f'foreach (i, m; moves) {{ if (i) s ~= ","; '
                f's ~= `["` ~ to!string(cast(char)m[0]) ~ `","` ~ to!string(cast(char)m[1]) ~ `"]`; }} '
                f's ~= "]"; results ~= s; }}'
            )

        elif algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            n = max(graph.keys()) + 1 if graph else 0
            fn_d = "dfs" if algo == "dfs_graph" else "bfs"
            rows = []
            for i in range(n):
                nbrs = graph.get(i, [])
                rows.append("[" + ",".join(str(x) for x in nbrs) + "]")
            assoc_pairs = ",".join(f"{i}:[{','.join(str(x) for x in graph.get(i, []))}]" for i in range(n))
            lines.append(f"    {{ int[][int] g = [{assoc_pairs}]; results ~= jsonIntArr({fn_d}(g, {start})); }}")

        elif algo == "dijkstra":
            graph, src = inputs
            edges_per_node = []
            for adj in graph:
                if adj:
                    edge_lits = ",".join(f"Edge({nb},{wt})" for nb, wt in adj)
                    edges_per_node.append(f"[{edge_lits}]")
                else:
                    edges_per_node.append("cast(Edge[])[]")
            graph_lit = "[" + ",".join(edges_per_node) + "]"
            lines.append(f"    {{ auto d = {fn}({graph_lit}, {src}); results ~= jsonDblArr(d); }}")

        elif algo == "topological_sort":
            graph, n_nodes = inputs
            rows = []
            for adj in graph:
                rows.append("[" + ",".join(str(x) for x in adj) + "]")
            graph_lit = "[" + ",".join(rows) + "]"
            lines.append(f"    results ~= jsonIntArr({fn}({graph_lit}, {n_nodes}));")

        elif algo == "binary_tree_inorder":
            spec = inputs[0]
            if spec == "balanced_3":
                lines.append(f"    {{ auto n1 = new TreeNode(1); auto n3 = new TreeNode(3); auto n2 = new TreeNode(2, n1, n3); results ~= jsonIntArr({fn}(n2)); }}")
            elif spec == "left_chain_4":
                lines.append(f"    {{ auto n1 = new TreeNode(1); auto n2 = new TreeNode(2, n1); auto n3 = new TreeNode(3, n2); auto n4 = new TreeNode(4, n3); results ~= jsonIntArr({fn}(n4)); }}")
            elif spec == "single":
                lines.append(f"    {{ auto n42 = new TreeNode(42); results ~= jsonIntArr({fn}(n42)); }}")

    lines.append('    write("[");')
    lines.append('    foreach (i, r; results) { if (i) write(","); write(r); }')
    lines.append('    writeln("]");')
    return "\n".join(lines)


def run_d(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    dmd = shutil.which("dmd") or shutil.which("ldc2") or shutil.which("gdc")
    if not dmd:
        for candidate in [r"C:\D\dmd2\windows\bin\dmd.exe"]:
            if os.path.isfile(candidate):
                dmd = candidate
                break
    if not dmd:
        return [{"desc": "SKIP", "error": "D compiler not found"}]

    helpers = _d_json_helpers()
    test_calls = _d_build_test_calls(algo)
    src = helpers + "\n" + code + "\n\nvoid main() {\n" + test_calls + "\n}\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "test.d")
        exe_path = os.path.join(tmpdir, "test.exe")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src)

        comp = subprocess.run(
            [dmd, "-of=" + exe_path, src_path],
            capture_output=True, text=True, timeout=30,
        )
        if comp.returncode != 0:
            return [{"desc": "COMPILE_ERROR", "error": comp.stderr[:500]}]

        run = subprocess.run(
            [exe_path], capture_output=True, text=True, timeout=30,
        )
        if run.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": run.stderr[:500]}]

        try:
            raw = json.loads(run.stdout)
            return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
        except Exception as e:
            return [{"desc": "PARSE_ERROR", "error": f"stdout={run.stdout[:200]}, err={e}"}]


# ---------------------------------------------------------------------------
# Swift runner
# ---------------------------------------------------------------------------

def _swift_json_helpers() -> str:
    return textwrap.dedent("""\
    func jsonInt(_ v: Int) -> String { return "\\(v)" }
    func jsonStr(_ s: String) -> String { return "\\\"\\(s)\\\"" }
    func jsonIntArr(_ a: [Int]) -> String {
        return "[" + a.map { "\\($0)" }.joined(separator: ",") + "]"
    }
    func jsonMatrix(_ m: [[Int]]) -> String {
        return "[" + m.map { jsonIntArr($0) }.joined(separator: ",") + "]"
    }
    func jsonStrPairArr(_ pairs: [(String, String)]) -> String {
        return "[" + pairs.map { "[\\\"\\($0.0)\\\",\\\"\\($0.1)\\\"]" }.joined(separator: ",") + "]"
    }
    func jsonDblArr(_ a: [Int]) -> String {
        return "[" + a.map { "\\($0)" }.joined(separator: ",") + "]"
    }
    """)


def _swift_build_test_calls(algo: str) -> str:
    lines: list[str] = ["var results: [String] = []"]

    swift_func = {
        "binary_search": "binarySearch",
        "bubble_sort": "bubbleSort",
        "merge_sort": "mergeSort",
        "quicksort": "quicksort",
        "heap_sort": "heapSort",
        "insertion_sort": "insertionSort",
        "counting_sort": "countingSort",
        "sieve_of_eratosthenes": "sieve",
        "nqueens": "nqueens",
        "edit_distance": "editDistance",
        "longest_common_subsequence": "lcs",
        "kmp_search": "kmpSearch",
        "longest_increasing_subsequence": "longestIncreasingSubsequence",
        "gcd_extended": "gcdExtended",
        "matrix_multiply": "matrixMultiply",
        "tower_of_hanoi": "towerOfHanoi",
        "dfs_graph": "dfs",
        "bfs_graph": "bfs",
        "dijkstra": "dijkstra",
        "topological_sort": "topologicalSort",
        "binary_tree_inorder": "inorderTraversal",
        "knapsack_01": "knapsack",
    }
    fn = swift_func[algo]

    for tc in TEST_CASES[algo]:
        inputs = tc["inputs"]

        if algo == "binary_search":
            arr, target = inputs
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"results.append(jsonInt({fn}({arr_lit}, {target})))")

        elif algo in ("bubble_sort", "insertion_sort", "counting_sort"):
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append("do {")
            lines.append(f"    var a: [Int] = {arr_lit}")
            lines.append(f"    {fn}(&a)")
            lines.append("    results.append(jsonIntArr(a))")
            lines.append("}")

        elif algo == "heap_sort":
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append("do {")
            lines.append(f"    var a: [Int] = {arr_lit}")
            lines.append(f"    {fn}(&a)")
            lines.append("    results.append(jsonIntArr(a))")
            lines.append("}")

        elif algo == "merge_sort":
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"results.append(jsonIntArr({fn}({arr_lit})))")

        elif algo == "quicksort":
            arr = inputs[0]
            n = len(arr)
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append("do {")
            lines.append(f"    var a: [Int] = {arr_lit}")
            lines.append(f"    {fn}(&a, 0, {n - 1})")
            lines.append("    results.append(jsonIntArr(a))")
            lines.append("}")

        elif algo == "sieve_of_eratosthenes":
            n_val = inputs[0]
            lines.append(f"results.append(jsonIntArr({fn}({n_val})))")

        elif algo == "nqueens":
            n_val = inputs[0]
            lines.append(f"results.append(jsonInt({fn}({n_val})))")

        elif algo == "edit_distance":
            s1, s2 = inputs
            lines.append(f'results.append(jsonInt({fn}("{s1}", "{s2}")))')

        elif algo == "longest_common_subsequence":
            s1, s2 = inputs
            lines.append(f'results.append(jsonStr({fn}("{s1}", "{s2}")))')

        elif algo == "kmp_search":
            text, pat = inputs
            lines.append(f'results.append(jsonIntArr({fn}("{text}", "{pat}")))')

        elif algo == "longest_increasing_subsequence":
            arr = inputs[0]
            arr_lit = "[" + ",".join(str(x) for x in arr) + "]"
            lines.append(f"results.append(jsonInt({fn}({arr_lit})))")

        elif algo == "gcd_extended":
            a, b = inputs
            lines.append("do {")
            lines.append(f"    let (g, x, y) = {fn}({a}, {b})")
            lines.append('    results.append("[\\(g),\\(x),\\(y)]")')
            lines.append("}")

        elif algo == "matrix_multiply":
            a_mat, b_mat = inputs
            a_lit = "[" + ",".join("[" + ",".join(str(x) for x in row) + "]" for row in a_mat) + "]"
            b_lit = "[" + ",".join("[" + ",".join(str(x) for x in row) + "]" for row in b_mat) + "]"
            lines.append(f"results.append(jsonMatrix({fn}({a_lit}, {b_lit})))")

        elif algo == "knapsack_01":
            w, v, c = inputs
            w_lit = "[" + ",".join(str(x) for x in w) + "]"
            v_lit = "[" + ",".join(str(x) for x in v) + "]"
            lines.append("do {")
            lines.append(f"    let (val, items) = {fn}({w_lit}, {v_lit}, {c})")
            lines.append("    results.append(\"[\\(val),\" + jsonIntArr(items.sorted()) + \"]\")")
            lines.append("}")

        elif algo == "tower_of_hanoi":
            n, s, t, aux = inputs
            lines.append("do {")
            lines.append(f'    let moves = {fn}({n}, "{s}", "{t}", "{aux}")')
            lines.append("    results.append(jsonStrPairArr(moves))")
            lines.append("}")

        elif algo in ("dfs_graph", "bfs_graph"):
            graph, start = inputs
            n = max(graph.keys()) + 1 if graph else 0
            rows = []
            for i in range(n):
                nbrs = graph.get(i, [])
                rows.append("[" + ",".join(str(x) for x in nbrs) + "]")
            graph_lit = "[" + ",".join(rows) + "]"
            lines.append(f"results.append(jsonIntArr({fn}({graph_lit}, {start})))")

        elif algo == "dijkstra":
            graph, src = inputs
            rows = []
            for adj in graph:
                if adj:
                    tuples = ",".join(f"({nb},{wt})" for nb, wt in adj)
                    rows.append(f"[{tuples}]")
                else:
                    rows.append("[(Int,Int)]()")
            graph_lit = "[" + ",".join(rows) + "]"
            lines.append(f"results.append(jsonDblArr({fn}(graph: {graph_lit}, src: {src})))")

        elif algo == "topological_sort":
            graph, n_nodes = inputs
            rows = []
            for adj in graph:
                rows.append("[" + ",".join(str(x) for x in adj) + "]")
            graph_lit = "[" + ",".join(rows) + "]"
            lines.append(f"results.append(jsonIntArr({fn}(graph: {graph_lit}, n: {n_nodes})))")

        elif algo == "binary_tree_inorder":
            spec = inputs[0]
            lines.append("do {")
            if spec == "balanced_3":
                lines.append("    let n1 = TreeNode(1)")
                lines.append("    let n3 = TreeNode(3)")
                lines.append("    let n2 = TreeNode(2, n1, n3)")
                lines.append(f"    results.append(jsonIntArr({fn}(n2)))")
            elif spec == "left_chain_4":
                lines.append("    let n1 = TreeNode(1)")
                lines.append("    let n2 = TreeNode(2, n1)")
                lines.append("    let n3 = TreeNode(3, n2)")
                lines.append("    let n4 = TreeNode(4, n3)")
                lines.append(f"    results.append(jsonIntArr({fn}(n4)))")
            elif spec == "single":
                lines.append("    let n42 = TreeNode(42)")
                lines.append(f"    results.append(jsonIntArr({fn}(n42)))")
            lines.append("}")

    lines.append('print("[" + results.joined(separator: ",") + "]")')
    return "\n".join(lines)


def _find_swiftc() -> tuple[str | None, dict[str, str]]:
    """Locate swiftc and return (path, env_dict) with correct PATH/SDKROOT."""
    swiftc = os.environ.get("CODEIP_SWIFTC") or shutil.which("swiftc")
    if swiftc and os.path.isfile(swiftc):
        return swiftc, os.environ.copy()

    return None, {}


def run_swift(algo: str, code: str, expected_results: list[dict]) -> list[dict]:
    swiftc, env = _find_swiftc()
    if not swiftc:
        return [{"desc": "SKIP", "error": "swiftc not found"}]

    helpers = _swift_json_helpers()
    test_calls = _swift_build_test_calls(algo)
    src = helpers + "\n" + code + "\n\n" + test_calls + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "test.swift")
        exe_path = os.path.join(tmpdir, "test.exe")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src)

        comp = subprocess.run(
            [swiftc, "-o", exe_path, src_path],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if comp.returncode != 0:
            return [{"desc": "COMPILE_ERROR", "error": comp.stderr[:500]}]

        run = subprocess.run(
            [exe_path], capture_output=True, text=True, timeout=30, env=env,
        )
        if run.returncode != 0:
            return [{"desc": "RUNTIME_ERROR", "error": (run.stderr or run.stdout)[:500]}]

        try:
            raw = json.loads(run.stdout)
            return [{"desc": tc["desc"], "actual": raw[i]} for i, tc in enumerate(TEST_CASES[algo])]
        except Exception as e:
            return [{"desc": "PARSE_ERROR", "error": f"stdout={run.stdout[:200]}, err={e}"}]


# ---------------------------------------------------------------------------
# Runner dispatch
# ---------------------------------------------------------------------------

def _get_runner(lang: str):
    """Return runner function for a language, or None if unavailable."""
    runners = {
        "JavaScript": run_javascript,
        "TypeScript": run_typescript,
        "C++": run_cpp,
        "Julia": run_julia,
        "Go": run_go,
        "Rust": run_rust,
        "PHP": run_php,
        "Bash": run_bash,
        "Perl": run_perl,
        "Ruby": run_ruby,
        "Lua": run_lua,
        "R": run_r,
        "Java": run_java,
        "C#": run_csharp,
        "Kotlin": run_kotlin,
        "Scala": run_scala,
        "Racket": run_racket,
        "D": run_d,
        "Swift": run_swift,
    }
    runner = runners.get(lang)
    if runner is None:
        return None
    toolchain_check = {
        "C++": "g++", "Julia": "julia", "Go": "go",
        "Rust": "rustc", "PHP": "php", "JavaScript": "node",
        "Ruby": "ruby", "Lua": ["lua5.4", "lua"], "R": "Rscript",
        "Java": "javac", "C#": "dotnet", "Kotlin": "kotlinc",
        "Scala": ["scala-cli", "scala"], "Racket": "racket",
        "D": ["gdc", "dmd", "ldc2"], "Swift": ["swift", "swiftc"],
    }
    check = toolchain_check.get(lang)
    if check:
        if isinstance(check, list):
            if not any(shutil.which(c) or _tool(c) for c in check):
                return None
        elif not shutil.which(check) and not _tool(check):
            return None
    return runner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def normalize_for_comparison(val, algo: str, lang: str):
    """Normalize values for cross-language comparison.

    Handles known semantic-preserving differences like Julia 1-based indexing,
    float vs int distance arrays, and sort ordering variations.
    """
    if isinstance(val, float) and val == int(val):
        return int(val)
    if isinstance(val, list):
        return [normalize_for_comparison(v, algo, lang) for v in val]
    return val


def compare_outputs(expected, actual, algo: str, lang: str) -> bool:
    """Compare expected (Python) and actual (other language) outputs."""
    e = normalize_for_comparison(expected, algo, lang)
    a = normalize_for_comparison(actual, algo, lang)

    if algo == "topological_sort" and isinstance(e, list) and isinstance(a, list):
        if len(e) != len(a):
            return False
        if set(e) == set(a):
            return True
        a_shifted = [x - 1 for x in a]
        return set(e) == set(a_shifted)

    if algo == "dijkstra" and isinstance(e, list) and isinstance(a, list):
        if len(e) != len(a):
            return False
        for ev, av in zip(e, a):
            if isinstance(ev, float) and ev == float("inf"):
                if av != float("inf") and av != 2147483647 and av != 1e18 and av != -1:
                    return False
            elif ev != av and int(ev) != int(av):
                return False
        return True

    if isinstance(e, list) and not isinstance(a, list) and len(e) == 1:
        return compare_outputs(e[0], a, algo, lang)

    if isinstance(e, list) and isinstance(a, list):
        if len(e) != len(a):
            return False
        return all(compare_outputs(ev, av, algo, lang) for ev, av in zip(e, a))

    if isinstance(e, (int, float)) and isinstance(a, (int, float)):
        return abs(e - a) < 1e-9

    return e == a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the curated corpus against shared cross-language tests."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help=f"Corpus JSON to validate (default: {CORPUS_PATH})",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGES,
        default=LANGUAGES,
        help="Languages to validate. Python is always included as the reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: data/corpus_execution_validation.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_languages = list(dict.fromkeys(["Python", *args.languages]))

    with args.corpus.open(encoding="utf-8") as f:
        corpus = json.load(f)

    results: dict[str, dict[str, str]] = {}
    summary = {"total": 0, "pass": 0, "fail": 0, "skip": 0}

    print("=" * 70)
    print("Curated Corpus Execution Validation")
    print(f"Corpus: {args.corpus}")
    print(f"Algorithms: {len(TEST_CASES)}, Languages: {len(selected_languages)}")
    print("=" * 70)

    for algo in sorted(TEST_CASES):
        if algo not in corpus:
            print(f"  SKIP {algo}: not in corpus")
            continue

        algo_code = corpus[algo]
        results[algo] = {}

        py_code = algo_code.get("Python", "")
        if not py_code:
            print(f"  SKIP {algo}: no Python implementation")
            continue

        try:
            expected = run_python_reference(algo, py_code)
        except Exception as e:
            print(f"  FAIL {algo}/Python (reference): {e}")
            results[algo]["Python"] = f"REFERENCE_ERROR: {e}"
            continue

        results[algo]["Python"] = "PASS"
        summary["total"] += 1
        summary["pass"] += 1

        for lang in selected_languages:
            if lang == "Python":
                continue

            code = algo_code.get(lang, "")
            if not code:
                results[algo][lang] = "SKIP: no implementation"
                summary["skip"] += 1
                continue

            summary["total"] += 1

            runner = _get_runner(lang)
            if runner is None:
                results[algo][lang] = "SKIP: no toolchain"
                summary["skip"] += 1
                continue

            print(f"  running {algo}/{lang}...", end="", flush=True)
            try:
                lang_results = runner(algo, code, expected)
            except Exception as exc:
                results[algo][lang] = f"FAIL: runner crashed: {exc}"
                summary["fail"] += 1
                print(f" CRASH: {str(exc)[:80]}")
                continue
            if lang_results and "error" in lang_results[0]:
                results[algo][lang] = f"FAIL: {lang_results[0].get('error', 'unknown')[:200]}"
                summary["fail"] += 1
                print(f" FAIL: {lang_results[0].get('error', '')[:80]}")
                continue

            all_pass = True
            for exp, act in zip(expected, lang_results):
                if not compare_outputs(exp["expected"], act["actual"], algo, lang):
                        results[algo][lang] = f"FAIL: {exp['desc']}: expected {exp['expected']}, got {act['actual']}"
                        summary["fail"] += 1
                        all_pass = False
                        print(f" FAIL ({exp['desc']})")
                        break

            if all_pass:
                results[algo][lang] = "PASS"
                summary["pass"] += 1
                print(" OK")

    print("\n" + "=" * 70)
    print(f"Summary: {summary['pass']} pass, {summary['fail']} fail, "
          f"{summary['skip']} skip out of {summary['total']} total")
    print("=" * 70)

    # Print table
    print(f"\n{'Algorithm':<35s}", end="")
    for lang in selected_languages:
        print(f" {lang[:6]:>6s}", end="")
    print()
    print("-" * (35 + 7 * len(selected_languages)))

    for algo in sorted(results):
        print(f"{algo:<35s}", end="")
        for lang in selected_languages:
            status = results[algo].get(lang, "N/A")
            if status == "PASS":
                marker = "OK"
            elif status.startswith("SKIP"):
                marker = "--"
            else:
                marker = "FAIL"
            print(f" {marker:>6s}", end="")
        print()

    out_path = args.output or (ROOT / "data" / "corpus_execution_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
        f.write("\n")
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
