import json
from pathlib import Path

MODEL_NAME = "./models/deepseek-coder-6.7b-instruct"
REFERENCE_LANG = "Python"
TARGET_LANGS = {
    REFERENCE_LANG, "Java", "JavaScript", "C++", "Lua", "R", "Rust", "Swift",
    "Go", "Ruby", "C#", "Kotlin", "PHP", "Julia", "TypeScript",
    "Scala", "D", "Racket", "Perl", "Bash",
}
MAX_TASKS = 50
MAX_SEQ_LEN = 2048
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

LANG_ALIASES = {
    "Cpp": "C++", "C Plus Plus": "C++", "C plus plus": "C++",
    "C Sharp": "C#", "C sharp": "C#", "Csharp": "C#",
    "Javascript": "JavaScript",
    "Python 3": "Python",
    "Golang": "Go",
}

MULTI_COL_TO_LANG = {
    "humaneval-python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "cpp": "C++",
    "lua": "Lua",
    "r": "R",
    "rust": "Rust",
    "swift": "Swift",
    "php": "PHP",
    "julia": "Julia",
    "d": "D",
    "racket": "Racket",
    "go": "Go",
    "csharp": "C#",
    "ruby": "Ruby",
    "typescript": "TypeScript",
}

HIGH_RESOURCE_LANGS = {
    "Python", "Java", "JavaScript", "C++", "C#", "TypeScript",
    "Go", "PHP", "Ruby", "Rust", "Swift",
}

LOW_RESOURCE_LANGS = {
    "Lua", "R", "Julia", "D", "Racket", "Kotlin", "Perl",
    "Scala", "Bash", "Shell", "Haskell", "Erlang", "Elixir",
    "Fortran", "Dart", "Common Lisp", "COBOL", "CoffeeScript",
}

# For report-level typed vs non-typed comparisons we use a simple binary split:
# any benchmark language not listed here is treated as non-typed.
TYPED_LANGS = {
    "C",
    "C++",
    "C#",
    "Java",
    "TypeScript",
    "Go",
    "Rust",
    "Swift",
    "D",
    "Kotlin",
    "Scala",
    "Haskell",
    "Fortran",
    "Dart",
    "COBOL",
}

MODEL_BENCHMARK_MAP = {
    "./models/deepseek-coder-6.7b-instruct": "DeepSeek-Coder-7b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek-Coder-7b-instruct",
    "./models/codellama-7b-instruct": "CodeLlama-7b-Instruct",
    "codellama/CodeLlama-7b-Instruct-hf": "CodeLlama-7b-Instruct",
    "./models/stable-code-3b": "Stable-code-3b",
    "stabilityai/stable-code-3b": "Stable-code-3b",
    "./models/starcoderbase-1b": "StarCoderBase-1.1B",
    "bigcode/starcoderbase-1b": "StarCoderBase-1.1B",
    "./models/starcoderbase-3b": "StarCoderBase-3B",
    "bigcode/starcoderbase-3b": "StarCoderBase-3B",
    "./models/starcoder2-3b": "StarCoder2-3B",
    "bigcode/starcoder2-3b": "StarCoder2-3B",
    "./models/qwen2.5-coder-0.5b": "Qwen2.5-Coder-0.5B",
    "Qwen/Qwen2.5-Coder-0.5B": "Qwen2.5-Coder-0.5B",
    "./models/qwen2.5-coder-0.5b-instruct": "Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": "Qwen2.5-Coder-0.5B-Instruct",
    "./models/deepseek-coder-1.3b-instruct": "Deepseek-Coder-1.3B",
    "deepseek-ai/deepseek-coder-1.3b-instruct": "Deepseek-Coder-1.3B",
    "./models/deepseek-coder-1.3b-base": "DeepSeek-Coder-1.3B-Base",
    "deepseek-ai/deepseek-coder-1.3b-base": "DeepSeek-Coder-1.3B-Base",
    "./models/qwen2.5-coder-1.5b": "Qwen2.5-Coder-1.5B",
    "Qwen/Qwen2.5-Coder-1.5B": "Qwen2.5-Coder-1.5B",
    "./models/qwen2.5-coder-1.5b-instruct": "Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "Qwen2.5-Coder-1.5B-Instruct",
    "./models/opencoder-1.5b-instruct": "OpenCoder-1.5B",
    "infly/OpenCoder-1.5B-Instruct": "OpenCoder-1.5B",
    "./models/opencoder-8b-instruct": "OpenCoder-8B",
    "infly/OpenCoder-8B-Instruct": "OpenCoder-8B",
    "./models/yi-coder-1.5b-chat": "Yi-Coder-1.5B",
    "01-ai/Yi-Coder-1.5B-Chat": "Yi-Coder-1.5B",
    "./models/codegen-6b-multi": "CodeGen-multi-6B",
    "Salesforce/codegen-6B-multi": "CodeGen-multi-6B",
    "./models/codegen-16b-multi": "CodeGen-16B-Multi",
    "Salesforce/codegen-16B-multi": "CodeGen-16B-Multi",
    "./models/codegen25-7b-mono": "CodeGen25-7B-Mono",
    "Salesforce/codegen25-7b-mono_P": "CodeGen25-7B-Mono",
    "./models/codegen25-7b-multi": "CodeGen25-7B-Multi",
    "Salesforce/codegen25-7b-multi_P": "CodeGen25-7B-Multi",
    "./models/phi-1_5": "phi-1_5",
    "microsoft/phi-1_5": "phi-1_5",
    "./models/phi-1": "phi-1",
    "microsoft/phi-1": "phi-1",
    "./models/phi-3-medium-4k-instruct": "Phi-3-Medium-4K-Instruct",
    "microsoft/Phi-3-medium-4k-instruct": "Phi-3-Medium-4K-Instruct",
    "./models/codellama-13b-hf": "CodeLlama-13B-Base",
    "codellama/CodeLlama-13b-hf": "CodeLlama-13B-Base",
    "./models/codellama-13b-instruct-hf": "CodeLlama-13B-Instruct",
    "codellama/CodeLlama-13b-Instruct-hf": "CodeLlama-13B-Instruct",
    "./models/codellama-13b-python-hf": "CodeLlama-13B-Python",
    "codellama/CodeLlama-13b-Python-hf": "CodeLlama-13B-Python",
    "./models/codellama-34b-hf": "CodeLlama-34B-Base",
    "codellama/CodeLlama-34b-hf": "CodeLlama-34B-Base",
    "./models/deepseek-coder-1b-base": "DeepSeek-Coder-1B-Base",
    "deepseek-ai/deepseek-coder-1b-base": "DeepSeek-Coder-1B-Base",
    "./models/deepseek-coder-v2-lite-instruct": "DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": "DeepSeek-Coder-V2-Lite-Instruct",
    "./models/yi-coder-9b-chat": "Yi-Coder-9B-Chat",
    "01-ai/Yi-Coder-9B-Chat": "Yi-Coder-9B-Chat",
    "./models/codefuse-codegeex2-6b": "CodeGeeX2-6B",
    "codefuse-ai/CodeFuse-CodeGeeX2-6B": "CodeGeeX2-6B",
    "./models/granite-20b-code-instruct-8k": "Granite-Coder-20B",
    "ibm-granite/granite-20b-code-instruct-8k": "Granite-Coder-20B",
    "./models/granite-34b-code-instruct-8k": "Granite-Coder-34B",
    "ibm-granite/granite-34b-code-instruct-8k": "Granite-Coder-34B",
    "./models/llama-3.1-8b": "Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
    "./models/qwen2.5-coder-3b": "Qwen2.5-Coder-3B",
    "Qwen/Qwen2.5-Coder-3B": "Qwen2.5-Coder-3B",
    "./models/qwen2.5-coder-3b-instruct": "Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "Qwen2.5-Coder-3B-Instruct",
    "./models/granite-coder-3b-base-128k": "Granite-Coder-3B",
    "ibm-granite/granite-3b-code-base-128k": "Granite-Coder-3B",
    "./models/granite-coder-8b-code-instruct-128k": "Granite-Coder-8B",
    "ibm-granite/granite-8b-code-instruct-128k": "Granite-Coder-8B",
    "./models/qwen3-0.6b": "Qwen3-0.6B",
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "./models/qwen3-1.7b": "Qwen3-1.7B",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "./models/qwen3-4b": "Qwen3-4B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "./models/qwen3-8b": "Qwen3-8B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "./models/deepseek-r1-distill-qwen-14b": "DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B",
    "./models/seed-coder-8b-instruct": "Seed-Coder-8B-Instruct",
    "ByteDance-Seed/Seed-Coder-8B-Instruct": "Seed-Coder-8B-Instruct",
    "./models/devstral-small-2505": "Devstral-Small-2505",
    "mistralai/Devstral-Small-2505": "Devstral-Small-2505",
    "./models/qwen3-30b-a3b": "Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "./models/qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B-Instruct-2507",
    "./models/openreasoning-nemotron-32b": "OpenReasoning-Nemotron-32B",
    "nvidia/OpenReasoning-Nemotron-32B": "OpenReasoning-Nemotron-32B",
    "./models/opencodereasoning-nemotron-1.1-32b": "OpenCodeReasoning-Nemotron-1.1-32B",
    "nvidia/OpenCodeReasoning-Nemotron-1.1-32B": "OpenCodeReasoning-Nemotron-1.1-32B",
    "./models/qwen2.5-coder-7b": "Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-7B": "Qwen2.5-Coder-7B",
    "./models/qwen2.5-coder-7b-instruct": "Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B-Instruct",
    "./models/deepseek-coder-6.7b-base": "DeepSeek-Coder-7b-base",
    "deepseek-ai/deepseek-coder-6.7b-base": "DeepSeek-Coder-7b-base",
    "./models/magicoder-s-ds-6.7b": "Magicoder-S-DS-6.7B",
    "ise-uiuc/Magicoder-S-DS-6.7B": "Magicoder-S-DS-6.7B",
    "./models/codellama-7b-hf": "CodeLlama-7b",
    "codellama/CodeLlama-7b-hf": "CodeLlama-7b",
    "./models/codeqwen1.5-7b-chat": "CodeQwen1.5-7B-Chat",
    "Qwen/CodeQwen1.5-7B-Chat": "CodeQwen1.5-7B-Chat",
    "./models/qwen2.5-coder-14b-instruct": "Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B-Instruct",
    "./models/qwen2.5-coder-14b": "Qwen2.5-Coder-14B",
    "Qwen/Qwen2.5-Coder-14B": "Qwen2.5-Coder-14B",
    "./models/qwen2.5-coder-32b-instruct": "Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "./models/qwen2.5-coder-32b": "Qwen2.5-Coder-32B",
    "Qwen/Qwen2.5-Coder-32B": "Qwen2.5-Coder-32B",
    "./models/deepseek-coder-33b-instruct": "DeepSeek-Coder-33B-Instruct",
    "deepseek-ai/deepseek-coder-33b-instruct": "DeepSeek-Coder-33B-Instruct",
    "./models/deepseek-coder-33b-base": "DeepSeek-Coder-33B-Base",
    "deepseek-ai/deepseek-coder-33b-base": "DeepSeek-Coder-33B-Base",
    "./models/codefuse-deepseek-33b": "CodeFuse-DeepSeek-33B",
    "codefuse-ai/CodeFuse-DeepSeek-33B": "CodeFuse-DeepSeek-33B",
    "./models/deepseek-r1-distill-qwen-32b": "DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B",
    "./models/qwen3-30b-a3b-thinking-2507": "Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-A3B-Thinking-2507",
    "./models/qwen3-coder-30b-a3b-instruct": "Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen3-Coder-30B-A3B-Instruct",
    "./models/qwen3-32b": "Qwen3-32B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "./models/qwen3-14b": "Qwen3-14B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "./models/codestral-22b-v0.1": "Codestral-22B-v0.1",
    "mistralai/Codestral-22B-v0.1": "Codestral-22B-v0.1",
    "./models/octocoder": "OctoCoder-15B",
    "bigcode/octocoder": "OctoCoder-15B",
    "./models/octogeex": "OctoGeeX-7B",
    "bigcode/octogeex": "OctoGeeX-7B",
    "./models/starcoder-15b": "StarCoder-15B",
    "bigcode/starcoder": "StarCoder-15B",
    "./models/starcoderbase-7b": "StarCoderBase-7B",
    "bigcode/starcoderbase-7b": "StarCoderBase-7B",
    "./models/starcoderbase-15b": "StarCoderBase-15B",
    "bigcode/starcoderbase": "StarCoderBase-15B",
    "./models/starcoder2-15b": "StarCoder2-15B",
    "bigcode/starcoder2-15b": "StarCoder2-15B",
    "./models/starcoder2-15b-instruct-v0.1": "StarCoder2-15B-Instruct-v0.1",
    "bigcode/starcoder2-15b-instruct-v0.1": "StarCoder2-15B-Instruct-v0.1",
    "./models/starcoder2-7b": "StarCoder2-7B",
    "bigcode/starcoder2-7b": "StarCoder2-7B",
    "./models/meta-llama-3-8b-instruct": "Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct",
    "./models/meta-llama-3.1-8b-instruct": "Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct",
    "./models/gpt-oss-20b": "Gpt-oss-20B",
    "openai/gpt-oss-20b": "Gpt-oss-20B",
    "./models/phind-codellama-34b-python-v1": "Phind-CodeLlama-34B-Python-v1",
    "Phind/Phind-CodeLlama-34B-Python-v1": "Phind-CodeLlama-34B-Python-v1",
    "./models/phind-codellama-34b-v1": "Phind-CodeLlama-34B-v1",
    "Phind/Phind-CodeLlama-34B-v1": "Phind-CodeLlama-34B-v1",
    "./models/phind-codellama-34b-v2": "Phind-CodeLlama-34B-v2",
    "Phind/Phind-CodeLlama-34B-v2": "Phind-CodeLlama-34B-v2",
    "./models/wizardcoder-15b": "WizardCoder-15B",
    "WizardLMTeam/WizardCoder-15B-V1.0": "WizardCoder-15B",
    "./models/wizardcoder-33b-v1.1": "WizardCoder-33B-V1.1",
    "WizardLMTeam/WizardCoder-33B-V1.1": "WizardCoder-33B-V1.1",
    "./models/wizardcoder-python-13b": "WizardCoder-Python-13B",
    "WizardLMTeam/WizardCoder-Python-13B-V1.0": "WizardCoder-Python-13B",
    "./models/wizardcoder-python-34b": "WizardCoder-Python-34B",
    "WizardLMTeam/WizardCoder-Python-34B-V1.0": "WizardCoder-Python-34B",
    "./models/codellama-34b-instruct-hf": "CodeLlama-34B-Instruct",
    "codellama/CodeLlama-34b-Instruct-hf": "CodeLlama-34B-Instruct",
    "./models/codellama-34b-python-hf": "CodeLlama-34B-Python",
    "codellama/CodeLlama-34b-Python-hf": "CodeLlama-34B-Python",
}

MODEL_REGISTRY = {
    "./models/deepseek-coder-6.7b-instruct": {
        "display_name": "DeepSeek-Coder-6.7B-Instruct",
        "family": "DeepSeek", "size_b": 6.7, "variant": "instruct",
    },
    "./models/deepseek-coder-6.7b-base": {
        "display_name": "DeepSeek-Coder-6.7B-Base",
        "family": "DeepSeek", "size_b": 6.7, "variant": "base",
    },
    "./models/magicoder-s-ds-6.7b": {
        "display_name": "Magicoder-S-DS-6.7B",
        "family": "Magicoder", "size_b": 6.7, "variant": "base",
    },
    "./models/codellama-7b-instruct": {
        "display_name": "CodeLlama-7B-Instruct",
        "family": "CodeLlama", "size_b": 7.0, "variant": "instruct",
    },
    "./models/codellama-7b-hf": {
        "display_name": "CodeLlama-7B-Base",
        "family": "CodeLlama", "size_b": 7.0, "variant": "base",
    },
    "./models/codellama-13b-hf": {
        "display_name": "CodeLlama-13B-Base",
        "family": "CodeLlama", "size_b": 13.0, "variant": "base",
    },
    "./models/codellama-13b-instruct-hf": {
        "display_name": "CodeLlama-13B-Instruct",
        "family": "CodeLlama", "size_b": 13.0, "variant": "instruct",
    },
    "./models/codellama-13b-python-hf": {
        "display_name": "CodeLlama-13B-Python",
        "family": "CodeLlama", "size_b": 13.0, "variant": "python",
    },
    "./models/stable-code-3b": {
        "display_name": "Stable-Code-3B",
        "family": "StableCode", "size_b": 3.0, "variant": "base",
    },
    "./models/starcoderbase-1b": {
        "display_name": "StarCoderBase-1.1B",
        "family": "StarCoderBase", "size_b": 1.1, "variant": "base",
    },
    "./models/starcoderbase-3b": {
        "display_name": "StarCoderBase-3B",
        "family": "StarCoderBase", "size_b": 3.0, "variant": "base",
    },
    "./models/starcoder2-3b": {
        "display_name": "StarCoder2-3B",
        "family": "StarCoder2", "size_b": 3.0, "variant": "base",
    },
    "./models/qwen2.5-coder-0.5b": {
        "display_name": "Qwen2.5-Coder-0.5B",
        "family": "Qwen2.5-Coder", "size_b": 0.5, "variant": "base",
    },
    "./models/qwen2.5-coder-0.5b-instruct": {
        "display_name": "Qwen2.5-Coder-0.5B-Instruct",
        "family": "Qwen2.5-Coder", "size_b": 0.5, "variant": "instruct",
    },
    "./models/qwen3-0.6b": {
        "display_name": "Qwen3-0.6B",
        "family": "Qwen3", "size_b": 0.6, "variant": "base",
    },
    "./models/deepseek-coder-1.3b-instruct": {
        "display_name": "DeepSeek-Coder-1.3B-Instruct",
        "family": "DeepSeek", "size_b": 1.3, "variant": "instruct",
    },
    "./models/deepseek-coder-1.3b-base": {
        "display_name": "DeepSeek-Coder-1.3B-Base",
        "family": "DeepSeek", "size_b": 1.3, "variant": "base",
    },
    "./models/deepseek-coder-1b-base": {
        "display_name": "DeepSeek-Coder-1B-Base",
        "family": "DeepSeek", "size_b": 1.0, "variant": "base",
    },
    "./models/deepseek-coder-v2-lite-instruct": {
        "display_name": "DeepSeek-Coder-V2-Lite-Instruct",
        "family": "DeepSeek-Coder-V2", "size_b": 16.0, "variant": "instruct",
    },
    "./models/qwen2.5-coder-1.5b": {
        "display_name": "Qwen2.5-Coder-1.5B",
        "family": "Qwen2.5-Coder", "size_b": 1.5, "variant": "base",
    },
    "./models/qwen2.5-coder-1.5b-instruct": {
        "display_name": "Qwen2.5-Coder-1.5B-Instruct",
        "family": "Qwen2.5-Coder", "size_b": 1.5, "variant": "instruct",
    },
    "./models/opencoder-1.5b-instruct": {
        "display_name": "OpenCoder-1.5B-Instruct",
        "family": "OpenCoder", "size_b": 1.5, "variant": "instruct",
    },
    "./models/opencoder-8b-instruct": {
        "display_name": "OpenCoder-8B-Instruct",
        "family": "OpenCoder", "size_b": 8.0, "variant": "instruct",
    },
    "./models/yi-coder-1.5b-chat": {
        "display_name": "Yi-Coder-1.5B-Chat",
        "family": "Yi-Coder", "size_b": 1.5, "variant": "chat",
    },
    "./models/yi-coder-9b-chat": {
        "display_name": "Yi-Coder-9B-Chat",
        "family": "Yi-Coder", "size_b": 9.0, "variant": "chat",
    },
    "./models/codefuse-codegeex2-6b": {
        "display_name": "CodeGeeX2-6B",
        "family": "CodeGeeX2", "size_b": 6.0, "variant": "base",
    },
    "./models/codegen-6b-multi": {
        "display_name": "CodeGen-multi-6B",
        "family": "CodeGen", "size_b": 6.0, "variant": "multi",
    },
    "./models/codegen-16b-multi": {
        "display_name": "CodeGen-16B-Multi",
        "family": "CodeGen", "size_b": 16.0, "variant": "multi",
    },
    "./models/codegen25-7b-mono": {
        "display_name": "CodeGen25-7B-Mono",
        "family": "CodeGen25", "size_b": 7.0, "variant": "mono",
    },
    "./models/codegen25-7b-multi": {
        "display_name": "CodeGen25-7B-Multi",
        "family": "CodeGen25", "size_b": 7.0, "variant": "multi",
    },
    "./models/phi-1_5": {
        "display_name": "Phi-1.5",
        "family": "Phi", "size_b": 1.3, "variant": "base",
    },
    "./models/phi-1": {
        "display_name": "Phi-1",
        "family": "Phi", "size_b": 1.3, "variant": "base",
    },
    "./models/phi-3-medium-4k-instruct": {
        "display_name": "Phi-3-Medium-4K-Instruct",
        "family": "Phi-3", "size_b": 14.0, "variant": "instruct",
    },
    "./models/qwen3-1.7b": {
        "display_name": "Qwen3-1.7B",
        "family": "Qwen3", "size_b": 1.7, "variant": "base",
    },
    "./models/qwen2.5-coder-3b": {
        "display_name": "Qwen2.5-Coder-3B",
        "family": "Qwen2.5-Coder", "size_b": 3.0, "variant": "base",
    },
    "./models/qwen2.5-coder-3b-instruct": {
        "display_name": "Qwen2.5-Coder-3B-Instruct",
        "family": "Qwen2.5-Coder", "size_b": 3.0, "variant": "instruct",
    },
    "./models/granite-coder-3b-base-128k": {
        "display_name": "Granite-Coder-3B-Base-128K",
        "family": "Granite", "size_b": 3.0, "variant": "base",
    },
    "./models/granite-coder-8b-code-instruct-128k": {
        "display_name": "Granite-Coder-8B-Instruct-128K",
        "family": "Granite", "size_b": 8.0, "variant": "instruct",
    },
    "./models/granite-20b-code-instruct-8k": {
        "display_name": "Granite-Coder-20B",
        "family": "Granite", "size_b": 20.0, "variant": "instruct",
    },
    "./models/granite-34b-code-instruct-8k": {
        "display_name": "Granite-Coder-34B",
        "family": "Granite", "size_b": 34.0, "variant": "instruct",
    },
    "./models/qwen3-4b": {
        "display_name": "Qwen3-4B",
        "family": "Qwen3", "size_b": 4.0, "variant": "base",
    },
    "./models/qwen3-8b": {
        "display_name": "Qwen3-8B",
        "family": "Qwen3", "size_b": 8.0, "variant": "base",
    },
    "./models/deepseek-r1-distill-qwen-14b": {
        "display_name": "DeepSeek-R1-Distill-Qwen-14B",
        "family": "DeepSeek-R1-Distill", "size_b": 14.0, "variant": "distill",
    },
    "./models/seed-coder-8b-instruct": {
        "display_name": "Seed-Coder-8B-Instruct",
        "family": "Seed-Coder", "size_b": 8.0, "variant": "instruct",
    },
    "./models/devstral-small-2505": {
        "display_name": "Devstral-Small-2505",
        "family": "Devstral", "size_b": 24.0, "variant": "instruct",
    },
    "./models/qwen3-30b-a3b": {
        "display_name": "Qwen3-30B-A3B",
        "family": "Qwen3", "size_b": 30.0, "variant": "base",
    },
    "./models/qwen3-30b-a3b-instruct-2507": {
        "display_name": "Qwen3-30B-A3B-Instruct-2507",
        "family": "Qwen3", "size_b": 30.0, "variant": "instruct",
    },
    "./models/qwen2.5-coder-7b": {
        "display_name": "Qwen2.5-Coder-7B",
        "family": "Qwen2.5-Coder", "size_b": 7.0, "variant": "base",
    },
    "./models/qwen2.5-coder-7b-instruct": {
        "display_name": "Qwen2.5-Coder-7B-Instruct",
        "family": "Qwen2.5-Coder", "size_b": 7.0, "variant": "instruct",
    },
    "./models/codeqwen1.5-7b-chat": {
        "display_name": "CodeQwen1.5-7B-Chat",
        "family": "CodeQwen", "size_b": 7.0, "variant": "chat",
    },
    "./models/qwen2.5-coder-14b-instruct": {
        "display_name": "Qwen2.5-Coder-14B-Instruct",
        "family": "Qwen2.5-Coder", "size_b": 14.0, "variant": "instruct",
    },
    "./models/qwen2.5-coder-14b": {
        "display_name": "Qwen2.5-Coder-14B",
        "family": "Qwen2.5-Coder", "size_b": 14.0, "variant": "base",
    },
    "./models/qwen2.5-coder-32b-instruct": {
        "display_name": "Qwen2.5-Coder-32B-Instruct",
        "family": "Qwen2.5-Coder", "size_b": 32.0, "variant": "instruct",
    },
    "./models/qwen2.5-coder-32b": {
        "display_name": "Qwen2.5-Coder-32B",
        "family": "Qwen2.5-Coder", "size_b": 32.0, "variant": "base",
    },
    "./models/deepseek-coder-33b-instruct": {
        "display_name": "DeepSeek-Coder-33B-Instruct",
        "family": "DeepSeek", "size_b": 33.0, "variant": "instruct",
    },
    "./models/deepseek-coder-33b-base": {
        "display_name": "DeepSeek-Coder-33B-Base",
        "family": "DeepSeek", "size_b": 33.0, "variant": "base",
    },
    "./models/codefuse-deepseek-33b": {
        "display_name": "CodeFuse-DeepSeek-33B",
        "family": "CodeFuse-DeepSeek", "size_b": 33.0, "variant": "base",
    },
    "./models/deepseek-r1-distill-qwen-32b": {
        "display_name": "DeepSeek-R1-Distill-Qwen-32B",
        "family": "DeepSeek-R1-Distill", "size_b": 32.0, "variant": "distill",
    },
    "./models/openreasoning-nemotron-32b": {
        "display_name": "OpenReasoning-Nemotron-32B",
        "family": "OpenReasoning-Nemotron", "size_b": 32.0, "variant": "reasoning",
    },
    "./models/opencodereasoning-nemotron-1.1-32b": {
        "display_name": "OpenCodeReasoning-Nemotron-1.1-32B",
        "family": "OpenCodeReasoning-Nemotron", "size_b": 32.0, "variant": "reasoning",
    },
    "./models/qwen3-30b-a3b-thinking-2507": {
        "display_name": "Qwen3-30B-A3B-Thinking-2507",
        "family": "Qwen3", "size_b": 30.0, "variant": "thinking",
    },
    "./models/qwen3-coder-30b-a3b-instruct": {
        "display_name": "Qwen3-Coder-30B-A3B-Instruct",
        "family": "Qwen3-Coder", "size_b": 30.0, "variant": "instruct",
    },
    "./models/qwen3-32b": {
        "display_name": "Qwen3-32B",
        "family": "Qwen3", "size_b": 32.0, "variant": "base",
    },
    "./models/qwen3-14b": {
        "display_name": "Qwen3-14B",
        "family": "Qwen3", "size_b": 14.0, "variant": "base",
    },
    "./models/codestral-22b-v0.1": {
        "display_name": "Codestral-22B-v0.1",
        "family": "Codestral", "size_b": 22.0, "variant": "base",
    },
    "./models/octocoder": {
        "display_name": "OctoCoder-15B",
        "family": "OctoCoder", "size_b": 15.0, "variant": "base",
    },
    "./models/octogeex": {
        "display_name": "OctoGeeX-7B",
        "family": "OctoGeeX", "size_b": 7.0, "variant": "base",
    },
    "./models/starcoder-15b": {
        "display_name": "StarCoder-15B",
        "family": "StarCoder", "size_b": 15.0, "variant": "base",
    },
    "./models/starcoderbase-7b": {
        "display_name": "StarCoderBase-7B",
        "family": "StarCoderBase", "size_b": 7.0, "variant": "base",
    },
    "./models/starcoderbase-15b": {
        "display_name": "StarCoderBase-15B",
        "family": "StarCoderBase", "size_b": 15.0, "variant": "base",
    },
    "./models/starcoder2-15b": {
        "display_name": "StarCoder2-15B",
        "family": "StarCoder2", "size_b": 15.0, "variant": "base",
    },
    "./models/starcoder2-15b-instruct-v0.1": {
        "display_name": "StarCoder2-15B-Instruct-v0.1",
        "family": "StarCoder2", "size_b": 15.0, "variant": "instruct",
    },
    "./models/starcoder2-7b": {
        "display_name": "StarCoder2-7B",
        "family": "StarCoder2", "size_b": 7.0, "variant": "base",
    },
    "./models/meta-llama-3-8b-instruct": {
        "display_name": "Meta-Llama-3-8B-Instruct",
        "family": "Llama-3", "size_b": 8.0, "variant": "instruct",
    },
    "./models/meta-llama-3.1-8b-instruct": {
        "display_name": "Meta-Llama-3.1-8B-Instruct",
        "family": "Llama-3.1", "size_b": 8.0, "variant": "instruct",
    },
    "./models/gpt-oss-20b": {
        "display_name": "Gpt-oss-20B",
        "family": "Gpt-oss", "size_b": 20.0, "variant": "base",
    },
    "./models/llama-3.1-8b": {
        "display_name": "Llama-3.1-8B",
        "family": "Llama-3.1", "size_b": 8.0, "variant": "base",
    },
    "./models/phind-codellama-34b-python-v1": {
        "display_name": "Phind-CodeLlama-34B-Python-v1",
        "family": "Phind-CodeLlama", "size_b": 34.0, "variant": "python",
    },
    "./models/phind-codellama-34b-v1": {
        "display_name": "Phind-CodeLlama-34B-v1",
        "family": "Phind-CodeLlama", "size_b": 34.0, "variant": "base",
    },
    "./models/phind-codellama-34b-v2": {
        "display_name": "Phind-CodeLlama-34B-v2",
        "family": "Phind-CodeLlama", "size_b": 34.0, "variant": "base",
    },
    "./models/wizardcoder-15b": {
        "display_name": "WizardCoder-15B",
        "family": "WizardCoder", "size_b": 15.0, "variant": "base",
    },
    "./models/wizardcoder-33b-v1.1": {
        "display_name": "WizardCoder-33B-V1.1",
        "family": "WizardCoder", "size_b": 33.0, "variant": "instruct",
    },
    "./models/wizardcoder-python-13b": {
        "display_name": "WizardCoder-Python-13B",
        "family": "WizardCoder", "size_b": 13.0, "variant": "python",
    },
    "./models/wizardcoder-python-34b": {
        "display_name": "WizardCoder-Python-34B",
        "family": "WizardCoder", "size_b": 34.0, "variant": "python",
    },
    "./models/codellama-34b-hf": {
        "display_name": "CodeLlama-34B-Base",
        "family": "CodeLlama", "size_b": 34.0, "variant": "base",
    },
    "./models/codellama-34b-instruct-hf": {
        "display_name": "CodeLlama-34B-Instruct",
        "family": "CodeLlama", "size_b": 34.0, "variant": "instruct",
    },
    "./models/codellama-34b-python-hf": {
        "display_name": "CodeLlama-34B-Python",
        "family": "CodeLlama", "size_b": 34.0, "variant": "python",
    },
}


def discover_models(results_dir: Path | None = None) -> dict[str, dict]:
    """Auto-discover models from results/metrics_*.json files.

    Returns {display_name: {"metrics_file": str, "model_path": str, "meta": dict}}
    ordered by filename.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    models: dict[str, dict] = {}
    for path in sorted(results_dir.glob("metrics_*.json")):
        with open(path) as f:
            data = json.load(f)
        model_path = data.get("model", "")
        if model_path not in MODEL_BENCHMARK_MAP:
            continue
        reg = MODEL_REGISTRY.get(model_path, {})
        display_name = reg.get("display_name", MODEL_BENCHMARK_MAP[model_path])
        models[display_name] = {
            "metrics_file": path.name,
            "model_path": model_path,
            "meta": {k: v for k, v in reg.items() if k != "display_name"},
        }
    return models


BENCH_DIR = Path("benchmark_evals")

MCEVAL_LANG_ALIASES = {
    "C++": "C++", "C#": "C#", "Clisp": "Common Lisp", "Cobol": "COBOL",
    "Coffeescript": "CoffeeScript", "F#": "F#", "Objective-C": "Objective-C",
    "Visual Basic": "Visual Basic", "VB": "Visual Basic",
    "JS": "JavaScript", "TS": "TypeScript",
    "Shell": "Bash",
}

CRUXEVAL_LANG_MAP = {
    "cs": "C#", "cpp": "C++", "d": "D", "go": "Go", "java": "Java",
    "js": "JavaScript", "jl": "Julia", "lua": "Lua", "pl": "Perl",
    "php": "PHP", "py": "Python", "r": "R", "rkt": "Racket", "rb": "Ruby",
    "rs": "Rust", "scala": "Scala", "sh": "Shell", "swift": "Swift",
    "ts": "TypeScript",
}

CRUXEVAL_MODEL_MAP = {
    "CodeLlama-34B-Base": [
        ("CodeLlama", "34B"),
    ],
    "DeepSeek-Coder-7b-instruct": [
        ("Deepseekcoder-instruct", "6.7B"),
    ],
    "DeepSeek-Coder-7b-base": [
        ("Deepseekcoder-base", "6.7B"),
    ],
    "CodeLlama-7b-Instruct": [
        ("CodeLlama-Instruct", "7B"),
    ],
    "CodeLlama-7b": [
        ("CodeLlama-hf", "7B"),
    ],
    "CodeQwen1.5-7B-Chat": [
        ("CodeQwen1.5-Chat", "7B"),
    ],
    "CodeGen-multi-6B": [
        ("CodeGen-multi", "6B"),
    ],
    "phi-1_5": [
        ("phi-1_5", "1.3B"),
    ],
    "phi-1": [
        ("phi-1", "1.3B"),
    ],
    "Phi-3-Medium-4K-Instruct": [
        ("phi-3-instruct", "14B"),
    ],
    "Meta-Llama-3-8B-Instruct": [
        ("Llama-3-Instruct", "8B"),
    ],
    "StarCoder-15B": [
        ("Starcoder", "15B"),
    ],
    "WizardCoder-15B": [
        ("WizardCoder-V1.0", "15B"),
    ],
    "WizardCoder-33B-V1.1": [
        ("WizardCoder-V1.1", "33B"),
    ],
}

CRUXEVAL_WEB_MODEL_MAP = {
    "CodeLlama-34B-Base": "CodeLlama-34b-hf",
    "CodeLlama-34B-Python": "CodeLlama-34b-Python-hf",
    "CodeLlama-34B-Instruct": "CodeLlama-34b-Instruct-hf",
    "CodeQwen1.5-7B-Chat": "CodeQwen1.5-7B-Chat",
    "DeepSeek-Coder-33B-Instruct": "deepseek-coder-33b-instruct",
    "DeepSeek-Coder-33B-Base": "deepseek-coder-33b-base",
    "StarCoder2-15B": "starcoder2-15b",
    "StarCoder-15B": "starcoder",
    "CodeGen-multi-6B": "codegen-6B-multi",
    "phi-1_5": "phi-1_5",
    "phi-1": "phi-1",
    "Phi-3-Medium-4K-Instruct": "Phi-3-medium-4k-instruct",
    "Meta-Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct-hf",
    "WizardCoder-15B": "WizardCoder-15B-V1.0",
    "WizardCoder-33B-V1.1": "WizardCoder-33B-V1.1",
}

FULLSTACK_MODEL_MAP = {
    # FullStack processed files sometimes use source-specific model labels even
    # when they refer to the same public checkpoint.
    "Qwen2.5-Coder-32B-Instruct": "QwenAPI-2.5-coder.32b.instruct",
}

MCEVAL_MODEL_MAP = {
    "DeepSeek-Coder-7b-instruct": [
        "DeepSeek-Coder-1.5-7b-instruct",
        "deepseek-coder-7b-instruct",
        "DeepSeek-Coder-7B-instruct",
    ],
    "Magicoder-S-DS-6.7B": ["Magicoder-S-DS-6.7B"],
    "CodeLlama-7b-Instruct": ["CodeLlama-7b-Instruct", "CodeLlama-7B-Instruct"],
    "CodeLlama-7b": ["code-llama-7b"],
    "CodeLlama-34B-Instruct": ["CodeLlama-34b-Instruct", "CodeLlama-34B-Instruct"],
    "Codestral-22B-v0.1": ["Codestral-22B-v0.1"],
    "CodeQwen1.5-7B-Chat": ["CodeQwen-1.5-7B-Chat"],
    "DeepSeek-Coder-33B-Instruct": ["DeepSeek-Coder-33b-instruct"],
    "Meta-Llama-3-8B-Instruct": ["Llama-3-8B-Instruct"],
}

QWEN_PAPER_MODEL_MAP = {
    "DeepSeek-Coder-7b-instruct": "DeepSeek-Coder-6.7B-Instruct",
    "DeepSeek-Coder-7b-base": "DeepSeek-Coder-6.7B-Base",
    "Deepseek-Coder-1.3B": "DeepSeek-Coder-1.3B-Instruct",
    "DeepSeek-Coder-1.3B-Base": "DeepSeek-Coder-1.3B-Base",
    "CodeLlama-7b-Instruct": "CodeLlama-7B-Instruct",
    "StarCoder2-3B": "StarCoder2-3B",
    "StarCoder2-7B": "StarCoder2-7B",
    "StarCoder2-15B": "StarCoder2-15B",
    "Qwen2.5-Coder-0.5B": "Qwen2.5-Coder-0.5B",
    "Qwen2.5-Coder-0.5B-Instruct": "Qwen2.5-Coder-0.5B-Instruct",
    "Qwen2.5-Coder-1.5B": "Qwen2.5-Coder-1.5B",
    "Qwen2.5-Coder-1.5B-Instruct": "Qwen2.5-Coder-1.5B-Instruct",
    "Qwen2.5-Coder-3B": "Qwen2.5-Coder-3B",
    "Qwen2.5-Coder-3B-Instruct": "Qwen2.5-Coder-3B-Instruct",
    "Qwen2.5-Coder-7B": "Qwen2.5-Coder-7B",
    "Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B-Instruct",
    "CodeQwen1.5-7B-Chat": "CodeQwen1.5-7B-Chat",
    "Yi-Coder-1.5B": "Yi-Coder-1.5B-Chat",
    "Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B-Instruct",
    "Qwen2.5-Coder-14B": "Qwen2.5-Coder-14B",
    "Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "Qwen2.5-Coder-32B": "Qwen2.5-Coder-32B",
    "CodeLlama-13B-Instruct": "CodeLlama-13B-Instruct",
    "CodeLlama-34B-Instruct": "CodeLlama-34B-Instruct",
    "Codestral-22B-v0.1": "CodeStral-22B",
    "DeepSeek-Coder-33B-Base": "DeepSeek-Coder-33B-Base",
    "DeepSeek-Coder-33B-Instruct": "DeepSeek-Coder-33B-Instruct",
    "Yi-Coder-9B-Chat": "Yi-Coder-9B-Chat",
}

MULTI_LCB_MODEL_MAP = {
    "Gpt-oss-20B": "Gpt-oss-20B* (Medium)",
    "DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B*",
    "OpenReasoning-Nemotron-32B": "OpenRsn-Nmt-32B*",
    "OpenCodeReasoning-Nemotron-1.1-32B": "OpenCodeRsn-Nmt-1.1-32B*",
    "Qwen3-8B": "Qwen3-8B*",
    "Seed-Coder-8B-Instruct": "Seed-Coder-8B-Instr",
    "Devstral-Small-2505": "Devstral-Small-2505*",
    "Qwen3-30B-A3B": "Qwen3-30B-A3B*",
    "Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B-Instr-2507",
    "Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-A3B-Thk-2507*",
    "Qwen3-Coder-30B-A3B-Instruct": "Qwen3-Coder-30B-A3B-Instr",
    "Qwen3-32B": "Qwen3-32B*",
    "Qwen3-14B": "Qwen3-14B*",
    "DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B*",
    "Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B-Instr",
    "Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instr",
    "DeepSeek-Coder-33B-Instruct": "Deepseek-Coder-33B-Instr",
}

# The Qwen paper (arXiv 2409.12186) uses a different eval harness than BigCode
# for the same MultiPL-E problems.  All languages score somewhat higher (+2-5 pts
# on average), but C# is inflated by +14-17 pts across every overlapping model
# (e.g. StarCoder2-3B: BigCode C#=20.5 vs Qwen C#=34.8, while Python is identical
# at 31.7).  This makes the Qwen-paper C# scores unreliable for cross-language
# correlation, so we exclude C# from that source.
QWEN_PAPER_EXCLUDE_LANGS = {"C#"}

PARADIGM_GROUPS = {
    "Scripting": ["Ruby", "JavaScript", "Lua", "R"],
    "Systems": ["C++", "Rust", "Go", "Swift"],
    "Enterprise": ["Java", "C#"],
}

CODECONTESTS_LANG_MAP = {1: "Python", 2: "C++", 3: "Python", 4: "Java"}
