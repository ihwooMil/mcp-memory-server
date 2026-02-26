#!/usr/bin/env python3
"""Environment setup verification script.

Checks:
1. Python version
2. Required packages
3. Ollama connection
4. Model availability (qwen2.5:7b)
5. Data directories
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_python_version() -> bool:
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 11):
        print("  ✗ Python 3.11+ required")
        return False
    print("  ✓ Python version OK")
    return True


def check_packages() -> bool:
    required = ["ollama", "pydantic", "pyarrow", "pandas", "tqdm", "tiktoken", "matplotlib"]
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} not installed")
            all_ok = False
    return all_ok


def check_ollama() -> bool:
    try:
        import ollama

        models = ollama.list()
        model_names = [m.model for m in models.models]
        print(f"  ✓ Ollama connected, {len(model_names)} models available")

        target = "qwen2.5:7b"
        if any(target in name for name in model_names):
            print(f"  ✓ {target} is available")
        else:
            print(f"  ✗ {target} not found. Run: ollama pull {target}")
            print(f"    Available models: {model_names}")
            return False
        return True
    except Exception as e:
        print(f"  ✗ Ollama connection failed: {e}")
        print("    Make sure Ollama is running: ollama serve")
        return False


def check_directories() -> bool:
    project_root = Path(__file__).resolve().parent.parent
    dirs = [
        project_root / "data" / "raw" / "episodes",
        project_root / "data" / "splits",
    ]
    all_ok = True
    for d in dirs:
        if d.exists():
            print(f"  ✓ {d.relative_to(project_root)}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {d.relative_to(project_root)} (created)")
    return all_ok


def check_mecab() -> bool:
    try:
        import MeCab

        tagger = MeCab.Tagger()
        result = tagger.parse("테스트 문장입니다")
        if result:
            print("  ✓ MeCab working")
            return True
        print("  ✗ MeCab parse returned empty")
        return False
    except Exception as e:
        print(f"  ✗ MeCab not available: {e}")
        print("    Install: pip install mecab-python3 unidic-lite")
        return False


def main() -> None:
    print("=" * 50)
    print("AI Memory System - Environment Check")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("MeCab (형태소 분석기)", check_mecab),
        ("Ollama Connection & Model", check_ollama),
        ("Data Directories", check_directories),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        results.append((name, check_fn()))

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All checks passed! Ready to generate episodes.")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
