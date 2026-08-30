"""fireworks-sp 发布预检：语法全检 -> pytest -> 干跑推荐（不推送）。

用法: uv run python scripts/preflight.py
全部通过输出 PREFLIGHT PASS 并退出码 0；任何失败退出码 1。
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def check_syntax():
    bad = []
    py_files = [f for f in sorted(ROOT.rglob("*.py"))
                if ".venv" not in f.relative_to(ROOT).parts]
    for f in py_files:
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{f.relative_to(ROOT)}: {e}")
    record("syntax", not bad, f"{len(py_files)} files" if not bad else "; ".join(bad[:3]))


def check_tests():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    tail = (r.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else f"exit={r.returncode}"
    record("pytest", r.returncode == 0, detail)


def check_recommend_dryrun():
    from fwsp import screener
    try:
        # persist=False：干跑推荐主链路但不写库（避免 preflight 副作用）
        top = screener.run_screen(top_n=3, persist=False)
        names = [f"{r['code']}({r.get('score', 0):.1f})" for r in top]
        ok = len(top) > 0
        record("recommend_dryrun", ok,
               ", ".join(names) if ok else "no candidates today")
    except Exception as e:  # noqa: BLE001
        record("recommend_dryrun", False, f"{type(e).__name__}: {e}")


def main():
    print("=" * 60)
    print("fireworks-sp preflight")
    print("=" * 60)
    check_syntax()
    check_tests()
    check_recommend_dryrun()
    print("-" * 60)
    failed = [n for n, ok, _ in CHECKS if not ok]
    if failed:
        print(f"PREFLIGHT FAIL: {', '.join(failed)}")
        return 1
    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
