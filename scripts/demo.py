"""In-process demo of the routing proxy with the mock backend (no API keys, no network).

Usage (from the repo root):  python scripts/demo.py [--artifact artifacts/router_lr] [--embedder hashing]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROMPTS = [
    "What is the capital of France?",
    "Hi! How are you today?",
    "Who wrote Romeo and Juliet?",
    "What color is the sky on a clear day?",
    "What is 17 + 25?",
    "Compute 144 / 12 - 3.",
    "What is 15% of 80?",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "A train leaves at 3pm going 60 mph and a second leaves at 4pm going 80 mph from the same station. "
    "At what time does the second train catch up? Show each step.",
    "Tom buys 3 notebooks at $2.50 each and 4 pens at $1.20 each, pays with a $20 bill, then spends half "
    "of the change on a snack. How much money does he have left?",
    "Write a Python function that returns the n-th Fibonacci number using memoization.",
    "Implement binary search in Python and explain its time complexity.",
    "Write a function that checks whether a string is a valid palindrome, ignoring punctuation.",
    "Write a SQL query that returns the top 3 customers by total order value from tables customers and orders.",
    "Which of the following is a noble gas?\nA) Neon\nB) Iron\nC) Sodium\nD) Chlorine\nAnswer:",
    "Which organelle is the site of photosynthesis?\nA) Mitochondrion\nB) Chloroplast\nC) Ribosome\nD) Nucleus",
    "What force keeps planets in orbit around the Sun?\nA) Magnetism\nB) Friction\nC) Gravity\nD) Buoyancy",
    "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Explain.",
    "Prove that the sum of two even integers is even, then discuss whether the converse is true.",
    "Three friends each tell one lie and one truth about who broke the vase. Given their statements, "
    "deduce who broke it and justify your reasoning step by step.",
    "Summarize the main trade-offs between microservices and a monolith for a small startup.",
    "Translate 'good morning' into Spanish.",
    "Explain why the derivative of x^3 is 3x^2 and use it to find the minimum of x^3 - 3x.",
    "Name three primary colors.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default=str(REPO_ROOT / "artifacts" / "router_lr"),
                    help="artifact stem (without .npz/.json)")
    ap.add_argument("--embedder", default=None, help="override the artifact's embedder name (e.g. hashing)")
    args = ap.parse_args()

    artifact = Path(args.artifact)
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact
    if not artifact.with_name(artifact.name + ".npz").exists():
        print(f"Router artifact not found at {artifact}(.npz/.json).\n"
              "Run `python scripts/reproduce.py` first to train and export it.")
        return 1

    db_path = REPO_ROOT / "data" / "demo_requests.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-journal", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()

    os.environ["ROUTER_BACKEND"] = "mock"
    os.environ["ROUTER_DB"] = str(db_path)
    os.environ["ROUTER_ARTIFACT"] = str(artifact)
    os.environ.pop("ROUTER_POLICY", None)
    if args.embedder:
        os.environ["ROUTER_EMBEDDER"] = args.embedder

    from fastapi.testclient import TestClient

    from router.serve.app import create_app

    print("Loading router (embedder warm-up may take a few seconds)...")
    client = TestClient(create_app())
    health = client.get("/health").json()
    print(f"policy={health['policy']}  threshold={health['threshold']:.3f}  backend={health['backend']}\n")

    print(f"{'prompt':<56} {'p_strong':>8}  {'route':<6} {'overhead ms':>11}")
    print("-" * 86)
    for prompt in PROMPTS:
        r = client.post("/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": prompt}]})
        info = r.json()["router"]
        snippet = prompt.replace("\n", " ")
        snippet = snippet if len(snippet) <= 54 else snippet[:51] + "..."
        print(f"{snippet:<56} {info['p_strong']:>8.3f}  {info['route']:<6} {info['overhead_ms']:>11.2f}")

    s = client.get("/stats").json()
    print("\n=== /stats summary ===")
    print(f"requests:                {s['n_requests']}")
    print(f"routed to cheap:         {(1 - s['frac_strong']) * 100:.1f}%  ({s['n_requests'] - s['n_strong']} of {s['n_requests']})")
    print(f"mock cost:               ${s['total_cost_usd']:.6f}")
    print(f"always-strong cost:      ${s['always_strong_cost_usd']:.6f}")
    print(f"cost saving:             {s['cost_saving_frac'] * 100:.1f}%")
    print(f"routing overhead p50/p95: {s['overhead_ms_p50']:.2f} / {s['overhead_ms_p95']:.2f} ms")
    print("\nNote: responses come from the mock backend (canned text, token counts estimated as chars/4);")
    print("costs use the price table in configs/default.yaml. No real LLM was called.")
    print(f"Request log: {db_path}  (view with `streamlit run router/dashboards/app.py`, ROUTER_DB set to it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
