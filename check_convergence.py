"""
Step 7's termination criterion: decides whether the loop should keep going
or stop, based on candidate_scores.json from the latest round.

Usage (call once per round, after score_and_return_top_candidates.py):
  python3 check_convergence.py \
    --candidate-scores outputs/exp_run1/scoring_live/candidate_scores.json \
    --history-file outputs/exp_run1/convergence_history.json

Exit code 0 = NOT converged, keep going (use in `while python3 check_convergence.py ...; do ...; done`)
Exit code 1 = CONVERGED, stop the loop
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
 
 
def load_history(history_file: Path) -> list:
    if history_file.exists():
        return json.loads(history_file.read_text())
    return []
 
 
def save_history(history_file: Path, history: list):
    history_file.write_text(json.dumps(history, indent=2))
 
 
def check_convergence(history: list, current_max: float,
                       relative_fraction: float, min_relative_improvement: float,
                       patience: int):
    """Returns (converged: bool, reason: str)."""
    if not history:
        return False, "First round -- no history yet, establishing baseline."
 
    first_max = history[0]["max_score"]
    if current_max < relative_fraction * first_max:
        return True, (f"Score dropped below {relative_fraction*100:.0f}% of first round's "
                       f"max ({current_max:.2f} < {relative_fraction:.2f} * {first_max:.2f})")
 
    recent = history[-(patience - 1):] + [{"max_score": current_max}]
    if len(recent) < patience:
        return False, (f"Still building history for the diminishing-returns check "
                        f"({len(recent)}/{patience} rounds so far)")
 
    improvements = []
    for i in range(1, len(recent)):
        prev, curr = recent[i - 1]["max_score"], recent[i]["max_score"]
        rel_improvement = (prev - curr) / prev if prev > 0 else 0
        improvements.append(rel_improvement)
 
    if all(abs(imp) < min_relative_improvement for imp in improvements):
        return True, (f"Diminishing returns -- last {patience} rounds each changed by "
                       f"less than {min_relative_improvement*100:.1f}% in magnitude "
                       f"({[f'{i*100:.2f}%' for i in improvements]})")
 
    return False, f"Still meaningfully improving ({[f'{i*100:.2f}%' for i in improvements]})"
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-scores", required=True, type=Path,
                     help="This round's candidate_scores.json (already sorted best-first)")
    ap.add_argument("--history-file", required=True, type=Path,
                     help="Persistent record across rounds -- created if it doesn't exist")
    ap.add_argument("--relative-fraction", type=float, default=0.1,
                     help="Stop when max score < this fraction of round 1's max (UNCALIBRATED default)")
    ap.add_argument("--min-relative-improvement", type=float, default=0.02,
                     help="Below this fractional improvement counts as 'not improving' (UNCALIBRATED default)")
    ap.add_argument("--patience", type=int, default=3,
                     help="Consecutive low-improvement rounds required before declaring convergence")
    args = ap.parse_args()
 
    scored = json.loads(args.candidate_scores.read_text())
    if not scored:
        print("[ERROR] candidate_scores.json is empty -- cannot evaluate convergence.")
        sys.exit(0)  # don't stop the loop on a data problem; keep going and flag it
 
    current_max = scored[0]["score"]  # sorted best-first, so this IS the max
    print(f"Round's top candidate score: {current_max:.4f}")
 
    history = load_history(args.history_file)
    converged, reason = check_convergence(
        history, current_max, args.relative_fraction,
        args.min_relative_improvement, args.patience
    )
 
    history.append({
        "timestamp": datetime.now().isoformat(),
        "max_score": current_max,
        "n_candidates": len(scored),
    })
    save_history(args.history_file, history)
 
    print(f"Round {len(history)}: {reason}")
 
    if converged:
        print(f"\n[CONVERGED] Stopping. {reason}")
        sys.exit(1)
    else:
        print(f"\n[CONTINUE] {reason}")
        sys.exit(0)
 
 
if __name__ == "__main__":
    main()