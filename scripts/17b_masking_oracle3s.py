"""Oracle for the three-state learner in the masking experiment (unmasked windows), so gap rules are compared
with the oracle of their own model class."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("m17", Path(__file__).parent / "17_masking.py")
src = (Path(__file__).parent / "17_masking.py").read_text(encoding="utf-8")
head = src[:src.index("prev = pd.read_csv")]          # reuse data/split/eval definitions only
exec(compile(head, "17_masking_head", "exec"))
g = GAMS(use_em=False).fit(O[SPL["tr"]], state[SPL["tr"]], X[SPL["tr"]], (ws[SPL["tr"]], wc[SPL["tr"]]),
                           O[SPL["va"]], state[SPL["va"]], X[SPL["va"]], (ws[SPL["va"]], wc[SPL["va"]]), verbose=False)
rows = []
for trans, st in [("onset", 0), ("recovery", 1)]:
    mte = te_idx[trans]; a, b, c = g.hazards(X[mte], ws[mte], wc[mte])
    rows.append(evaluate(trans, a if st == 0 else b, "Oracle 3-state (no gaps)", "all", 0))
R = pd.read_csv(MD / "masking_results.csv"); R = R[R.rule != "Oracle 3-state (no gaps)"]
R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True); R.to_csv(MD / "masking_results.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
