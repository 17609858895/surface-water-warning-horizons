"""Small checks for scientific helper invariants; no model fitting or data access."""
from pathlib import Path
import ast
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from survlib import surv_from_h, nll_rc, person_period

for path in ROOT.rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
h = np.array([[.2, .3, .4], [.2, .3, .4], [.2, .3, .4]])
np.testing.assert_allclose(surv_from_h(h)[0], [.8, .56, .336])
assert np.all(np.diff(1 - surv_from_h(h), axis=1) >= 0)
landmarks = pd.DataFrame({"T": [2, 2, 0], "E": [1, 0, 0]})
np.testing.assert_allclose(nll_rc(h, landmarks, reduce=False), [-np.log(.8*.3), -np.log(.8*.7), 0])
X = np.arange(3)[:, None]
xx, week, event, index = person_period(landmarks, X, kmax=3)
np.testing.assert_array_equal(week, [1, 2, 1, 2])
np.testing.assert_array_equal(event, [0, 1, 0, 0])
np.testing.assert_array_equal(index, [0, 0, 1, 1])
print("PASS: syntax, survival monotonicity, event/censoring likelihood, person-period labels")
