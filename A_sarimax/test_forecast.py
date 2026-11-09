"""Check: vectorised Z T^h a_t + x'beta equals statsmodels' own h-step forecast from origin t."""
import sys, warnings
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import asarimax as A
from statsmodels.tsa.statespace.sarimax import SARIMAX

rng = np.random.default_rng(0)
n = 3000
X = np.column_stack([np.ones(n + 30), rng.normal(size=(n + 30, 2))])
e = np.zeros(n)
for t in range(2, n):
    e[t] = 0.7 * e[t - 1] - 0.1 * e[t - 2] + rng.normal()
y = X[:n] @ [1.0, 0.5, -0.3] + e
y[100:110] = np.nan
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    full = SARIMAX(y, X[:n], order=(2, 0, 1)).fit(disp=False)
for i, h in [(500, 3), (105, 6), (2000, 24), (n - 1, 12)]:
    mine = A.forecasts(full, X, np.array([i]), h)[0]
    ref = full.apply(y[:i + 1], X[:i + 1]).forecast(h, exog=X[i + 1:i + h + 1])[-1]
    assert abs(mine - ref) < 1e-8, (i, h, mine, ref)
assert (A.lagged(np.arange(5.0), 2, 6) == [0, 0, 0, 1, 2, 3]).all()
print("ok")
