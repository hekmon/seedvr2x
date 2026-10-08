# False alarms of the band-energy rules for one further fp16 seed (Gaussian seed scatter, Monte Carlo, pure Python):
# range rule: X_new outside [min, max] of the n reference seeds; spread rule: |X_new - X_42| > max - min.
import random

random.seed(0)
N = 400_000
for n in (2, 3):
    r = s = 0
    for _ in range(N):
        x = [random.gauss(0, 1) for _ in range(n)]
        new = random.gauss(0, 1)
        lo, hi = min(x), max(x)
        r += new < lo or new > hi
        s += abs(new - x[0]) > hi - lo
    r, s = r / N, s / N
    print(
        f"n={n}: range rule {r:.3f} per clip-variant (2/(n+1) = {2 / (n + 1):.3f}); spread rule {s:.3f}; "
        f"any of 8: range {1 - (1 - r) ** 8:.2f}, spread {1 - (1 - s) ** 8:.2f}; any of 6: range {1 - (1 - r) ** 6:.2f}, spread {1 - (1 - s) ** 6:.2f}"
    )
