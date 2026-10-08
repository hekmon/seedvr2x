# The multiple c of the 7B's seed spread (max - min of n seeds) such that one further fp16 seed has
# |X_new - X_42| > c * spread with probability p (Gaussian seed scatter; both directions), Monte Carlo.
import random

random.seed(1)
N = 400_000
for n in (2, 3):
    ratios = []
    for _ in range(N):
        x = [random.gauss(0, 1) for _ in range(n)]
        new = random.gauss(0, 1)
        ratios.append(abs(new - x[0]) / (max(x) - min(x)))
    ratios.sort()
    q = lambda p: ratios[int((1 - p) * N)]
    print(
        f"n={n}: c(29%/50% baseline) = 1; c for 10% = {q(0.10):.2f}, 5% = {q(0.05):.2f}, 1% = {q(0.01):.2f}; "
        f"one direction only halves these rates"
    )
