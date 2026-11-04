"""Learn-then-Test (Angelopoulos et al. 2021): certify every grid configuration whose risks are all <= alpha."""
import math
from scipy.stats import binom


def hb_pvalue(risk, n, alpha):
    """Hoeffding-Bentkus p-value for H0: true risk > alpha, given empirical risk over n samples."""
    if n == 0:
        return 1.0
    r = min(risk, alpha)
    h = (r * math.log(r / alpha) if r > 0 else 0.0) + (1 - r) * math.log((1 - r) / (1 - alpha))
    return min(1.0, math.exp(-n * h), math.e * binom.cdf(math.ceil(n * risk), n, alpha))


def ltt(configs, risks, alphas, delta=0.1):
    """configs: list; risks(cfg) -> [(empirical_risk, n), ...] aligned with alphas. Bonferroni over the grid.

    A config is valid when every risk's null is rejected; max of the per-risk p-values is a valid p-value
    for the union null. Returns [(cfg, p)] for the valid ones, which may be empty.
    """
    out = []
    for cfg in configs:
        p = max(hb_pvalue(r, n, a) for (r, n), a in zip(risks(cfg), alphas))
        if p <= delta / len(configs):
            out.append((cfg, p))
    return out


def floor(n, n_configs, delta=0.1):
    """Smallest alpha LTT can ever certify with n samples and zero observed errors (HB tail ~ (1-alpha)^n)."""
    return 1 - (delta / n_configs) ** (1 / n)


if __name__ == "__main__":
    assert hb_pvalue(0.0, 1000, 0.02) < 1e-8 and hb_pvalue(0.05, 1000, 0.02) == 1.0
    print("floor n=700 grid=343:", round(floor(700, 343), 4))
