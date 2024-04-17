from pde import CartesianGrid, MemoryStorage, PDEBase, ScalarField, plot_kymograph
import math
import matplotlib.pyplot as plt


class BlackScholes:
    def __init__(self, r, K, sigma):
        self.r = r
        self.K = K
        self.sigma = sigma

    def solve(self, S, T):
        d1 = (1 / (self.sigma * math.sqrt(T))) * (math.log((S / self.K), math.e) + ((self.r + (0.5*(self.sigma**2)))*T))
        d2 = d1 - (self.sigma * math.sqrt(T))
        inc_term = 0.5 * S * (1 + math.erf(d1 / math.sqrt(2)))
        deg_term = -0.5 * math.exp(-1 * self.r * T) * self.K * (1 + math.erf(d2 / math.sqrt(2)))
        out = inc_term + deg_term
        return out


appl = BlackScholes(0.5, 1, 0.5)
print(appl.solve(5, 10))