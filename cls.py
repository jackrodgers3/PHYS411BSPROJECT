import numpy as np
from datetime import date
import random
import matplotlib.pyplot as plt
import networkx as nx
import math
from itertools import count


class Market:
    def __init__(self, ror: float, sigma: float, n: int, k: int, p: float, mt: float, otype: str, otype2: str, Nprec: float):
        self.ror = ror
        self.sigma = sigma
        self.stocks = []
        self.n = n
        self.k = k
        self.p = p
        self.G = None
        self.S_tracker = []
        self.deltaS_tracker = []
        self.neighbor_tracker = []
        self.mt = mt
        self.otype = otype
        self.otype2 = otype2
        self.Nprec = Nprec

    def create(self, init_range: tuple, seed=None):
        self.G = nx.watts_strogatz_graph(self.n, self.k, self.p, seed=seed)
        initS = []
        initdS = []
        for i in range(self.n):
            self.G.nodes[i]['S'] = random.uniform(init_range[0], init_range[1])
            self.G.nodes[i]['num_shares'] = 100
            self.G.nodes[i]['options'] = []
            self.G.nodes[i]['stocks'] = []
            initS.append(self.G.nodes[i]['S'])
            initdS.append(0.0)
        self.S_tracker.append(initS)
        self.deltaS_tracker.append(initdS)
        for i in range(self.n):
            self.neighbor_tracker.append([n for n in self.G.neighbors(i)])
        return self.G.nodes.data()

    def draw_graph(self, type: str):
        assert type == 'circ' or type == 'kk'
        options = {
            'node_color': 'red',
            'node_size': 100,
            'width': 1,
        }

        if type == 'circ':
            pos = nx.circular_layout(self.G)
        elif type == 'kk':
            pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, pos=pos, **options, with_labels=True)
        plt.show()

    def get_gl(self, get_eigs = False):
        gl = nx.laplacian_matrix(self.G).toarray()
        if get_eigs:
            eigs = np.linalg.eig(gl)
            return gl, eigs[0]
        else:
            return gl

    def update(self, t, dt):
        int_S = []
        for i in range(self.n):
            self.G.nodes[i]['S'] += ((self.ror * self.G.nodes[i]['S'] * dt) + (self.sigma * self.G.nodes[i]['S'] * random.gauss(0.0, 1.0)))
            int_S.append(self.G.nodes[i]['S'])
        if int(t) % int(self.mt) == 0 and int(t) != 0:
            for i in range(self.n):
                for option in self.G.nodes[i]['options']:
                    self.G.nodes[i]['S'] += option.exercise(self.G)
                    print(f"Node {i} sold an option for a gain of {option.exercise(self.G)}")
                for stock in self.G.nodes[i]['stocks']:
                    self.G.nodes[i]['S'] += stock.exercise(self.G)
                    print(f"Node {i} sold {stock.get_N()} stocks for a gain of {stock.exercise(self.G)}")
                self.G.nodes[i]['options'] = []
                self.G.nodes[i]['stocks'] = []
            self.assign_options()
        self.S_tracker.append(int_S)
        return self.G.nodes.data()

    def print_stock_history(self):
        return self.S_tracker

    def show_stock_history(self):
        s_history = np.array(self.S_tracker).transpose()
        for i in range(len(s_history)):
            plt.plot(s_history[i])
        plt.title("Stock value over time")
        plt.xlabel("Time (Financial Quarter)")
        plt.ylabel("Value ($)")
        plt.show()
        plt.clf()
        plt.cla()

    def assign_options(self):
        if self.otype == 'call' and self.otype2 == 'digital':
            for i in range(self.n):
                for j in range(len(self.neighbor_tracker[i])):
                    self.G.nodes[i]['S'] -= 0.1
                    self.G.nodes[i]['options'].append(DigitalCallOption(self.ror,
                    self.G.nodes[self.neighbor_tracker[i][j]]['S'], self.mt, self.neighbor_tracker[i][j]))
        elif self.otype == 'put' and self.otype2 == 'digital':
            for i in range(self.n):
                for j in range(len(self.neighbor_tracker[i])):
                    self.G.nodes[i]['S'] -= 0.1
                    self.G.nodes[i]['options'].append(DigitalPutOption(self.ror,
                    self.G.nodes[self.neighbor_tracker[i][j]]['S'], self.mt, self.neighbor_tracker[i][j]))
        elif self.otype == 'put' and self.otype2 == 'euro':
            for i in range(self.n):
                for j in range(len(self.neighbor_tracker[i])):
                    opt = EuroPutOption(self.ror, self.sigma,
                    self.G.nodes[self.neighbor_tracker[i][j]]['S'], self.mt, self.neighbor_tracker[i][j])
                    self.G.nodes[i]['S'] -= opt.get_cost()
                    self.G.nodes[i]['options'].append(opt)
        elif self.otype == 'call' and self.otype2 == 'euro':
            for i in range(self.n):
                for j in range(len(self.neighbor_tracker[i])):
                    opt = EuroCallOption(self.ror, self.sigma,
                                        self.G.nodes[self.neighbor_tracker[i][j]]['S'], self.mt,
                                        self.neighbor_tracker[i][j])
                    opt2 = EuroCallOption(self.ror, self.sigma,
                                        self.G.nodes[self.neighbor_tracker[i][j]]['S'] + self.Nprec, self.mt,
                                        self.neighbor_tracker[i][j])
                    N = abs(opt2.get_cost() - opt.get_cost() / self.Nprec)
                    stk = EuroStock(self.G.nodes[self.neighbor_tracker[i][j]]['S'], N, self.neighbor_tracker[i][j])
                    self.G.nodes[i]['stocks'].append(stk)
                    self.G.nodes[i]['S'] -= opt.get_cost()
                    print(f"Node {i} bought option for {opt.get_cost()}")
                    print(f"Node {i} bought {N} stocks for {stk.get_cost()}")
                    self.G.nodes[i]['S'] -= stk.get_cost()
                    self.G.nodes[i]['options'].append(opt)


class DigitalCallOption:
    def __init__(self, ror, spot_price, maturity_time, seller_id):
        self.strike_price = spot_price + (ror*spot_price*maturity_time)
        self.seller_id = seller_id

    def exercise(self, G):
        if G.nodes[self.seller_id]['S'] <= self.strike_price:
            return 0
        else:
            return 1


class DigitalPutOption:
    def __init__(self, ror, spot_price, maturity_time, seller_id):
        self.strike_price = spot_price + (ror * spot_price * maturity_time)
        self.seller_id = seller_id

    def exercise(self, G):
        if G.nodes[self.seller_id]['S'] <= self.strike_price:
            return 1
        else:
            return 0


class EuroCallOption:
    def __init__(self, ror, sigma, spot_price, maturity_time, seller_id):
        self.spot_price = spot_price
        self.T = maturity_time
        self.strike_price = spot_price + (ror*spot_price*maturity_time)
        self.seller_id = seller_id
        self.ror = ror
        self.sigma = sigma

    def exercise(self, G):
        ret = max(G.nodes[self.seller_id]['S'] - self.strike_price, 0)
        return ret

    def get_cost(self):
        d1 = ((math.log((self.spot_price / self.strike_price), math.e)) + ((self.ror + (0.5*(self.sigma**2)))*self.T)) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - (self.sigma * math.sqrt(self.T))
        cost = self.spot_price * math.erf(d1) - (math.exp(-1 * self.ror * self.T) * self.strike_price * math.erf(d2))
        return cost


class EuroPutOption:
    def __init__(self, ror, sigma, spot_price, maturity_time, seller_id):
        self.ror = ror
        self.sigma = sigma
        self.T = maturity_time
        self.spot_price = spot_price
        self.strike_price = spot_price + (ror * spot_price * maturity_time)
        self.seller_id = seller_id

    def exercise(self, G):
        ret = max(self.strike_price - G.nodes[self.seller_id]['S'], 0)
        return ret

    def get_cost(self):
        d1 = ((math.log((self.spot_price / self.strike_price), math.e)) + (
                    (self.ror + (0.5 * (self.sigma ** 2))) * self.T)) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - (self.sigma * math.sqrt(self.T))
        cost = (math.exp(-1 * self.ror * self.T) * self.strike_price * math.erf(-1 * d2)) - (self.spot_price * math.erf(-1* d1))
        return cost


class EuroStock:
    def __init__(self, spot_price, N, seller_id):
        self.N = N
        self.seller_id = seller_id
        self.spot_price = spot_price

    def exercise(self, G):
        ret = self.N * (G.nodes[self.seller_id]['S'] - self.spot_price)
        return ret

    def get_cost(self):
        return self.N * self.spot_price

    def get_N(self):
        return self.N


if __name__ == '__main__':
    pass