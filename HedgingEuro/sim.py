import networkx as nx
import math as m
import random as r
import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=float, help='short rate')
    parser.add_argument('--sigma', type=float, help='volatility')
    parser.add_argument('--SW_n', type=int, help='number of nodes')
    parser.add_argument('--SW_k', type=int, help='degree of nodes')
    parser.add_argument('--SW_p', type=float, help='rewiring probability')
    parser.add_argument('--nxd', type=str, help='nxdraw format', choices=['kk', 'circ'])
    parser.add_argument('--init_range', type=tuple, help='range for initial stock value')
    parser.add_argument('--num_shares', type=int, help='number of shares for each stock')
    parser.add_argument('--dt', type=float, help = 'incremental time step')
    parser.add_argument('--ttm', type=float, help='time to maturity for each option')

    parser.set_defaults(
        r = 0.05,
        sigma = 0.4,
        n = 20,
        k = 2,
        p = 0.9,
        nxd = 'circ',
        init_range = (1.0, 2.0),
        num_shares = 100,
        dt = 1.0,
        ttm = 10.0

    )
    return parser.parse_args()


def normal_cum(x):
    out = (1.0 + m.erf(x / m.sqrt(2.0))) / 2.0
    return out


def get_n(stk_pps, ror, sigma, ttm, prec = 0.01):
    stk1 = Stock(stk_pps, 0, 0)
    stk2 = Stock(stk_pps+prec, 0, 0)
    eco1 = EuropeanCallOption(stk1, ror, sigma, ttm, 0)
    eco2 = EuropeanCallOption(stk2, ror, sigma, ttm, 0)
    dVdS = (eco2.get_cost() - eco1.get_cost()) / prec
    return dVdS


class Market:
    def __init__(self, args):
        assert args.k % 2 == 0 and 0.0 <= args.p <= 1.0 and args.n > 1
        self.G = nx.watts_strogatz_graph(args.n, args.k, args.p)
        self.n = args.n
        self.k = args.k
        self.p = args.p
        self.r = args.r
        self.sigma = args.sigma
        self.dt = args.dt
        self.nxd = args.nxd
        self.ttm = args.ttm
        self.t = 0.0
        self.neighbor_tracker = []
        for i in range(args.n):
            pps = r.uniform(args.init_range[0], args.init_range[1])
            self.G.nodes[i]['self_stock'] = [Stock(pps, i, i) for _ in range(args.num_shares)]
            self.G.nodes[i]['self_stock_history'] = []
            self.G.nodes[i]['bought_stocks'] = []
            self.G.nodes[i]['bought_options'] = []
            self.G.nodes[i]['portfolio_value'] = []
            self.G.nodes[i]['portfolio_history'] = []
        for i in range(self.n):
            self.neighbor_tracker.append([n for n in self.G.neighbors(i)])

    def draw_graph(self):
        options = {
            'node_color': 'red',
            'node_size': 100,
            'width': 1,
        }

        if self.nxd == 'circ':
            pos = nx.circular_layout(self.G)
        elif self.nxd == 'kk':
            pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, pos=pos, **options, with_labels=True)
        plt.show()

    def get_pm_degree_dist(self, m):
        nm = 0
        for i in range(self.n):
            if self.G.degree[i] == m:
                nm += 1
        pm = nm / self.n
        return pm

    def get_mth_degree_moment(self, m):
        km = 0
        for i in range(self.n + 1):
            km += ((i**m) * self.get_pm_degree_dist(i))
        return km

    def get_mr(self):
        mr = self.get_mth_degree_moment(2) / self.get_mth_degree_moment(1)
        if mr > 2:
            print("Giant component exists")
        else:
            print("No giant component exists")

    def get_portfolio_value(self):
        pass

    def get_stock_value(self, node_num):
        totv = 0
        for stock in self.G.nodes[node_num]['self_stock']:
            totv += stock.get_value()
        return totv

    def evolve(self, tmax):
        for i in range(self.n):
            self.G.nodes[i]['self_stock_history'].append(self.G.nodes[i]['self_stock'][0].get_value())
            self.G.nodes[i]['portfolio_history'].append(self.get_stock_value(i))
        while self.t < tmax:

            # day-to-day stock fluctuations
            for i in range(self.n):
                weiner = r.gauss(0.0, 1.0)
                for stock in self.G.nodes[i]['self_stock']:
                    sv = stock.get_value()
                    sv = sv * (1 + (self.r * self.dt) + (self.sigma * weiner))
                    if sv <= 0.0:
                        sv = 0.1
                    stock.set_value(sv)
                self.G.nodes[i]['self_stock_history'].append(self.G.nodes[i]['self_stock'][0].get_value())

            # now we HEDGE!!!!!
            for i in range(self.n):
                num_neighbors = len(self.neighbor_tracker[i])
                stock_price = self.get_stock_value(i)
                price_per_neighbor = stock_price / num_neighbors
                for j in range(len(self.neighbor_tracker[i])):
                    # create option
                    st = Stock(price_per_neighbor, -1, -1)
                    eo = EuropeanCallOption(st, self.r, self.sigma, self.ttm, i)
                    n = round(get_n(stock_price, self.r, self.sigma, self.ttm))



            # update time
            self.t += self.dt

    def show_stock_history(self):
        for i in range(self.n):
            plt.plot(self.G.nodes[i]['self_stock_history'])
        plt.title("Stock value over time")
        plt.xlabel("Time (Financial Quarter)")
        plt.ylabel("Value ($)")
        plt.show()
        plt.clf()
        plt.cla()

    def show_portfolio_history(self):
        for i in range(self.n):
            plt.plot(self.G.nodes[i]['portfolio_history'])
        plt.title("Portfolio value over time")
        plt.xlabel("Time (Financial Quarter)")
        plt.ylabel("Value ($)")
        plt.show()
        plt.clf()
        plt.cla()


class Stock:
    def __init__(self, pps, owner_id, seller_id):
        self.price_per_share = pps
        self.owner_id = owner_id
        self.seller_id = seller_id

    def get_value(self):
        return self.price_per_share

    def set_value(self, new_pps):
        self.price_per_share = new_pps

    def get_owner_id(self):
        return self.owner_id

    def set_owner_id(self, noi):
        self.owner_id = noi

    def get_seller_id(self):
        return self.seller_id


class EuropeanCallOption:
    def __init__(self, stock, ror, sigma, maturity_time, seller_id):
        self.strike_price = stock.get_value() + (stock.get_value() * ror * maturity_time)
        self.maturity_time = maturity_time
        self.ror = ror
        self.sigma = sigma
        self.spot_price = stock.get_value()
        self.seller_id = seller_id

    def get_cost(self):
        d1 = (1.0 / (self.sigma * m.sqrt(self.maturity_time))) * ((m.log(self.spot_price / self.strike_price, m.e)) + (self.maturity_time*(self.ror + (0.5 * self.sigma**2))))
        d2 = d1 - (self.sigma * m.sqrt(self.maturity_time))
        pv = self.strike_price*m.exp(-1.0 * self.ror * self.maturity_time)
        cost = (normal_cum(d1) * self.spot_price) - (normal_cum(d2) * pv)
        return cost


def simulate(tmax: float):
    args = get_args()
    mkt = Market(args)
    mkt.draw_graph()
    mkt.evolve(tmax)
    mkt.show_stock_history()
    mkt.get_mr()


if __name__ == '__main__':
    simulate(100.0)