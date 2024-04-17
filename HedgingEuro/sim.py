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

    parser.set_defaults(
        r = 0.01,
        sigma = 0.10,
        n = 20,
        k = 2,
        p = 0.9,
        nxd = 'circ',
        init_range = (1.0, 2.0),
        num_shares = 100,
        dt = 1.0

    )
    return parser.parse_args()


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
        self.t = 0.0
        self.neighbor_tracker = []
        for i in range(args.n):
            self.G.nodes[i]['self_stock'] = Stock(args.init_range, args.num_shares)
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

    def evolve(self, tmax):
        for i in range(self.n):
            self.G.nodes[i]['self_stock_history'].append(self.G.nodes[i]['self_stock'].get_value())
            self.G.nodes[i]['portfolio_history'].append(self.G.nodes[i]['self_stock'].get_value() * self.G.nodes[i]['self_stock'].get_num_shares())
        while self.t < tmax:

            # day-to-day stock fluctuations
            for i in range(self.n):
                sv = self.G.nodes[i]['self_stock'].get_value()
                sv = sv * (1 + (self.r * self.dt) + (self.sigma * r.gauss(0.0, 1.0)))
                if sv <= 0.0:
                    sv = 0.1
                self.G.nodes[i]['self_stock_history'].append(sv)
                self.G.nodes[i]['self_stock'].set_value(sv)
                self.G.nodes[i]['portfolio_value'] = sv * self.G.nodes[i]['self_stock'].get_num_shares()
                self.G.nodes[i]['portfolio_history'].append(sv * self.G.nodes[i]['self_stock'].get_num_shares())

            # buying and selling stocks from neighbors
            for i in range(self.n):
                for j in range(len(self.neighbor_tracker[i])):
                    pass

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
    def __init__(self, init_range, num_shares):
        self.price_per_share = r.uniform(init_range[0], init_range[1])
        self.num_shares = num_shares

    def get_value(self):
        return self.price_per_share

    def set_value(self, new_pps):
        self.price_per_share = new_pps

    def get_num_shares(self):
        return self.num_shares

    def set_num_shares(self, new_ns):
        self.num_shares = new_ns


class EuropeanPutOption:
    def __init__(self):
        self.value = 0


def simulate(tmax: float):
    args = get_args()
    mkt = Market(args)
    mkt.draw_graph()
    mkt.evolve(tmax)
    mkt.show_stock_history()
    mkt.show_portfolio_history()
    mkt.get_mr()


if __name__ == '__main__':
    simulate(100.0)