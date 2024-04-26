import sys
import copy
import networkx as nx
import math as m
import random as r
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


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
    parser.add_argument('--dt', type=float, help='incremental time step')
    parser.add_argument('--ttm', type=float, help='time to maturity for each option')

    parser.set_defaults(
        r=0.05,
        sigma=0.1,
        n=30,
        k=2,
        p=0.23,
        nxd='circ',
        init_range=(1.0, 2.0),
        num_shares=20,
        dt=1.0,
        ttm=5.0

    )
    return parser.parse_args()


def normal_cum(x):
    out = (1.0 + m.erf(x / m.sqrt(2.0))) / 2.0
    return out


class Market:
    def __init__(self, args):
        assert 0.0 <= args.p <= 1.0 and args.n > 1
        self.G = nx.watts_strogatz_graph(args.n, args.k, args.p)
        self.n = args.n
        self.k = args.k
        self.p = args.p
        self.r = args.r
        self.sigma = args.sigma
        self.dt = args.dt
        self.nxd = args.nxd
        self.stock_history = []
        self.num_shares = float(args.num_shares)
        self.ttm = args.ttm
        self.t = 0.0
        self.adjacency_matrix = []
        self.S_matrix = []
        self.num_shares_matrix = []
        self.am_matrix = []
        self.valuation_matrix = None
        self.portfolio_history = []
        self.portfolio_vector = [0.0 for _ in range(self.n)]
        for i in range(args.n):
            pps = r.uniform(args.init_range[0], args.init_range[1])
            temp = []
            for j in range(args.n):
                if i == j:
                    temp.append(pps)
                else:
                    temp.append(0.0)
            self.S_matrix.append(temp)

        for i in range(args.n):
            temp = []
            for j in range(args.n):
                if i in self.G.neighbors(j):
                    temp.append(1.0)
                else:
                    temp.append(0.0)
            self.adjacency_matrix.append(temp)

        for i in range(args.n):
            ins = self.num_shares
            temp = []
            for j in range(args.n):
                if i == j:
                    temp.append(ins)
                else:
                    temp.append(0.0)
            self.num_shares_matrix.append(temp)

        for i in range(args.n):
            scale = (1.0 / sum(self.adjacency_matrix[i]))
            temp = []
            for j in range(len(self.adjacency_matrix[i])):
                temp.append(scale * self.adjacency_matrix[i][j])
            self.am_matrix.append(temp)

        # making operations easier
        self.adjacency_matrix = np.array(self.adjacency_matrix)
        self.S_matrix = np.array(self.S_matrix)


    def draw_graph(self):
        options = {
            'node_color': 'red',
            'node_size': 150,
            'width': 1,
        }

        if self.nxd == 'circ':
            pos = nx.circular_layout(self.G)
        elif self.nxd == 'kk':
            pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, pos=pos, **options, with_labels=True)
        plt.show()

    def get_s_matrix(self):
        return self.S_matrix

    def get_adj_matrix(self):
        return self.adjacency_matrix

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
            km += ((i ** m) * self.get_pm_degree_dist(i))
        return km

    def get_mr(self):
        mr = self.get_mth_degree_moment(2) / self.get_mth_degree_moment(1)
        if mr > 2:
            print("Giant component exists")
        else:
            print("No giant component exists")

    def get_n(self, stk_pps, prec=0.01):
        stk1 = stk_pps
        stk2 = stk_pps + prec
        eco1 = EuropeanCallOption(stk1, self.r, self.sigma, self.ttm, 0)
        eco2 = EuropeanCallOption(stk2, self.r, self.sigma, self.ttm, 0)
        dVdS = (eco2.get_cost() - eco1.get_cost()) / prec
        return dVdS

    def get_stock_value(self, node_num):
        totv = 0
        for i in range(len(self.S_matrix[node_num])):
            totv += self.S_matrix[node_num][i]
        return totv

    def evolve(self, tmax):
        temp = []
        for i in range(self.n):
            temp.append(self.S_matrix[i][i])
        self.stock_history.append(temp)
        temp = []
        for i in range(self.n):
            temp.append(self.num_shares * self.S_matrix[i][i])
        self.portfolio_vector = temp

        while self.t < tmax:
            # day-to-day stock fluctuations
            temp = []
            for i in range(self.n):
                weiner = r.gauss(0.0, 1.0)
                for j in range(self.n):
                    sv = self.S_matrix[i][j]
                    sv = sv * (1 + (self.r * self.dt) + (self.sigma * weiner))
                    if sv < 0.0:
                        sv = 0.1
                    self.S_matrix[i][j] = sv
                temp.append(self.S_matrix[i][i])
            self.stock_history.append(temp)

            if int(self.t) == 0 or int(self.t) % int(self.ttm) == 0:
                if int(self.t) > 0:
                    Vpay_matrix = []
                    for i in range(self.n):
                        temp = []
                        for j in range(self.n):
                            if i == j:
                                temp.append(float(active_options[i].exercise(self.S_matrix[i][i])))
                            else:
                                temp.append(0.0)
                        Vpay_matrix.append(temp)
                    #print("cur S matrix")
                    #print(self.S_matrix)
                    #print("past S matrix")
                    #print(cur_S_matrix)
                    delta_S_matrix = np.subtract(self.S_matrix, cur_S_matrix)
                    #print("delta S matrix")
                    #print(delta_S_matrix)
                    stock_part = (delta_S_matrix @ self.adjacency_matrix)
                    stock_part = n_matrix @ stock_part
                    #print(stock_part)
                    option_cost_part = (Vcost_matrix @ self.adjacency_matrix)
                    option_pay_part = (Vpay_matrix @ self.adjacency_matrix)
                    portfolio_matrix = stock_part - option_cost_part
                    portfolio_matrix = portfolio_matrix.transpose() + option_pay_part
                    temp = []
                    for i in range(self.n):
                        self.portfolio_vector[i] += sum(portfolio_matrix[i])
                        temp.append(sum(portfolio_matrix[i]))
                    self.portfolio_history.append(temp)

                #construct options
                cur_S_matrix = copy.copy(self.S_matrix)
                Vcost_matrix = []
                active_options = []
                n_matrix = []
                # make options and get Ns
                for i in range(self.n):
                    eo = EuropeanCallOption(self.S_matrix[i][i], self.r, self.sigma, self.ttm, -1)
                    active_options.append(eo)
                    temp = []
                    for j in range(self.n):
                        if i == j:
                            temp.append(self.get_n(self.S_matrix[i][i]))
                        else:
                            temp.append(0.0)
                    n_matrix.append(temp)
                n_matrix = np.array(n_matrix)
                # construct V_matrix
                for i in range(self.n):
                    temp = []
                    for j in range(self.n):
                        if i == j:
                            temp.append(active_options[i].get_cost())
                        else:
                            temp.append(0.0)
                    Vcost_matrix.append(temp)


            # update time
            self.t += self.dt

    def show_stock_history(self):
        temp_stock_history = np.array(self.stock_history).transpose()
        for i in range(self.n):
            plt.plot(temp_stock_history[i])
        plt.title("Stock value over time")
        plt.xlabel("Time (days)")
        plt.ylabel("Value ($)")
        plt.show()
        plt.clf()
        plt.cla()

    def show_portfolio_history(self):
        temp_portfolio_history = np.array(self.portfolio_history).transpose()
        for i in range(self.n):
            plt.plot(temp_portfolio_history)
        plt.title("Portfolio value change over time")
        plt.xlabel("Exercise period")
        plt.ylabel("Value ($)")
        plt.show()
        plt.clf()
        plt.cla()

    def show_portfolio_value(self):
        return self.portfolio_vector, np.mean(self.portfolio_vector)


class EuropeanCallOption:
    def __init__(self, stock_pr, ror, sigma, maturity_time, seller_id):
        self.strike_price = stock_pr + (stock_pr * ror * maturity_time)
        self.maturity_time = maturity_time
        self.ror = ror
        self.sigma = sigma
        self.spot_price = stock_pr
        self.seller_id = seller_id

    def get_cost(self):
        d1 = (1.0 / (self.sigma * m.sqrt(self.maturity_time))) * ((m.log(self.spot_price / self.strike_price, m.e)) + (
                    self.maturity_time * (self.ror + (0.5 * self.sigma ** 2))))
        d2 = d1 - (self.sigma * m.sqrt(self.maturity_time))
        pv = self.strike_price * m.exp(-1.0 * self.ror * self.maturity_time)
        cost = (normal_cum(d1) * self.spot_price) - (normal_cum(d2) * pv)
        return cost

    def exercise(self, new_price):
        return max(new_price - self.strike_price, 0)


def simulate(tmax: float, args):
    mkt = Market(args)
    mkt.evolve(tmax)
    return mkt.show_portfolio_value()


def simulate_show(tmax: float, args):
    mkt = Market(args)
    mkt.evolve(tmax)
    mkt.show_stock_history()
    mkt.show_portfolio_history()
    return mkt.show_portfolio_value()


def monte_carlo_sim_k():
    ks = [2, 4, 6, 8, 10, 12]
    iters = 1000
    temp = []
    for i in tqdm(range(len(ks)), desc="k"):
        tval = 0.0
        args = get_args()
        args.r = 0.0
        args.p = 0.0
        args.k = ks[i]
        for j in tqdm(range(iters)):
            pv, mp = simulate(100.0, args)
            tval += mp
        temp.append((tval / iters) / 1.5)
    plt.plot(ks, temp)
    plt.title('k vs. factor increase')
    plt.xlabel('k')
    plt.ylabel('<S> / <S0>')
    plt.show()


def monte_carlo_sim_p():
    ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    iters = 1000
    temp = []
    for i in tqdm(range(len(ps)), desc="p"):
        tval = 0.0
        args = get_args()
        args.r = 0.0
        args.p = ps[i]
        args.k = 6
        for j in tqdm(range(iters)):
            pv, mp = simulate(100.0, args)
            tval += mp
        temp.append((tval / iters) / 1.5)
    plt.plot(ps, temp)
    plt.title('p vs. factor increase')
    plt.xlabel('p')
    plt.ylabel('<S> / <S0>')
    plt.show()


def monte_carlo_sim_ttm_rate():
    ttms = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    iters = 1000
    temp = []
    for i in tqdm(range(len(ttms)), desc="ttm"):
        tval = 0.0
        args = get_args()
        args.r = 0.05
        args.p = 0.2
        args.k = 6
        args.ttm = ttms[i]
        for j in tqdm(range(iters)):
            pv, mp = simulate(100.0, args)
            tval += mp
        temp.append((tval / iters) / 1.5)
    plt.plot(ttms, temp)
    plt.title('ttm vs. factor increase')
    plt.xlabel('ttm')
    plt.ylabel('<S> / <S0>')
    plt.show()


def monte_carlo_sim_ttm_free():
    ttms = [2.0, 4.0, 6.0, 8.0, 10.0, 12.]
    iters = 1000
    temp = []
    for i in tqdm(range(len(ttms)), desc="ttm"):
        tval = 0.0
        args = get_args()
        args.r = 0.00
        args.p = 0.2
        args.k = 6
        args.ttm = ttms[i]
        for j in tqdm(range(iters)):
            pv, mp = simulate(100.0, args)
            tval += mp
        temp.append((tval / iters) / 1.5)
    plt.plot(ttms, temp)
    plt.title('ttm vs. factor increase')
    plt.xlabel('ttm')
    plt.ylabel('<S> / <S0>')
    plt.show()


if __name__ == '__main__':
    monte_carlo_sim_ttm_rate()