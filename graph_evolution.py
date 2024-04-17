import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random as r


class DigitalOption:
    def __init__(self, spot_price, mu, time_to_maturity, owner_id, seller_id):
        self.spot_price = spot_price
        self.time_to_maturity = time_to_maturity
        self.mu = mu
        self.strike_price = math.floor(self.spot_price * math.exp(self.mu*self.time_to_maturity))
        self.owner_id = owner_id
        self.seller_id = seller_id

    def eval_option(self, price_at_t):
        if price_at_t < self.strike_price:
            return self.strike_price - price_at_t
        else:
            return -1

    def get_strike_price(self):
        return self.strike_price

    def get_owner_id(self):
        return self.owner_id

    def get_seller_id(self):
        return self.seller_id

    def get_ttm(self):
        return self.time_to_maturity


class Economy:
    def __init__(self, n, k, p):
        self.G = None
        self.n = n
        self.k = k
        self.p = p
        self.S_tracker = []
        self.neighbor_tracker = []

    def create(self):
        self.G = nx.watts_strogatz_graph(self.n, self.k, self.p)
        initS = []
        for i in range(self.n):
            self.G.nodes[i]['S'] = r.randint(5, 20)
            self.G.nodes[i]['V'] = 0
            self.G.nodes[i]['num_shares'] = 100
            self.G.nodes[i]['options'] = []
            initS.append(self.G.nodes[i]['S'])
        self.S_tracker.append(initS)
        for i in range(self.n):
            self.neighbor_tracker.append([n for n in self.G.neighbors(i)])
        return self.G.nodes.data()

    def draw_graph(self, type):
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

    def update_stocks(self, se_params):
        int_S = []
        for i in range(self.n):
            self.G.nodes[i]['S'] += ((se_params[0]*self.G.nodes[i]['S']) + (se_params[1]*self.G.nodes[i]['S']*r.gauss(0.0, 1.0)))
            int_S.append(self.G.nodes[i]['S'])
        self.S_tracker.append(int_S)
        return self.G.nodes.data()

    def get_stock_history(self):
        return self.S_tracker

    def get_neighbors(self):
        return self.neighbor_tracker

    def assign_options(self):
        for i in range(self.n):
            for j in range(len(self.neighbor_tracker[i])):
                # get neighboring node
                neighbor_node = self.neighbor_tracker[i][j]
                # owner buys option
                self.G.nodes[i]['S'] -= 1
                # seller sells option
                self.G.nodes[neighbor_node]['S'] += 1
                # acquire options
                self.G.nodes[i]['options'].append(DigitalOption(self.G.nodes[neighbor_node]['S'], 0.01, 10, i, neighbor_node))

    def evaluate_options(self):
        for i in range(self.n):
            for option in self.G.nodes[i]['options']:
                a = option.eval_option(self.G.nodes[option.get_seller_id()]['S'])
                self.G.nodes[i]['S'] += a
            self.G.nodes[i]['options'] = []
        self.assign_options()


def evolve_system(sw_params, se_params, tmax, give_info = False):
    """
    :param sw_params: strogatz-watts graph params (n, k, p)
    :param se_params: stock evolution params (mu, sigma)
    :param tmax: maximum time of simulation
    :param give_info: display graph info to terminal
    :return: none
    """
    if give_info:
        tlist = [0]
    t = 0
    dt = 1
    economy = Economy(sw_params[0], sw_params[1], sw_params[2])
    g_data = economy.create()
    assign = True
    gl, eigs = economy.get_gl(get_eigs=True)
    if give_info:
        plt.plot(eigs, 'ro')
        plt.title("Eigenvalue distribution")
        plt.xlabel("Node Number")
        plt.ylabel("Value")
        plt.show()
        plt.clf()
        plt.cla()
    if give_info:
        economy.draw_graph('kk')
    while t < tmax:
        if t == 0:
            economy.assign_options()
        elif t > 0 and t % 10 == 0:
            economy.evaluate_options()
        g_data = economy.update_stocks(se_params)
        if give_info:
            tlist.append(t)
        t += dt
    if give_info:
        s_history = np.array(economy.get_stock_history()).transpose()
        for i in range(len(s_history)):
            plt.plot(s_history[i])
        plt.title("Stock value over time")
        plt.xlabel("Time (financial quarter)")
        plt.ylabel("Value ($)")
        if sw_params[0] > 20:
            plt.yscale('log')
        plt.show()
        plt.clf()
        plt.cla()
        print(economy.get_neighbors())


if __name__ == '__main__':
    SW_PARAMS = (20, 2, 0.7)
    SE_PARAMS = (0.01, 0.075)
    evolve_system(SW_PARAMS, SE_PARAMS, tmax=100, give_info=True)
