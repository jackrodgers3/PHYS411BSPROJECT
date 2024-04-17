import networkx as nx
import numpy as np
import pandas as pd
from datetime import date
import argparse
import warnings
from cls import *
warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd', type=date, help='start_date')
    parser.add_argument('--r', type=float, help='short rate')
    parser.add_argument('--sigma', type=float, help='volatility')
    parser.add_argument('--SW_n', type=int, help='number of nodes')
    parser.add_argument('--SW_k', type=int, help='degree of nodes')
    parser.add_argument('--SW_p', type=float, help='rewiring probability')
    parser.add_argument('--nxd', type=str, help='nxdraw format', choices=['kk', 'circ'])
    parser.add_argument('--init_range', type=tuple, help='range for initial stock value')
    parser.add_argument('--dt', type=float, help='incremental time step')
    parser.add_argument('--maturity_time', type=float, help='maturity time for options')
    parser.add_argument('--otype', type=str, help='option type', choices = ['call', 'put'])
    parser.add_argument('--otype2', type=str, help='option type 2', choices=['digital', 'euro'])
    parser.add_argument('--Nprec', type=float, help = 'precision for partial derivative to get N')

    parser.set_defaults(
        r = 0.01,
        sigma = 0.05,
        sd = date(2024, 4, 22),
        n = 20,
        k = 2,
        p = 0.9,
        nxd = 'circ',
        init_range = (1.0, 3.0),
        dt = 1.0,
        maturity_time = 10.0,
        otype = 'call',
        otype2 = 'euro',
        Nprec = 0.1
    )
    return parser.parse_args()


def run_simulation(tmax: int):
    args = get_args()
    mkt = Market(args.r, args.sigma, args.n, args.k, args.p, args.maturity_time, args.otype, args.otype2, args.Nprec)
    mkt.create(args.init_range)
    mkt.draw_graph(args.nxd)
    t = 0
    while t < tmax:
        mkt.update(t, args.dt)
        t += args.dt
    mkt.show_stock_history()


if __name__ == '__main__':
    run_simulation(100)