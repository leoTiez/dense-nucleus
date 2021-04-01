#!/usr/bin/env python3
import multiprocessing
from itertools import combinations

import networkx as nx
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from modules.proteins import *


class Nucleus:
    INTERACT_RAD = .015

    def __init__(self, num_proteins, pos_dim=2, t=.2):
        plt.ion()
        prot_types = Protein.get_types()
        assert len(prot_types) == len(num_proteins)
        self.proteins = [ProteinFactory.create(t_prot, pos_dim)
                         for t_prot, n in zip(prot_types, num_proteins) for _ in range(int(n))]

        self.pos_dim = pos_dim
        assert 0. <= t <= 1.
        self.t = t

        self.pos = []
        self.state = None
        self._fetch_pos()

    def _fetch_pos(self):
        with multiprocessing.Pool(np.maximum(multiprocessing.cpu_count() - 1, 1)) as parallel:
            results = []
            for p in self.proteins:
                res = parallel.apply_async(p.get_position, args=())
                results.append(res)
            parallel.close()
            parallel.join()
            self.pos = [r.get() for r in results]
        self.state = KDTree(np.asarray(self.pos))

    def global_event(self, target, update, prob):
        [x.add_message(target, update, prob) for x in self.proteins]

    def global_event_obj(self, m):
        [x.add_message_obj(m) for x in self.proteins]

    def update(self):
        def handshake(x, y, k):
            m = self.proteins[x].broadcast(k)
            self.proteins[y].add_message_obj(m)

        collapse_mask = np.asarray([x.is_collapsing() for x in self.proteins])
        idx = np.arange(len(self.proteins))
        rev_idx = np.flip(idx[collapse_mask])
        for i in rev_idx:
            self.proteins.extend(self.proteins[i].prot_list)
            del self.proteins[i]

        adj = np.zeros((len(self.proteins), len(self.proteins)))
        idc = self.state.query_radius(self.pos, r=Nucleus.INTERACT_RAD)
        for num, i in enumerate(idc):
            adj[num, i] = 1
        adj = np.multiply(adj, adj.T)
        interact_graph = nx.from_numpy_matrix(adj)
        pot_inter = sorted(nx.connected_components(interact_graph), reverse=True, key=len)

        complexes = []
        success_inter = set()
        for inter_group in pot_inter:
            if len(inter_group) == 1:
                continue
            interactions = list(combinations(list(inter_group), 2))
            success_deliver = []
            for i, j in interactions:
                if self.proteins[i].share_info(self.proteins[j]) and i not in success_deliver:
                    success_deliver.append(i)
                    keys_i = list(self.proteins[i].get_message_keys())
                    for k_i in keys_i:
                        handshake(i, j, k_i)
                if self.proteins[j].share_info(self.proteins[i]) and j not in success_deliver:
                    success_deliver.append(i)
                    keys_j = list(self.proteins[j].get_message_keys())
                    for k_j in keys_j:
                        handshake(j, i, k_j)

            for i, j in interactions:
                if self.proteins[i].interact(self.proteins[j]) or self.proteins[j].interact(self.proteins[i]):
                    if i in success_inter or j in success_inter:
                        continue
                    complexes.append(ProteinComplex([self.proteins[i], self.proteins[j]]))
                    success_inter.add(i)
                    success_inter.add(j)

        self.proteins.extend(complexes)
        success_inter = sorted(list(success_inter), reverse=True)
        for si in success_inter:
            del self.proteins[si]
        [x.update_position(self.t) for x in self.proteins]
        self._fetch_pos()

    def display(self):
        results = []
        with multiprocessing.Pool(np.maximum(multiprocessing.cpu_count() - 1, 1)) as parallel:
            for p in self.proteins:
                res = parallel.apply_async(p.get_features, args=())
                results.append(res)
            parallel.close()
            parallel.join()
            results = [r.get() for r in results]

        x_all = list(map(lambda x: x[0], results))
        x_all = [x for x_group in x_all for x in x_group]
        y_all = list(map(lambda x: x[1], results))
        y_all = [y for y_group in y_all for y in y_group]
        c_all = list(map(lambda x: x[2], results))
        c_all = [c for c_group in c_all for c in c_group]
        edge_all = list(map(lambda x: x[3], results))
        edge_all = [edge for edge_group in edge_all for edge in edge_group]

        plt.scatter(x_all, y_all, s=1./Nucleus.INTERACT_RAD, c=c_all, edgecolors=edge_all)
        figure = plt.gcf()
        figure.canvas.flush_events()
        figure.canvas.draw()
        plt.cla()


def main():
    num_proteins = len(Protein.get_types())
    nucleus = Nucleus(200 * np.ones(num_proteins), t=.02)
    idx = np.random.choice(1200, size=200)
    [nucleus.proteins[i].add_message(target=Protein.RAD3, update=Protein.POL2, prob=1.) for i in idx]
    for _ in range(200):
        nucleus.update()
        nucleus.display()


if __name__ == '__main__':
    main()

