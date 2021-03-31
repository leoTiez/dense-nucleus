#!/usr/bin/env python3
import multiprocessing
from abc import ABC
from itertools import combinations

import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


class Message:
    def __init__(self, target, update, prob):
        prot_types = Protein.get_types()
        if target not in prot_types or update not in prot_types:
            raise ValueError('Invalid message for protein %s to update interaction with protein %s' % (target, update))
        self.target = target
        self.update = update

        assert 0 <= prob <= 1
        self.prob = prob

    def __ne__(self, other):
        return '%s:%s\tProbability:%s' % (self.target, self.update, self.prob)


class Protein(ABC):
    RAD3 = 'rad3'
    IC_RAD3 = 'ic rad3'
    POL2 = 'pol2'
    IC_POL2 = 'ic pol2'
    RAD26 = 'rad26'
    IC_RAD26 = 'ic rad26'

    def __init__(self, species, p_inter, color, pos_dim=2):
        self.color = color
        self.species = species
        self.position = np.random.random(pos_dim)
        self.p_inter = p_inter
        self.messages = {}

    def set_position_delta(self, delta):
        self.position += delta
        self.position = np.maximum(0, self.position)
        self.position = np.minimum(1, self.position)

    def get_position(self):
        return self.position

    def update_position(self, mag):
        self.position += mag * np.random.uniform(-1, 1, len(self.position))
        self.position = np.maximum(0, self.position)
        self.position = np.minimum(1, self.position)

    def update_p_inter(self, species, p_update):
        print('UPDATE INTERACTION PROBABILITY %s, %s, %s.' % (self.species, species, p_update))
        self.p_inter[species] = p_update

    def interact(self, species):
        tossing = np.random.random()
        return self.p_inter[species] >= tossing

    def add_message(self, target, update, prob):
        if self.species == target:
            self.update_p_inter(update, prob)
        elif self.species == update:
            self.update_p_inter(target, prob)
        else:
            self.messages['%s:%s' % (target, update)] = Message(target, update, prob)

    def add_message_obj(self, m):
        self.add_message(m.target, m.update, m.prob)

    def del_message(self, key):
        self.messages.pop(key)

    def get_features(self):
        c = self.color
        x = self.position[0]
        y = self.position[1]
        edge = 'red' if self.messages.keys() else 'white'
        return x, y, c, edge

    @staticmethod
    def get_types():
        return [Protein.RAD3, Protein.IC_RAD3, Protein.POL2, Protein.IC_POL2, Protein.RAD26, Protein.IC_RAD26]


class Rad3(Protein):
    def __init__(self, p_inter, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .2,
                Protein.IC_RAD3: .5,
                Protein.POL2: .05,
                Protein.IC_POL2: .1,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .1,
            }
        super().__init__(Protein.RAD3, p_inter, 'orange', pos_dim)


class InfoRad3(Protein):
    def __init__(self, p_inter, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .5,
                Protein.IC_RAD3: .5,
                Protein.POL2: .1,
                Protein.IC_POL2: .4,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .4
            }
        # super().__init__(Protein.IC_RAD3, p_inter, 'lightsalmon', pos_dim)
        super().__init__(Protein.IC_RAD3, p_inter, 'grey', pos_dim)


class Pol2(Protein):
    def __init__(self, p_inter, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .1,
                Protein.POL2: .2,
                Protein.IC_POL2: .5,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .1
            }
        super().__init__(Protein.POL2, p_inter, 'green', pos_dim)
        
        
class InfoPol2(Protein):
    def __init__(self, p_inter, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .4,
                Protein.POL2: .5,
                Protein.IC_POL2: .5,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .4
            }
        # super().__init__(Protein.IC_POL2, p_inter, 'springgreen', pos_dim)
        super().__init__(Protein.IC_POL2, p_inter, 'grey', pos_dim)


class Rad26(Protein):
    def __init__(self, p_inter, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .1,
                Protein.POL2: .05,
                Protein.IC_POL2: .1,
                Protein.RAD26: .2,
                Protein.IC_RAD26: .5
            }
        # super().__init__(Protein.RAD26, p_inter, 'blue', pos_dim)
        super().__init__(Protein.RAD26, p_inter, 'grey', pos_dim)


class InfoRad26(Protein):
    def __init__(self, p_inter, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .4,
                Protein.POL2: .1,
                Protein.IC_POL2: .4,
                Protein.RAD26: .5,
                Protein.IC_RAD26: .5
            }
        # super().__init__(Protein.IC_POL2, p_inter, 'lightblue', pos_dim)
        super().__init__(Protein.IC_POL2, p_inter, 'grey', pos_dim)


class ProteinFactory:
    @staticmethod
    def create(prot_type, pos_dim, p_inter=None):
        if prot_type == Protein.RAD3:
            return Rad3(p_inter, pos_dim)
        elif prot_type == Protein.IC_RAD3:
            return InfoRad3(p_inter, pos_dim)
        elif prot_type == Protein.POL2:
            return Pol2(p_inter, pos_dim)
        elif prot_type == Protein.IC_POL2:
            return InfoPol2(p_inter, pos_dim)
        elif prot_type == Protein.RAD26:
            return Rad26(p_inter, pos_dim)
        elif prot_type == Protein.IC_RAD26:
            return InfoRad26(p_inter, pos_dim)
        else:
            raise ValueError('Protein type %s is not supported' % prot_type)


class Nucleus:
    INTERACT_RAD = .01

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

    def update(self):
        adj = np.zeros((len(self.proteins), len(self.proteins)))
        idc = self.state.query_radius(self.pos, r=Nucleus.INTERACT_RAD)
        for num, i in enumerate(idc):
            adj[num, i] = 1
        adj = np.multiply(adj, adj.T)
        interact_graph = nx.from_numpy_matrix(adj)
        pot_inter = sorted(nx.connected_components(interact_graph), reverse=True, key=len)
        success_inter = set()
        for inter_group in pot_inter:
            if len(inter_group) == 1:
                continue
            interactions = combinations(list(inter_group), 2)
            for i, j in interactions:
                if self.proteins[i].interact(self.proteins[j].species):
                    keys = list(self.proteins[i].messages.keys())
                    for k in keys:
                        m = self.proteins[i].messages[k]
                        if self.proteins[j].species == m.target:
                            self.proteins[j].update_p_inter(m.update, m.prob)
                        elif self.proteins[j].species == m.update:
                            self.proteins[j].update_p_inter(m.target, m.prob)
                        else:
                            if k not in self.proteins[j].messages.keys():
                                self.proteins[j].add_message_obj(m)
                        self.proteins[i].del_message(k)
                    keys = list(self.proteins[j].messages.keys())
                    for k in keys:
                        m = self.proteins[j].messages[k]
                        if self.proteins[i].species == m.target:
                            self.proteins[i].update_p_inter(m.update, m.prob)
                        elif self.proteins[i].species == m.update:
                            self.proteins[i].update_p_inter(m.target, m.prob)
                        else:
                            if k not in self.proteins[i].messages.keys():
                                self.proteins[i].add_message_obj(m)
                        self.proteins[j].del_message(k)

                    success_inter.add(i)
                    success_inter.add(j)
        idx = np.arange(len(self.proteins)).astype('int')
        fail_mask = np.ones(idx.size)
        fail_mask[np.asarray(list(success_inter))] = 0.

        [self.proteins[fi].update_position(self.t) for fi in idx[fail_mask.astype('bool')]]
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
        y_all = list(map(lambda x: x[1], results))
        c_all = list(map(lambda x: x[2], results))
        edge_all = list(map(lambda x: x[3], results))

        plt.scatter(x_all, y_all, c=c_all, edgecolors=edge_all)
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

