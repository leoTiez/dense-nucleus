#!/usr/bin/env python3
from abc import ABC
from itertools import combinations
import numpy as np

from modules.abstractClasses import *
from modules.messengers import *


class Protein(AbstractProtein, ABC):
    RAD3 = 'rad3'
    IC_RAD3 = 'ic rad3'
    POL2 = 'pol2'
    IC_POL2 = 'ic pol2'
    RAD26 = 'rad26'
    IC_RAD26 = 'ic rad26'

    def __init__(self, species, p_inter, p_info, color, pos_dim=2):
        self.color = color
        self.species = species
        self.position = np.random.random(pos_dim)
        self.p_inter = p_inter
        self.p_info = p_info
        self.is_associated = False
        self.messages = {}

    def _tossing(self, p, prob_type='inter'):
        species = p.species
        toss = np.random.random()
        if prob_type == 'inter':
            if species in self.p_inter.keys():
                return self.p_inter[species] >= toss
            else:
                return False
        elif prob_type == 'info':
            if species in self.p_inter.keys():
                return self.p_info[species] >= toss
            else:
                return False
        else:
            raise ValueError('Probability type %s is not understood' % prob_type)

    def _update_p_inter(self, species, p_update):
        print('UPDATE INTERACTION PROBABILITY %s, %s, %s.' % (self.species, species, p_update))
        self.p_inter[species] = p_update

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

    def interact(self, protein):
        if isinstance(protein, Protein):
            return self._tossing(protein, prob_type='inter')
        elif isinstance(protein, ProteinComplex):
            return all(map(lambda x: self._tossing(x, prob_type='inter'), protein.prot_list))
        elif isinstance(protein, AbstractDNASegment):
            return self._tossing(protein, prob_type='inter')
        else:
            raise ValueError('Passed protein is not of class Protein or Protein Complex')

    def share_info(self, protein):
        if isinstance(protein, Protein):
            return self._tossing(protein, prob_type='info')
        elif isinstance(protein, ProteinComplex):
            return all(map(lambda x: self._tossing(x, prob_type='info'), protein.prot_list))
        else:
            raise ValueError('Passed protein is not of class Protein or Protein Complex')

    def add_message(self, target, update, prob):
        if self.species == target:
            self._update_p_inter(update, prob)
        elif self.species == update:
            self._update_p_inter(target, prob)
        else:
            self.messages['%s:%s' % (target, update)] = Message(target, update, prob)

    def add_message_obj(self, m):
        self.add_message(m.target, m.update, m.prob)

    def del_message(self, key):
        self.messages.pop(key)

    def clear_messages(self):
        self.messages = {}

    def get_message_keys(self):
        return self.messages.keys()

    def broadcast(self, key):
        m = self.messages[key]
        self.del_message(key)
        return m

    @staticmethod
    def is_collapsing():
        return False

    def get_features(self):
        c = self.color
        x = self.position[0]
        y = self.position[1]
        edge = 'limegreen' if self.messages.keys() else 'white'
        return [x], [y], [c], [edge]

    @staticmethod
    def get_types():
        return [Protein.RAD3, Protein.IC_RAD3, Protein.POL2, Protein.IC_POL2, Protein.RAD26, Protein.IC_RAD26]


class ProteinComplex(AbstractProteinComplex):
    def __init__(self, prot_list):
        def flatten_list():
            proteins = []
            for p in prot_list:
                if isinstance(p, ProteinComplex):
                    proteins.extend(p.prot_list)
                elif isinstance(p, Protein):
                    proteins.append(p)
                else:
                    raise ValueError('Not all proteins are valid to form a complex')
            return proteins

        self.prot_list = flatten_list()
        self.position = self.prot_list[0].get_position()
        self.is_associated = self.prot_list[0].is_associated
        self._init_broadcast()

    def _init_broadcast(self):
        temp_mes = {}
        for p in self.prot_list:
            for k, m in p.messages.items():
                if k not in temp_mes.keys():
                    temp_mes[k] = []
                temp_mes[k].append(m)

        keys = temp_mes.keys()
        for k in keys:
            if len(temp_mes[k]) == 1:
                temp_mes[k] = temp_mes[k][0]
            else:
                probs = np.asarray(list(map(lambda x: x.prob, temp_mes[k])))
                unique, counts = np.unique(probs, return_counts=True)
                unique, _ = zip(*sorted(zip(unique, counts), reverse=True, key=lambda x: x[1]))
                temp_mes[k] = Message(temp_mes[k][0].target, temp_mes[k][0].update, unique[0])

        for p in self.prot_list:
            [p.add_message_obj(m) for m in temp_mes.values()]

    def set_position_delta(self, delta):
        [x.set_position_delta(delta) for x in self.prot_list]
        self.position = self.prot_list[0].get_position()

    def get_position(self):
        return self.position

    def update_position(self, mag):
        delta = (1. / float(len(self.prot_list))) * mag * np.random.uniform(-1, 1, len(self.position))
        self.set_position_delta(delta)

    def interact(self, protein):
        return all([x.interact(protein) for x in self.prot_list])

    def share_info(self, species):
        # TODO ANY OR ALL BETTER FOR SHARING INFO?
        return any([x.share_info(species) for x in self.prot_list])

    def add_message(self, target, update, prob):
        [x.add_message(target, update, prob) for x in self.prot_list]

    def add_message_obj(self, m):
        self.add_message(m.target, m.update, m.prob)

    def del_message(self, key):
        try:
            [x.del_message(key) for x in self.prot_list]
        except KeyError:
            print('Different number of messages in complex')
            pass

    def clear_messages(self):
        [x.clear_messages() for x in self.prot_list]

    def get_message_keys(self):
        return self.prot_list[0].messages.keys()

    def is_collapsing(self):
        interactions = combinations(range(len(self.prot_list)), 2)
        results = []
        for i, j in interactions:
            results.append(self.prot_list[i].interact(self.prot_list[j]))
        return not all(results)

    def broadcast(self, key):
        m = self.prot_list[0].messages[key]
        self.del_message(key)
        return m

    def get_features(self):
        c = list(map(lambda p: p.color, self.prot_list))
        x = list(map(lambda p: p.position[0], self.prot_list))
        y = list(map(lambda p: p.position[1], self.prot_list))
        edge = 'blue'
        return x, y, c, len(self.prot_list) * [edge]


class InfoProtein(Protein):
    pass


class Rad3(Protein):
    def __init__(self, p_inter, p_info, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05,
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .3,
                Protein.IC_RAD3: .9,
                Protein.POL2: .1,
                Protein.IC_POL2: .1,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .1,
            }
        super().__init__(Protein.RAD3, p_inter, p_info, 'orange', pos_dim)


class InfoRad3(InfoProtein):
    def __init__(self, p_inter, p_info, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .9,
                Protein.IC_RAD3: .9,
                Protein.POL2: .1,
                Protein.IC_POL2: .9,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .9,
            }
        # super().__init__(Protein.IC_RAD3, p_inter, p_info, 'lightsalmon', pos_dim)
        super().__init__(Protein.IC_RAD3, p_inter, p_info, 'grey', pos_dim)


class Pol2(Protein):
    def __init__(self, p_inter, p_info, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .1,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05
            }

        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .1,
                Protein.POL2: .3,
                Protein.IC_POL2: .9,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .1,
            }
        super().__init__(Protein.POL2, p_inter, p_info, 'green', pos_dim)
        # super().__init__(Protein.POL2, p_inter, p_info, 'grey', pos_dim)


class InfoPol2(InfoProtein):
    def __init__(self, p_inter, p_info, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .9,
                Protein.POL2: .9,
                Protein.IC_POL2: .9,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .9,
            }
        # super().__init__(Protein.IC_POL2, p_inter, p_info, 'springgreen', pos_dim)
        super().__init__(Protein.IC_POL2, p_inter, p_info, 'grey', pos_dim)


class Rad26(Protein):
    def __init__(self, p_inter, p_info, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .1
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .1,
                Protein.POL2: .1,
                Protein.IC_POL2: .1,
                Protein.RAD26: .3,
                Protein.IC_RAD26: .9,
            }
        # super().__init__(Protein.RAD26, p_inter, p_info, 'cyan', pos_dim)
        super().__init__(Protein.RAD26, p_inter, p_info, 'grey', pos_dim)


class InfoRad26(InfoProtein):
    def __init__(self, p_inter, p_info, pos_dim):
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05
            }

        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .9,
                Protein.POL2: .1,
                Protein.IC_POL2: .9,
                Protein.RAD26: .9,
                Protein.IC_RAD26: .9,
            }
        # super().__init__(Protein.IC_POL2, p_inter, p_info, 'lightblue', pos_dim)
        super().__init__(Protein.IC_POL2, p_inter, p_info, 'grey', pos_dim)


class ProteinFactory:
    @staticmethod
    def create(prot_type, pos_dim, p_inter=None, p_info=None):
        if prot_type == Protein.RAD3:
            return Rad3(p_inter, p_info, pos_dim)
        elif prot_type == Protein.IC_RAD3:
            return InfoRad3(p_inter, p_info, pos_dim)
        elif prot_type == Protein.POL2:
            return Pol2(p_inter, p_info, pos_dim)
        elif prot_type == Protein.IC_POL2:
            return InfoPol2(p_inter, p_info, pos_dim)
        elif prot_type == Protein.RAD26:
            return Rad26(p_inter, p_info, pos_dim)
        elif prot_type == Protein.IC_RAD26:
            return InfoRad26(p_inter, p_info, pos_dim)
        else:
            raise ValueError('Protein type %s is not supported' % prot_type)

