#!/usr/bin/env python3
from collections.abc import Iterable
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from modules.proteins import Protein
import matplotlib.pyplot as plt
from datahandler.seqDataHandler import smooth

DEFAULT_DNA_SPEC = {
    'range': (0, 100, 50, 51),
    'cp': [(0, 20, 50, 51)],
    'tss': [(20, 30, 50, 51)],
    'tts': [(90, 100, 50, 51)]
}

LENGTH = 100
DEFAULT_DNA_SPEC_1DIM = {
    'cp': [0, 10],
    'tss': [10, 25],  # Set TSS into transcript
    'transcript': [15, 90],
    'tts': [90, LENGTH]
}

DEFAULT_CPD_LENGTH = 5
BACKTRACKING_LENGTH = 5
DEFAULT_CPD = [30, 30 + DEFAULT_CPD_LENGTH]

CPD_STATES = {
    'new': 0,
    'recognised': 1,
    'opened': 2,
    'incised': 3,
    'replaced': 4,
    'removed': 5
}

ACTIVE_POL2 = 'active %s' % Protein.POL2


class Lesion:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.state = CPD_STATES['new']

    def update_state_to(self, new_state):
        self.state = CPD_STATES[new_state]


class Rule:
    def __init__(self, reactants, products, c):
        self.reactants = reactants
        self.products = products
        self.c = c

    def react(self, elements):
        if self.is_reacting(elements):
            return self.products

    def is_reacting(self, elements):
        return np.all([i in self.reactants for i in elements])

    def is_involved(self, element):
        return element in self.reactants


class Gillespie(ABC):
    @abstractmethod
    def h(self, reactants):
        pass

    @abstractmethod
    def reaction_prob(self):
        pass

    @abstractmethod
    def simulate(self):
        pass


class PoolGillespie(Gillespie):
    def __init__(self, protein_conc, rules):
        self.state = np.asarray(list(protein_conc.values()))
        self.prot_idx = {prot: num for num, prot in enumerate(protein_conc.keys())}
        self.rules = rules
        self.t = 0
        self.a = np.zeros(len(rules))

        self.protein_rule_map = {}
        for protein in self.prot_idx.keys():
            self.protein_rule_map[protein] = [i for i, r in enumerate(rules)
                                              if protein in r.reactants or protein in r.products]
        self.reaction_prob(range(len(rules)))

    def h(self, reactants):
        if len(reactants) == 0:
            return 1
        elif len(reactants) == 1:
            return self.state[self.prot_idx[reactants[0]]]
        elif len(reactants) == 2:
            if reactants[0] == reactants[1]:
                return .5 * self.state[self.prot_idx[reactants[0]]] * (self.state[self.prot_idx[reactants[0]]] - 1)
            else:
                return self.state[self.prot_idx[reactants[0]]] * self.state[self.prot_idx[reactants[1]]]
        else:
            raise ValueError('Reactions with more than 2 reactants not supported')

    def reaction_prob(self, update_idx):
        for i in update_idx:
            reactants = self.rules[i].reactants
            self.a[i] = self.h(reactants) * self.rules[i].c

    def spec_state(self, reactant):
        return self.state[self.prot_idx[reactant]]

    def _change_reactant_delta(self, reactant, delta):
        self.state[self.prot_idx[reactant]] = np.maximum(0, self.state[self.prot_idx[reactant]] + delta)

    def reduce(self, reactant):
        self._change_reactant_delta(reactant, -1)

    def increase(self, reactant):
        self._change_reactant_delta(reactant, +1)

    def simulate(self):
        r1, r2 = np.random.random(2)
        a0 = np.sum(self.a)
        if a0 == 0:
            return 0
        tau = 1./a0 * np.log(1./r1)
        self.t += tau
        mu = np.searchsorted([np.sum(self.a[:i]) for i in range(1, self.a.size + 1)], a0 * r2)
        reactants = self.rules[mu].reactants
        products = self.rules[mu].products

        update_idx = []
        for r in reactants:
            update_idx.extend(self.protein_rule_map[r])
            self.state[self.prot_idx[r]] = np.maximum(0, self.state[self.prot_idx[r]] - 1)
        for p in products:
            update_idx.extend(self.protein_rule_map[p])
            self.state[self.prot_idx[p]] += 1

        self.reaction_prob(list(set(update_idx)))
        return tau


class DNAGillespie(Gillespie):
    def __init__(
            self,
            gille_pool,
            size=100,
            protein_names=Protein.get_types_gillespie(),
            dna_spec=DEFAULT_DNA_SPEC_1DIM,
            rules=[],
            elong_speed=1200
    ):
        self.gille_pool = gille_pool
        self.size = size
        self.protein_names = protein_names
        num_species = len(self.protein_names)

        # One-dim case
        if isinstance(size, int):
            self.state = np.zeros((size, num_species))
        # Two-dim case. This is in experimental state and hasn't been properly tested yet
        elif isinstance(size, Iterable):
            state_dim = list(size)
            state_dim.append(num_species)
            self.state = np.zeros(tuple(state_dim))

        self.dna_spec = dna_spec
        self.rules = None
        self.elong_speed = None
        self.a = None
        self.set_rules(rules, elong_speed)

        self.t = 0
        self.lesions = []

        self.protein_to_idx = {prot: num for num, prot in enumerate(self.protein_names)}
        self.reaction_prob()

    def set_rules(self, rules, elong_speed):
        self.rules = rules
        self.elong_speed = elong_speed
        self.a = [np.zeros(len(r)) for r in self.rules]

    def add_lesion(self, start, end):
        self.lesions.append(Lesion(start, end))

    def _determine_dna_idx(self, dna_react='', dna_prod='', proteins=None):
        area = []
        p_idc = np.asarray([self.protein_to_idx[p.strip('!')] for p in proteins]) if proteins is not None else None

        if dna_react == '':
            if dna_prod == '':
                raise ValueError('DNA string must be either passed as reactant or product.')
            is_reacting = False
            dna_string_org = dna_prod
        else:
            is_reacting = True
            dna_string_org = dna_react

        dna_string = dna_string_org.strip('!')
        must_free = '!' == dna_string_org[0] or not is_reacting
        if 'dna' == dna_string.lower():
            # If reactant dna string is equal to 'dna', protein associates from pool and therefore needs to find
            # free position
            if not must_free:
                interact_mask = np.ones(self.state.shape[0]).astype('bool')
                for p_i, prot in zip(p_idc, proteins):
                    if prot[0] == '!':
                        interact_mask = np.logical_and(interact_mask, self.state[:, p_i] < 1)
                    else:
                        interact_mask = np.logical_and(interact_mask, self.state[:, p_i] > 0)
            else:
                interact_mask = np.all(self.state[:, p_idc] < 1, axis=1)
            return np.arange(self.size)[interact_mask] if p_idc is not None else np.arange(self.size)

        split = dna_string.split('_')
        if len(split) == 1:
            if split[0] == 'lesion':
                if not self.lesions:
                    return []
                else:
                    area = []
                    for cpd in self.lesions:
                        area.extend(list(range(cpd.start, cpd.end)))
                    return area
            else:
                raise ValueError('DNA segment is neither DNA nor lesion')
        dna_type = split[0]
        pos = split[1]

        try:
            # Concrete position
            pos = int(pos)
            if p_idc is not None:
                if not must_free:
                    interact_mask = np.ones(self.state.shape[0]).astype('bool')
                    for p_i, prot in zip(p_idc, proteins):
                        if prot[0] == '!':
                            interact_mask = np.logical_and(interact_mask, self.state[pos, p_i] < 1)
                        else:
                            interact_mask = np.logical_and(interact_mask, self.state[pos, p_i] > 0)
                else:
                    interact_mask = np.all(self.state[pos, p_idc] < 1)
                area = [pos] if interact_mask else []
            else:
                area = [pos]
            return area
        except ValueError:
            pass
        if not area:
            border_start, border_end = [], []
            if dna_type == 'lesion':
                if not self.lesions:
                    pass
                else:
                    border_range = [(cpd.start, cpd.end) for cpd in self.lesions if CPD_STATES[pos] == cpd.state]
                    if border_range:
                        border_start, border_end = zip(*border_range)
            else:
                for key in self.dna_spec.keys():
                    if key.lower() in dna_string.lower():
                        if not self.dna_spec[key]:
                            break
                        border_start = self.dna_spec[key][::2]
                        border_end = self.dna_spec[key][1::2]
                        break

            for s, e in zip(border_start, border_end):
                interact_mask = np.zeros(self.state.shape[0]).astype('bool')
                interact_mask[s:e] = True
                if proteins is not None:
                    if not must_free:
                        for p_i, prot in zip(p_idc, proteins):
                            if prot[0] == '!':
                                interact_mask = np.logical_and(
                                    interact_mask,
                                    self.state[:, p_i] < 1
                                )
                            else:
                                interact_mask = np.logical_and(
                                    interact_mask,
                                    self.state[:, p_i] > 0
                                )
                    else:
                        interact_mask = np.logical_and(interact_mask, np.all(self.state[:, p_idc] < 1, axis=1))

                area_range = np.arange(self.state.shape[0])[interact_mask]
                area.extend(area_range)

        return list(np.unique(area))

    def determine_dna_idx_react(self, dna_string, proteins=None):
        return self._determine_dna_idx(dna_react=dna_string, dna_prod='', proteins=proteins)

    def determine_dna_idx_prod(self, dna_string, proteins):
        return self._determine_dna_idx(dna_react='', dna_prod=dna_string, proteins=proteins)

    def get_reacting_protein(self, reactant_org):
        # reactant = reactant_org.strip('!')
        reactant = reactant_org
        split = reactant.split('_')
        proteins = [protein for protein in split if protein.strip('!') in self.protein_names]
        return proteins if proteins else None

    def get_reacting_dna(self, reactant_org):
        reactant = reactant_org.strip('!')
        free_prefix = '!' if '!' == reactant_org[0] else ''
        if 'dna' not in reactant and 'lesion' not in reactant:
            return
        split = reactant.split('_')
        if split[0] == 'lesion':
            lesion_type = self.get_reacting_lesion(reactant)
            if lesion_type:
                return '%slesion_%s' % (free_prefix, lesion_type)
            else:
                return '%slesion' % free_prefix

        try:
            pos = int(split[1])
            return '%sdna_%s' % (free_prefix, split[1])
        except ValueError:
            if split[1] not in self.dna_spec.keys():
                return '%sdna' % free_prefix
        except IndexError:
            # Know that there's nothing after dna because of IndexError
            if split[0] == 'dna':
                return '%sdna' % free_prefix

        for key in self.dna_spec.keys():
            if key.lower() in reactant.lower():
                return '%sdna_%s' % (free_prefix, key)

    def get_reacting_lesion(self, reactants):
        split = reactants.split('_')
        try:
            if split[1] in self.protein_names:
                return ''
            else:
                return split[1]
        except IndexError:
            return ''

    def h(self, reactants, is_elong=False):
        dna_strings = [r for r in reactants if 'dna' in r.lower() or 'lesion' in r.lower()]
        reactant_strings = [r for r in reactants
                            if 'dna' not in r.lower() and 'lesion' not in r.lower() and '!' not in r.lower()]

        dna_react = 1
        for dna_string in dna_strings:
            proteins = self.get_reacting_protein(dna_string)
            dna_area = self.determine_dna_idx_react(self.get_reacting_dna(dna_string), proteins=proteins)
            if dna_area is None:
                continue
            if proteins is not None and len(dna_area) > 0 and '!' != dna_string[0]:
                p_idc = [self.protein_to_idx[p.strip('!')] for p in proteins]
                dna_react *= np.sum(self.state[np.asarray(dna_area)][:, np.asarray(p_idc)])
            else:
                dna_react *= len(dna_area)
            if dna_react == 0:
                return 0.
        return self.gille_pool.h(reactant_strings) * dna_react

    def reaction_prob(self):
        for r in range(len(self.rules)):
            for i in range(len(self.rules[r])):
                reactants = self.rules[r][i].reactants
                self.a[r][i] = self.h(reactants, is_elong=False) * self.rules[r][i].c

    def _sample_reaction(self, rs_idx=0):
        a = self.a[rs_idx]
        a0 = np.sum(a)
        if a0 == 0:
            return np.inf, -1
        r1, r2 = np.random.random(2)
        tau = 1./a0 * np.log(1./r1)
        mu = np.searchsorted([np.sum(a[:i]) for i in range(1, a.size + 1)], a0 * r2)
        return tau, mu

    def _update(self, mu, rs_idx):
        reactants = self.rules[rs_idx][mu].reactants
        products = self.rules[rs_idx][mu].products
        lesion_inter = []
        dna_interact_dict = {}
        for r in reactants:
            if 'dna' in r.lower() or 'lesion' in r.lower():
                if r[0] == '!':
                    continue

                proteins = self.get_reacting_protein(r)
                dna_seg = self.get_reacting_dna(r)
                dna_interact_dict[dna_seg] = {}
                area = self.determine_dna_idx_react(dna_seg, proteins=proteins)
                pos = np.random.choice(area)
                if 'lesion' in r.lower():
                    lesion_inter.extend([cpd for cpd in self.lesions if cpd.start <= pos < cpd.end])

                if proteins is not None:
                    for p in proteins:
                        if '!' != p[0]:
                            dna_interact_dict[dna_seg][p.strip('!')] = pos
                            self.state[pos, self.protein_to_idx[p.strip('!')]] -= 1
            else:
                self.gille_pool.reduce(r)

        lesion_inter = list(set(lesion_inter))
        new_lesion_state = []
        for p in products:
            if 'dna' in p.lower():
                proteins = self.get_reacting_protein(p)
                if proteins is not None:
                    if 'before' not in p.lower():
                        dna_seg = self.get_reacting_dna(p)
                        area = self.determine_dna_idx_prod(dna_seg, proteins=proteins)
                    else:
                        dna_seg = ''
                        area = []
                        for cpd in lesion_inter:
                            area_backtrack = []
                            for prot in proteins:
                                if prot == Protein.POL2:
                                    a = np.where(
                                        np.logical_and(
                                            self.state[:, self.protein_to_idx[ACTIVE_POL2]] < 1,
                                            self.state[:, self.protein_to_idx[Protein.POL2]] < 1,
                                        )
                                    )[0]
                                else:
                                    a = np.where(self.state[:, self.protein_to_idx[prot]] < 1)[0]
                                a = a[np.logical_and(cpd.start - BACKTRACKING_LENGTH <= a, a < cpd.start)]
                                area_backtrack.extend(a.tolist())
                            area.extend(np.unique(area_backtrack).tolist())
                        # Degradation. Can only happen when backtracking is not possible
                        if not area:
                            continue
                    pos = np.random.choice(area)
                    for prot in proteins:
                        if dna_seg in dna_interact_dict:
                            # if prot in dna_interact_dict[dna_seg]:
                            #     self.state[
                            #         dna_interact_dict[dna_seg][prot],
                            #         self.protein_to_idx[prot]
                            #     ] += 1
                            #     continue
                            update_mask = [prot_temp in dna_interact_dict[dna_seg] for prot_temp in proteins]
                            if any(update_mask):
                                update_idx = np.where(update_mask)[0]
                                self.state[
                                    dna_interact_dict[dna_seg][proteins[update_idx[0]]],
                                    self.protein_to_idx[prot]
                                ] += 1
                                continue
                        self.state[pos, self.protein_to_idx[prot.strip('!')]] += 1

            elif 'lesion' in p.lower():
                proteins = self.get_reacting_protein(p)
                lesion_state = self.get_reacting_lesion(p)
                prev_lesion_state = ''
                for cpd in lesion_inter:
                    prev_lesion_state = 'lesion_%s' % list(CPD_STATES.keys())[cpd.state]
                    new_lesion_state.append(lesion_state)

                if proteins is not None:
                    for prot in proteins:
                        if prev_lesion_state in dna_interact_dict:
                            if prot in dna_interact_dict[prev_lesion_state]:
                                self.state[
                                    dna_interact_dict[prev_lesion_state][prot.strip('!')],
                                    self.protein_to_idx[prot.strip('!')]
                                ] += 1
                                continue

                        if prev_lesion_state == '':
                            prev_lesion_state = 'lesion_%s' % lesion_state
                        area = self.determine_dna_idx_prod(prev_lesion_state, proteins=proteins)
                        pos = np.random.choice(area)
                        self.state[pos, self.protein_to_idx[prot]] += 1

            else:
                self.gille_pool.increase(p)

        # Consider only the first new lesion states, as it's assumed that they're all the same for all products
        for lesion_state, cpd in zip(new_lesion_state, lesion_inter):
            cpd.update_state_to(lesion_state)

        self.reaction_prob()

    def simulate(self, max_iter=10, random_power=5):
        # TODO REMOVE
        def update_rad26(s=None, e=None):
            s = s if s is not None else self.dna_spec['transcript'][0]
            e = e if e is not None else self.dna_spec['transcript'][1]
            rad26_idc = np.where(
                np.logical_and(
                    np.logical_and(self.state[:, p_idx] > 0, np.sum(transcript_mask, axis=1)),
                    self.state[:, rad26_idx] > 0
                )
            )[0]
            free_neighbour = rad26_idc[self.state[rad26_idc + 1, rad26_idx] == 0]
            free_neighbour = free_neighbour[np.logical_and(s <= free_neighbour, free_neighbour < e)]
            if np.isin(e-1, rad26_idc) and not np.isin(e-1, free_neighbour):
                free_neighbour = np.append(free_neighbour, [e-1])
            self.state[free_neighbour, rad26_idx] -= 1
            free_neighbour = free_neighbour[free_neighbour + 1 < e]
            self.state[free_neighbour + 1, rad26_idx] += 1

        def update_pol2(s=None, e=None):
            s = s if s is not None else self.dna_spec['transcript'][0]
            e = e if e is not None else self.dna_spec['transcript'][1]
            inactive_pol2 = self.protein_to_idx[Protein.POL2]
            pol2_mask = np.where(
                np.logical_and(
                    self.state[:, p_idx] == 1,
                    self.state[:, p_idx] + np.roll(self.state[:, inactive_pol2], shift=-1) < 2
                )
            )[0]
            pol2_mask = pol2_mask[np.logical_and(s <= pol2_mask, pol2_mask < e)]

            self.state[pol2_mask, p_idx] -= 1
            pol2_mask = pol2_mask[pol2_mask + 1 < e]
            self.state[pol2_mask + 1, p_idx] += 1
            # Handling of stacked active Pol2 that got blocked by inactive Pol2
            # See Saeki 2009
            if np.any(self.state[:, p_idx] > 1):
                stack_idx = np.where(self.state[:, p_idx] > 1)[0]
                self.state[stack_idx, p_idx] -= 1
                for si in stack_idx:
                    head_collision = np.arange(si, 0, -1)[np.argmin(self.state[si::-1, p_idx])]
                    self.state[head_collision, p_idx] += 1

        tau_pool = self.gille_pool.simulate()
        tau_count = np.zeros(len(self.rules))
        for rs_idx in range(len(self.rules)):
            tau, mu = self._sample_reaction(rs_idx=rs_idx)
            if mu == -1:
                continue
            tau_count[rs_idx] += tau
            self._update(mu, rs_idx)

        # Run until no more reactions in the longest reaction time possible
        max_tau = np.maximum(np.max(tau_count), tau_pool)
        for rs_idx in range(len(self.rules)):
            counter = 0
            while counter < max_iter:
                counter += 1
                tau, mu = self._sample_reaction(rs_idx=rs_idx)
                if tau_count[rs_idx] + tau > max_tau:
                    break
                tau_count[rs_idx] += tau
                self._update(mu, rs_idx)

        p_idx = self.protein_to_idx[ACTIVE_POL2]
        rad26_idx = self.protein_to_idx[Protein.RAD26]
        transcript_mask = np.zeros(self.state.shape)
        transcript_mask[self.dna_spec['tss'][0]:self.dna_spec['transcript'][1], p_idx] = 1
        transcript_mask = transcript_mask.astype('bool')

        elong_update = int(self.elong_speed * max_tau)
        while elong_update > 0:
            upd_step = self.state[transcript_mask].sum()
            if upd_step == 0:
                break
            elong_update -= upd_step

            if not self.lesions:
                update_pol2()
            else:
                start = self.dna_spec['tss'][0]
                for cpd in self.lesions:
                    end = cpd.start
                    if cpd.state == CPD_STATES['new']:
                        if self.state[end - 1:cpd.end, p_idx].sum() > 0:
                            cpd.update_state_to('recognised')
                        else:
                            continue

                    last_idx = np.arange(start, end)[::-1][np.argmin(self.state[start:end][:, p_idx][::-1])]
                    update_pol2(start, last_idx)
                    start = cpd.end
                end = self.dna_spec['transcript'][1]
                update_pol2(start, end)

        self.reaction_prob()

        self.t += max_tau
        return max_tau


def routine_gille_pool():
    rules = [
        Rule(
            reactants=[Protein.POL2, Protein.RAD26],
            products=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=.7
        ),
        Rule(
            reactants=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=[Protein.POL2, Protein.RAD26],
            c=.4
        )
    ]

    concentrations = {
        Protein.RAD3: 1000,
        Protein.POL2: 1000,
        Protein.RAD26: 1000,
        Protein.IC_RAD3: 500,
        Protein.IC_POL2: 500,
        Protein.IC_RAD26: 500,
        '_'.join(sorted([Protein.POL2, Protein.RAD26])): 50
    }

    gille = PoolGillespie(protein_conc=concentrations, rules=rules)

    pol2 = []
    complex = []
    for _ in range(2000):
        print(gille.state)
        pol2.append(gille.state[1])
        complex.append(gille.state[-1])
        print(gille.t)
        print('\n')
        gille.simulate()

    plt.plot(pol2, label='Pol2')
    plt.plot(complex, label='Pol2:Rad26')
    plt.legend(loc='upper right')
    plt.xlabel('Update Steps')
    plt.ylabel('#Proteins')
    plt.title('Protein Evolution after 1.5 sec')
    plt.show()


def routine_gille_dna():
    def plot():
        plt.plot(smooth(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD3]], 3), label='Rad3', color='tab:orange')
        plt.plot(smooth(
            gille_dna.state[:, gille_dna.protein_to_idx[Protein.POL2]]
            + gille_dna.state[:, gille_dna.protein_to_idx[ACTIVE_POL2]],
            3), label='Pol2', color='tab:green')
        plt.plot(smooth(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD26]], 3), label='Rad26', color='tab:cyan')
        plt.plot(smooth(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD4]], 3), label='Rad4', color='tab:red')
        plt.xlabel('DNA Position')
        plt.ylabel('#Molecules')
        plt.title('Smoothed ChIP-seq Simulation')
        plt.legend(loc='upper right')
        plt.show()

    # Pivotal definitions
    # Time unit = minute
    # Considering genes with high transcription rate
    gille_proteins = Protein.get_types_gillespie()
    elong_speed = 1200
    num_prot = 1e5
    chip_norm = 1e5
    random_chip = 2.6
    rad3_cp_chip = 6.8 * 2

    pol2_trans_chip = 7.01 * 2
    rad26_trans_chip = 3.1 * 2
    disso_const = 10.

    random_asso = random_chip / (chip_norm * LENGTH)
    random_disso = disso_const / float(LENGTH)
    rad3_cp_asso = rad3_cp_chip / float(chip_norm * (DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]))
    rad3_cp_disso = disso_const / (DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0])

    pol2_trans_c = pol2_trans_chip / float(
        chip_norm
        * (
                DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]
                + DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0]
        )
    )
    pol2_disso = disso_const / (DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0])

    rad26_asso = rad26_trans_chip / (
            chip_norm * (DEFAULT_DNA_SPEC_1DIM['transcript'][1] - DEFAULT_DNA_SPEC_1DIM['transcript'][0])
        )
    rad26_disso = disso_const / (DEFAULT_DNA_SPEC_1DIM['transcript'][1] - DEFAULT_DNA_SPEC_1DIM['transcript'][0])

    rules_pool = []
    concentrations_pool = {gp: num_prot for gp in gille_proteins}
    gille_proteins_elong = deepcopy(gille_proteins)
    gille_proteins_elong.append(ACTIVE_POL2)

    # #############################################
    # Rules define association/dissociation behaviour between proteins and DNA
    # #############################################
    rules_random = [
        Rule(reactants=['!dna_%s' % gp, gp], products=['dna_%s' % gp], c=random_asso)
        for gp in gille_proteins if gp != Protein.POL2
    ]

    rules_random.append(
        Rule(
            reactants=['!dna_%s_%s' % (ACTIVE_POL2, Protein.POL2), Protein.POL2],
            products=['dna_%s' % Protein.POL2],
            c=random_asso * .1
        )
    )

    rules_random.extend([
        Rule(reactants=['dna_%s' % gp], products=[gp], c=random_disso)
        for gp in gille_proteins
    ])

    rules_dna = [
        # Rad3 associating to the core promoter
        Rule(
            reactants=['!dna_cp_%s' % Protein.RAD3, Protein.RAD3],
            products=['dna_cp_%s' % Protein.RAD3], c=rad3_cp_asso
        ),
        Rule(reactants=['dna_cp_%s' % Protein.RAD3], products=[Protein.RAD3], c=rad3_cp_disso),
        # Pol2 associating to the TSS if rad3 present at the core promoter but swiftly moving it to the
        # beginning of the transcript
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, '!dna_tss_%s' % ACTIVE_POL2, Protein.POL2],
            products=['dna_cp_%s' % Protein.RAD3, 'dna_tss_%s' % ACTIVE_POL2],
            c=pol2_trans_c
        ),
        Rule(
            reactants=['dna_tss_%s' % ACTIVE_POL2],
            products=[Protein.POL2],
            c=pol2_disso
        ),
        Rule(
            reactants=[
                'dna_transcript_%s_!%s' % (ACTIVE_POL2, Protein.RAD26),
                Protein.RAD26
            ],
            products=['dna_transcript_%s_%s' % (ACTIVE_POL2, Protein.RAD26)],
            c=rad26_asso
        ),
        Rule(
            reactants=['dna_transcript_%s_%s' % (ACTIVE_POL2, Protein.RAD26)],
            products=['dna_transcript_%s' % ACTIVE_POL2, Protein.RAD26],
            c=rad26_disso
        )
    ]

    # #############################################
    # Damage response
    # #############################################
    elong_speed_treated = 400
    rad4_cpd_chip = 2.8  # TODO Replace made up value
    rad4_cpd_asso = rad4_cpd_chip / (chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))

    rad26_cpd_chip = 3.1  # TODO Replace made up value
    rad26_cpd_asso = rad26_cpd_chip / (chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))

    rad3_cpd_chip = 6.1  # TODO Replace made up value
    rad3_cpd_asso = rad3_cpd_chip / (chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))
    pol2_backtracking = rad3_cpd_asso

    # Exclude Pol2 from random association/dissociation
    rules_cpd_random = [
        Rule(reactants=['!dna_%s' % gp, gp], products=['dna_%s' % gp], c=random_asso * .1)  # TODO Replace made up value
        for gp in gille_proteins if gp != Protein.POL2
    ]

    rules_cpd_random.extend([
        Rule(reactants=['dna_%s' % gp], products=[gp], c=random_disso * .1)  # TODO Replace made up value
        for gp in gille_proteins if gp != Protein.POL2
    ])

    damage_response = [
        # ##########################
        # TC-NER
        # ##########################
        # Still Pol2 transcription and interaction with TSS but reduced probability
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, '!dna_tss_%s' % ACTIVE_POL2, Protein.POL2],
            products=['dna_cp_%s' % Protein.RAD3, 'dna_tss_%s' % ACTIVE_POL2],
            c=pol2_trans_c * .1
        ),
        Rule(
            reactants=['dna_tss_%s' % ACTIVE_POL2],
            products=[Protein.POL2],
            c=pol2_disso * .1
        ),

        # Same dissociation probability for RAD3 from cp but no further association
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3],
            products=[Protein.RAD3],
            c=rad3_cp_disso
        ),

        # Rad26 association to Pol2
        # TODO is that reasonable? Or only slightly higher association to stalled Pol2
        Rule(
            reactants=[
                'dna_transcript_%s_!%s' % (Protein.POL2, Protein.RAD26),
                Protein.RAD26
            ],
            products=['dna_transcript_%s_%s' % (Protein.POL2, Protein.RAD26)],
            c=rad26_cpd_asso
        ),
        Rule(
            reactants=[
                'dna_transcript_%s_!%s' % (ACTIVE_POL2, Protein.RAD26),
                Protein.RAD26
            ],
            products=['dna_transcript_%s_%s' % (ACTIVE_POL2, Protein.RAD26)],
            c=rad26_cpd_asso
        ),

        # Create correlation between Rad26 and Pol2 due to higher dissociation of Rad26
        Rule(
            reactants=['dna_transcript_!%s_%s' % (Protein.POL2, Protein.RAD26)],
            products=[Protein.RAD26],
            c=rad26_disso
        ),
        Rule(
            reactants=['dna_transcript_!%s_%s' % (ACTIVE_POL2, Protein.RAD26)],
            products=[Protein.RAD26],
            c=rad26_disso
        ),

        # Recruitment Rad26 to lesion in TC-NER
        Rule(
            reactants=['lesion_recognised_%s' % ACTIVE_POL2, '!lesion_recognised_%s' % Protein.RAD26, Protein.RAD26],
            products=['lesion_recognised_%s' % ACTIVE_POL2, 'lesion_recognised_%s' % Protein.RAD26],
            c=rad26_cpd_asso
        ),

        # Recruitment Rad3 TC-NER
        Rule(
            reactants=[
                'lesion_recognised_%s' % ACTIVE_POL2,
                'lesion_recognised_%s' % Protein.RAD26,
                '!lesion_recognised_%s' % Protein.RAD3,
                Protein.RAD3
            ],
            products=['lesion_opened_%s' % Protein.RAD26, 'lesion_opened_%s' % Protein.RAD3, Protein.POL2],  # Removal
            c=rad3_cpd_asso
        ),
        Rule(
            reactants=[
                'lesion_recognised_%s' % ACTIVE_POL2,
                'lesion_recognised_%s' % Protein.RAD26,
                '!lesion_recognised_%s' % Protein.RAD3,
                Protein.RAD3
            ],
            products=['dna_before_%s' % Protein.RAD26, 'lesion_opened_%s' % Protein.RAD3,
                      'dna_before_%s' % Protein.POL2  # Backtracking, doesn't move anymore
                      ],
            c=pol2_backtracking
        ),

        # Continue recruiting Rad3 in the opened state
        Rule(
            reactants=['!lesion_opened_%s' % Protein.RAD3, Protein.RAD3],
            products=['lesion_opened_%s' % Protein.RAD3],
            c=rad3_cpd_asso
        ),

        Rule(
            reactants=[
                'lesion_opened_%s' % ACTIVE_POL2,
                'lesion_opened_%s' % Protein.RAD26,
                'lesion_opened_%s' % Protein.RAD3
            ],
            products=[
                'dna_before_%s' % Protein.RAD26,
                'lesion_opened_%s' % Protein.RAD3,
                'dna_before_%s' % Protein.POL2  # Backtracking, doesn't move anymore
            ],
            c=pol2_backtracking
        ),

        # Continue to remove Pol2
        Rule(
            reactants=[
                'lesion_opened_%s' % ACTIVE_POL2,
                'lesion_opened_%s' % Protein.RAD26,
                '!lesion_opened_%s' % Protein.RAD3,
                Protein.RAD3
            ],
            products=['lesion_opened_%s' % Protein.RAD26, 'lesion_opened_%s' % Protein.RAD3, Protein.POL2],  # Removal
            c=rad3_cpd_asso
        ),

        # ##########################
        # GG-NER
        # ##########################

        # Recruitment Rad3 GG-NER
        # Rule(
        #     reactants=['lesion_recognised_%s' % Protein.RAD4, Protein.RAD3],
        #     products=[Protein.RAD4, 'lesion_opened_%s' % Protein.RAD3],
        #     c=rad3_cpd_asso
        # ),

        # # Dissociation of Rad3 from lesion
        # Rule(
        #     reactants=['dna_lesion_%s' % Protein.RAD3, 'dna_lesion_%s' % Protein.POL2],
        #     products=[Protein.RAD3, 'dna_lesion_%s' % Protein.POL2],
        #     c=rad3_cpd_disso
        # )
    ]

    rad3_nouv = []
    pol2_nouv = []
    rad26_nouv = []

    rad3_t0 = []
    pol2_t0 = []
    rad26_t0 = []
    cpd_distribution = []
    for t in range(10):
        # Put rules together. Although possible to use different rule sets, the single cell scale should make
        # functional interactions much more likely than random intractions. Overall, interactions are slower
        # on a single-cell scale
        rules = [[]]
        rules[0].extend(rules_dna)
        rules[0].extend(rules_random)
        print('%s' % t)
        gille_pool = PoolGillespie(protein_conc=concentrations_pool, rules=rules_pool)
        gille_dna = DNAGillespie(
            gille_pool,
            dna_spec=DEFAULT_DNA_SPEC_1DIM.copy(),
            protein_names=gille_proteins_elong,
            rules=rules,
            elong_speed=elong_speed
        )
        i = 0
        is_radiated = False
        radiation_t = -1
        while True:
            i += 1
            print('Time point: %.3f min' % gille_dna.t)
            print('\n')
            gille_dna.simulate(max_iter=100, random_power=5)

            if gille_dna.t > 20. and not is_radiated:
                plot()
                rad3_nouv.append(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD3]].copy())
                pol2_nouv.append(
                    gille_dna.state[:, gille_dna.protein_to_idx[Protein.POL2]].copy()
                    + gille_dna.state[:, gille_dna.protein_to_idx[ACTIVE_POL2]].copy()
                )
                rad26_nouv.append(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD26]].copy())

                radiation_t = gille_dna.t
                print('##################### UV RADIATION')
                # cpd_start = np.random.choice(
                #     np.arange(DEFAULT_DNA_SPEC_1DIM['transcript'][0], DEFAULT_DNA_SPEC_1DIM['transcript'][1])
                # )
                cpd_start = DEFAULT_CPD[0]
                cpd_distribution.append(np.arange(cpd_start, cpd_start + DEFAULT_CPD_LENGTH))
                rules = [[]]
                rules[0].extend(rules_cpd_random)
                rules[0].extend(damage_response)
                gille_dna.set_rules(rules, elong_speed_treated)
                gille_dna.add_lesion(cpd_start, cpd_start + DEFAULT_CPD_LENGTH)
                gille_dna.reaction_prob()
                is_radiated = True

            if radiation_t > 0 and gille_dna.t - radiation_t > 20:
                plot()
                # gille_dna.simulate()
                rad3_t0.append(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD3]].copy())
                pol2_t0.append(
                    gille_dna.state[:, gille_dna.protein_to_idx[Protein.POL2]].copy()
                    + gille_dna.state[:, gille_dna.protein_to_idx[ACTIVE_POL2]].copy())
                rad26_t0.append(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD26]].copy())
                break

    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    ax[0][0].plot(np.mean(rad3_nouv, axis=0), color='tab:orange')
    ax[0][0].fill_between(
        np.arange(100),
        np.maximum(np.mean(rad3_nouv, axis=0) - np.var(rad3_nouv, axis=0), 0),
        np.mean(rad3_nouv, axis=0) + np.var(rad3_nouv, axis=0),
        color='tab:orange',
        alpha=.2
    )
    ax[0][0].set_title('Rad3 NoUV')
    ax[0][0].set_ylabel('#Molecules')

    ax[0][1].plot(np.mean(rad3_t0, axis=0), color='tab:orange')
    ax[0][1].fill_between(
        np.arange(100),
        np.maximum(np.mean(rad3_t0, axis=0) - np.var(rad3_t0, axis=0), 0),
        np.mean(rad3_t0, axis=0) + np.var(rad3_t0, axis=0),
        color='tab:orange',
        alpha=.2
    )
    ax[0][1].set_title('Rad3 T0')
    ax[0][1].set_ylabel('#Molecules')

    ax[1][0].plot(np.mean(pol2_nouv, axis=0), color='tab:green')
    ax[1][0].fill_between(
        np.arange(100),
        np.maximum(np.mean(pol2_nouv, axis=0) - np.var(pol2_nouv, axis=0), 0),
        np.mean(pol2_nouv, axis=0) + np.var(pol2_nouv, axis=0),
        color='tab:green',
        alpha=.2
    )
    ax[1][0].set_title('Pol2 NoUV')
    ax[1][0].set_ylabel('#Molecules')

    ax[1][1].plot(np.mean(pol2_t0, axis=0), color='tab:green')
    ax[1][1].fill_between(
        np.arange(100),
        np.maximum(np.mean(pol2_t0, axis=0) - np.var(pol2_t0, axis=0), 0),
        np.mean(pol2_t0, axis=0) + np.var(pol2_t0, axis=0),
        color='tab:green',
        alpha=.2
    )
    ax[1][1].set_title('Pol2 T0')
    ax[1][1].set_ylabel('#Molecules')

    ax[2][0].plot(np.mean(rad26_nouv, axis=0), color='tab:cyan')
    ax[2][0].fill_between(
        np.arange(100),
        np.maximum(np.mean(rad26_nouv, axis=0) - np.var(rad26_nouv, axis=0), 0),
        np.mean(rad26_nouv, axis=0) + np.var(rad26_nouv, axis=0),
        color='tab:cyan',
        alpha=.2
    )
    ax[2][0].set_title('Rad26 NoUV')
    ax[2][0].set_ylabel('#Molecules')

    ax[2][1].plot(np.mean(rad26_t0, axis=0), color='tab:cyan')
    ax[2][1].fill_between(
        np.arange(100),
        np.maximum(np.mean(rad26_t0, axis=0) - np.var(rad26_t0, axis=0), 0),
        np.mean(rad26_t0, axis=0) + np.var(rad26_t0, axis=0),
        color='tab:cyan',
        alpha=.2
    )
    ax[2][1].set_title('Rad26 T0')
    ax[2][1].set_ylabel('#Molecules')

    fig.suptitle('Simulated ChIP-seq')
    fig.tight_layout()

    fig_cpd, ax_cpd = plt.subplots(1, 1)
    ax_cpd.hist(np.asarray(cpd_distribution).reshape(-1), bins=50, range=(0, 100))
    ax_cpd.set_title('Lesion Histogram')
    fig_cpd.tight_layout()
    plt.show()


def main():
    # routine_gille_pool()
    routine_gille_dna()


if __name__ == '__main__':
    main()

