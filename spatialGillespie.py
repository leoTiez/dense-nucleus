#!/usr/bin/env python3
from collections.abc import Iterable
from abc import ABC, abstractmethod
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

DEFAULT_DNA_SPEC_1DIM = {
    'cp': [0, 10],
    'tss': [10, 15],
    'transcript': [15, 90],
    'tts': [90, 100]
}


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
    def reaction_prob(self, update_idx):
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
        self.state[self.prot_idx[reactant]] += delta

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
            self.state[self.prot_idx[r]] -= 1
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
            num_species=7,
            dna_spec=DEFAULT_DNA_SPEC_1DIM,
            rules=[],
            rules_elongation=[]
    ):
        self.gille_pool = gille_pool
        self.size = size
        if isinstance(size, int):
            self.state = np.zeros((size, num_species))
        elif isinstance(size, Iterable):
            self.state = np.zeros(tuple(list(size).append(num_species)))
        self.dna_spec = dna_spec
        self.rules = rules
        self.rules_elong = rules_elongation
        self.a = np.zeros(len(self.rules))
        self.a_elong = np.zeros(len(self.rules_elong))
        self.t = 0
        self.protein_rule_map = {}
        for protein in self.gille_pool.prot_idx.keys():
            self.protein_rule_map[protein] = [i for i, r in enumerate(rules)
                                              if protein in r.reactants or protein in r.products]
        self.reaction_prob(range(self.a.size), is_elong=False)
        self.reaction_prob(range(self.a_elong.size), is_elong=True)

    def determine_dna_idx(self, dna_string):
        area = []
        if 'dna' == dna_string.lower():
            return np.arange(self.size)

        pos = dna_string.split('_')[1]
        try:
            area = [int(pos)]
        except ValueError:
            pass
        if not area:
            for key in self.dna_spec.keys():
                if key.lower() in dna_string.lower():
                    protein = self.get_reacting_protein(dna_string)
                    border_start = self.dna_spec[key][::2]
                    border_end = self.dna_spec[key][1::2]
                    for s, e in zip(border_start, border_end):
                        if protein is None:
                            area.extend(list(range(s, e)))
                        else:
                            area.extend(np.where(self.state[s:e, self.gille_pool.prot_idx[protein]] > 0)[0])
                    break

        return area

    def get_reacting_protein(self, reactant):
        split = reactant.split('_')
        try:
            # Check whether this can be represented as integer position
            pos = int(split[1])
            return reactant.split('dna_%s_' % split[1])[1]
        except ValueError:
            pass

        for key in self.dna_spec.keys():
            if key.lower() in reactant.lower():
                if 'dna_%s' % key != reactant.lower():
                    return reactant.split('dna_%s_' % key)[1]
                else:
                    return

        return reactant.split('dna_')[1]

    def get_reacting_dna(self, reactant):
        if 'dna' not in reactant:
            return
        split = reactant.split('_')
        try:
            pos = int(split[1])
            return 'dna_%s' % split[1]
        except ValueError:
            pass
        if len(split) == 2 or split[1] not in self.dna_spec.keys():
            return 'dna'

        for key in self.dna_spec.keys():
            if key.lower() in reactant.lower():
                return 'dna_%s' % key

    def _h_elong(self, reactants):
        try:
            pos_begin = [int(self.get_reacting_dna(r).split('dna_')[1]) for r in reactants]
            # pos_begin = [self.determine_dna_idx(r)[0] for r in reactants]
        except ValueError:
            raise ValueError('DNA elongation does not convey position information')
        proteins = [self.get_reacting_protein(r) for r in reactants]
        return np.product([self.state[pos, self.gille_pool.prot_idx[p]] for pos, p in zip(pos_begin, proteins)])

    def _h_interact(self, reactants):
        dna_strings = [r for r in reactants if 'dna' in r.lower()]
        reactant_strings = [r for r in reactants if 'dna' not in r.lower()]
        area_len = 1
        for dna_string in dna_strings:
            area_len *= len(self.determine_dna_idx(dna_string))
        return self.gille_pool.h(reactant_strings) * area_len

    def h(self, reactants, is_elong=False):
        if not is_elong:
            return self._h_interact(reactants)
        else:
            return self._h_elong(reactants)

    def reaction_prob(self, update_idx, is_elong=False):
        for i in range(len(self.rules)):
            reactants = self.rules[i].reactants
            self.a[i] = self.h(reactants, is_elong=False) * self.rules[i].c
        for i in range(len(self.rules_elong)):
            reactants = self.rules_elong[i].reactants
            self.a_elong[i] = self.h(reactants, is_elong=True) * self.rules_elong[i].c

        # for i in update_idx:
        #     reactants = self.rules[i].reactants if not is_elong else self.rules_elong[i].reactants
        #     self.a[i] = self.h(reactants, is_elong=False) * self.rules[i].c
        #     self.a_elong[i] = self.h(reactants, is_elong=True) * self.rules_elong[i].c

    def _sample_reaction(self, is_elong=False):
        a = self.a if not is_elong else self.a_elong
        a0 = np.sum(a)
        if a0 == 0:
            return np.inf, -1
        r1, r2 = np.random.random(2)
        tau = 1./a0 * np.log(1./r1)
        mu = np.searchsorted([np.sum(a[:i]) for i in range(1, a.size + 1)], a0 * r2)
        return tau, mu

    def _update(self, mu, is_elong):
        reactants = self.rules[mu].reactants if not is_elong else self.rules_elong[mu].reactants
        products = self.rules[mu].products if not is_elong else self.rules_elong[mu].products
        update_idx = []
        for r in reactants:
            if 'dna' in r.lower():
                if np.any([t in r for t in Protein.get_types()]):
                    area = self.determine_dna_idx(r)
                    try:
                        pos = np.random.choice(area)
                    except ValueError:
                        print('SHITE')
                        pass
                    protein = self.get_reacting_protein(r)
                    self.state[pos, self.gille_pool.prot_idx[protein]] -= 1
                    update_idx.extend(self.protein_rule_map[protein])
            else:
                self.gille_pool.reduce(r)
                update_idx.extend(self.protein_rule_map[r])
        for p in products:
            if 'dna' in p.lower():
                if np.any([t in p for t in Protein.get_types()]):
                    try:
                        area = self.determine_dna_idx(self.get_reacting_dna(p))
                    except:
                        self.get_reacting_dna(p)
                    pos = np.random.choice(area)
                    protein = self.get_reacting_protein(p)
                    self.state[pos, self.gille_pool.prot_idx[protein]] += 1
                    update_idx.extend(self.protein_rule_map[protein])
            else:
                self.gille_pool.increase(p)
                update_idx.extend(self.protein_rule_map[p])

        self.reaction_prob(list(set(update_idx)), is_elong=is_elong)

    def simulate(self):
        tau_pool = self.gille_pool.simulate()
        self.t += tau_pool
        tau, mu = self._sample_reaction(is_elong=False)
        self.t += tau
        self._update(mu, is_elong=False)
        tau_elong, mu_elong = self._sample_reaction(is_elong=True)
        if not np.isinf(tau_elong):
            self._update(mu_elong, is_elong=True)
            self.t += tau_elong

        return tau + tau_elong if not np.isinf(tau_elong) else tau


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
    rules_pool = [
        Rule(
            reactants=[Protein.POL2, Protein.RAD26],
            products=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=.01
        ),
        Rule(
            reactants=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=[Protein.POL2, Protein.RAD26],
            c=.3
        )
    ]

    concentrations_pool = {
        Protein.RAD3: 1000,
        Protein.POL2: 1000,
        Protein.RAD26: 1000,
        Protein.IC_RAD3: 500,
        Protein.IC_POL2: 500,
        Protein.IC_RAD26: 500,
        '_'.join(sorted([Protein.POL2, Protein.RAD26])): 50
    }

    # #############################################
    # Rules define association/dissociation behaviour between proteins and DNA
    # #############################################
    rules_dna = [
        # Random association/dissociation
        Rule(reactants=['dna', Protein.RAD3], products=['dna_%s' % Protein.RAD3], c=0.005),
        Rule(reactants=['dna_%s' % Protein.RAD3], products=['dna', Protein.RAD3], c=0.008),
        Rule(reactants=['dna', Protein.POL2], products=['dna_%s' % Protein.POL2], c=0.005),
        Rule(reactants=['dna_%s' % Protein.POL2], products=['dna', Protein.POL2], c=0.0008),
        Rule(reactants=['dna', Protein.RAD26], products=['dna_%s' % Protein.RAD26], c=0.005),
        Rule(reactants=['dna_%s' % Protein.RAD26], products=['dna', Protein.RAD26], c=0.008),
        Rule(reactants=['dna', Protein.IC_RAD3], products=['dna_%s' % Protein.IC_RAD3], c=0.005),
        Rule(reactants=['dna_%s' % Protein.IC_RAD3], products=['dna', Protein.IC_RAD3], c=0.008),
        Rule(reactants=['dna', Protein.IC_POL2], products=['dna_%s' % Protein.IC_POL2], c=0.005),
        Rule(reactants=['dna_%s' % Protein.IC_POL2], products=['dna', Protein.IC_POL2], c=0.008),
        Rule(reactants=['dna', Protein.IC_RAD26], products=['dna_%s' % Protein.IC_RAD26], c=0.005),
        Rule(reactants=['dna_%s' % Protein.IC_RAD26], products=['dna', Protein.IC_RAD26], c=0.008),
        Rule(
            reactants=['dna', '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=['dna_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=0.005
        ),
        Rule(
            reactants=['dna_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=['dna', '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=0.0008
        ),
        # Rad3 associating to the core promoter
        Rule(reactants=['dna_cp', Protein.RAD3], products=['dna_cp_%s' % Protein.RAD3], c=0.02),
        Rule(reactants=['dna_cp_%s' % Protein.RAD3], products=['dna_cp', Protein.RAD3], c=0.03),
        # Pol2(:Rad26) associating to the TSS if rad3 present at the core promoter but swiftly moving it to the
        # beginning of the transcript
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, 'dna_tss', Protein.POL2],
            products=['dna_cp_%s' % Protein.RAD3, 'dna_transcript_%s' % Protein.POL2],
            c=0.08
        ),
        # Rule(reactants=['dna_tss_%s' % Protein.POL2], products=['dna_cp_%s' % Protein.RAD3, Protein.POL2], c=0.01),
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, 'dna_tss', '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=[
                'dna_cp_%s' % Protein.RAD3,
                'dna_transcript_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))
            ],
            c=0.08
        ),
        # Rule(
        #     reactants=['dna_tss_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
        #     products=['dna_cp_%s' % Protein.RAD3, '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
        #     c=0.02
        # ),
        # Dissociation from the TTS
        Rule(reactants=['dna_tts_%s' % Protein.POL2], products=['dna_tts', Protein.POL2], c=0.4),
        Rule(
            reactants=['dna_tts_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=['dna_tts', '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=0.4
        ),
    ]

    # #############################################
    # Elongation
    # #############################################
    rules_elongation = [
        Rule(
            reactants=['dna_%s_%s' % (i, Protein.POL2)], products=['dna_%s_%s' % (i + 1, Protein.POL2)], c=0.4)
        for i in range(DEFAULT_DNA_SPEC_1DIM['tss'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ]
    rules_elongation.extend([
        Rule(
            reactants=['dna_%s_%s' % (i, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
            products=['dna_%s_%s' % (i + 1, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
            c=0.4
        )
        for i in range(DEFAULT_DNA_SPEC_1DIM['tss'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ])

    rad3 = []
    pol2 = []
    rad26 = []
    rad26pol2 = []
    for t in range(10):
        print('%s' % t)
        gille_pool = PoolGillespie(protein_conc=concentrations_pool, rules=rules_pool)
        gille_dna = DNAGillespie(gille_pool, rules=rules_dna, rules_elongation=rules_elongation)
        for i in range(2000):
            # print(gille_dna.state[:, 0])
            # print(gille_dna.t)
            # print('\n')
            gille_dna.simulate()
            # if i % 100 == 0:
            #     plt.plot(smooth(gille_dna.state[:, gille_pool.prot_idx[Protein.RAD3]], 3), label='Rad3')
            #     plt.plot(smooth(gille_dna.state[:, gille_pool.prot_idx[Protein.POL2]], 3), label='Pol2')
            #     plt.plot(smooth(gille_dna.state[:, gille_pool.prot_idx[Protein.RAD26]], 3), label='Rad26')
            #     plt.plot(
            #         smooth(gille_dna.state[:, gille_pool.prot_idx['_'.join(sorted([Protein.POL2, Protein.RAD26]))]], 7),
            #         label='Pol2:Rad26'
            #     )
            #     plt.xlabel('DNA Position')
            #     plt.ylabel('#Molecules')
            #     plt.title('Smoothed ChIP-seq Simulation')
            #     plt.legend(loc='upper right')
            #     plt.show()

        rad3.append(gille_dna.state[:, gille_pool.prot_idx[Protein.RAD3]])
        pol2.append(
            gille_dna.state[:, gille_pool.prot_idx[Protein.POL2]]
            + gille_dna.state[:, gille_pool.prot_idx['_'.join(sorted([Protein.POL2, Protein.RAD26]))]])
        rad26.append(gille_dna.state[:, gille_pool.prot_idx[Protein.RAD26]]
                     + gille_dna.state[:, gille_pool.prot_idx['_'.join(sorted([Protein.POL2, Protein.RAD26]))]]
        )
        rad26pol2.append(gille_dna.state[:, gille_pool.prot_idx['_'.join(sorted([Protein.POL2, Protein.RAD26]))]])

    # fig, [ax_rad3, ax_pol2, ax_rad26, ax_rad26pol2] = plt.subplots(4, 1, figsize=(8, 8))
    fig, [ax_rad3, ax_pol2, ax_rad26] = plt.subplots(3, 1, figsize=(8, 8))
    ax_rad3.plot(np.mean(rad3, axis=0), color='tab:orange')
    ax_rad3.fill_between(
        np.arange(100),
        np.maximum(np.mean(rad3, axis=0) - np.var(rad3, axis=0), 0),
        np.mean(rad3, axis=0) + np.var(rad3, axis=0),
        color='tab:orange',
        alpha=.2
    )
    ax_rad3.set_title('Rad3')
    ax_rad3.set_ylabel('#Molecules')

    ax_pol2.plot(np.mean(pol2, axis=0), color='tab:green')
    ax_pol2.fill_between(
        np.arange(100),
        np.maximum(np.mean(pol2, axis=0) - np.var(pol2, axis=0), 0),
        np.mean(pol2, axis=0) + np.var(pol2, axis=0),
        color='tab:green',
        alpha=.2
    )
    ax_pol2.set_title('Pol2')
    ax_pol2.set_ylabel('#Molecules')

    ax_rad26.plot(np.mean(rad26, axis=0), color='tab:cyan')
    ax_rad26.fill_between(
        np.arange(100),
        np.maximum(np.mean(rad26, axis=0) - np.var(rad26, axis=0), 0),
        np.mean(rad26, axis=0) + np.var(rad26, axis=0),
        color='tab:cyan',
        alpha=.2
    )
    ax_rad26.set_title('Rad26')
    ax_rad26.set_ylabel('#Molecules')

    # ax_rad26pol2.plot(np.mean(rad26pol2, axis=0), color='tab:red')
    # ax_rad26pol2.fill_between(
    #     np.arange(100),
    #     np.maximum(np.mean(rad26pol2, axis=0) - np.var(rad26pol2, axis=0), 0),
    #     np.mean(rad26pol2, axis=0) + np.var(rad26pol2, axis=0),
    #     color='tab:red',
    #     alpha=.2
    # )
    # ax_rad26pol2.set_title('Pol2:Rad26')
    # ax_rad26pol2.set_ylabel('#Molecules')
    # ax_rad26pol2.set_xlabel('Position')

    fig.suptitle('Simulated ChIP-seq')
    fig.tight_layout()
    plt.show()


def main():
    # routine_gille_pool()
    routine_gille_dna()


if __name__ == '__main__':
    main()

