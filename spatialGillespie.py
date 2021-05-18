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
    'tts': [90, 100],
    'lesion': [],  # No CPDs yet
    'before': [],  # Before the lesion, no CPDs in the beginning of the simulation
}

DEFAULT_CPD = [30, 35]


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
            rules=[]
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
        self.rules = rules
        self.a = [np.zeros(len(r)) for r in self.rules]
        self.t = 0

        self.protein_to_idx = {prot: num for num, prot in enumerate(self.protein_names)}
        self.reaction_prob()

    def determine_dna_idx(self, dna_string, proteins=None):
        area = []
        p_idc = np.asarray([self.protein_to_idx[p] for p in proteins]) if proteins is not None else None
        if 'dna' == dna_string.lower():
            return np.arange(self.size)[np.all(self.state[:, p_idc] > 0, axis=1)] if p_idc is not None\
                else np.arange(self.size)

        pos = dna_string.split('_')[1]
        try:
            pos = int(pos)
            if p_idc is not None:
                area = [pos] if np.all(self.state[pos, p_idc] > 0) else []
            else:
                area = [pos]
            return area
        except ValueError:
            pass
        if not area:
            for key in self.dna_spec.keys():
                if key.lower() in dna_string.lower():
                    if not self.dna_spec[key]:
                        return []
                    border_start = self.dna_spec[key][::2]
                    border_end = self.dna_spec[key][1::2]
                    for s, e in zip(border_start, border_end):
                        area_range = np.arange(s, e)[np.all(self.state[s:e, p_idc] > 0, axis=1)] if p_idc is not None\
                            else np.arange(s, e)
                        area.extend(area_range)
                    break

        return list(np.unique(area))

    def get_reacting_protein(self, reactant):
        split = reactant.split('_')
        try:
            # Check whether this can be represented as integer position
            pos = int(split[1])
            return reactant.split('dna_%s_' % split[1])[1].split('_')
        except ValueError:
            pass
        except IndexError:
            return

        for key in self.dna_spec.keys():
            if key.lower() in reactant.lower():
                if 'dna_%s' % key != reactant.lower():
                    return reactant.split('dna_%s_' % key)[1].split('_')
                else:
                    return

        return reactant.split('dna_')[1].split('_')

    def get_reacting_dna(self, reactant):
        if 'dna' not in reactant:
            return
        split = reactant.split('_')
        try:
            pos = int(split[1])
            return 'dna_%s' % split[1]
        except ValueError:
            pass
        if split[1] not in self.dna_spec.keys():
            return 'dna'

        for key in self.dna_spec.keys():
            if key.lower() in reactant.lower():
                return 'dna_%s' % key

    def h(self, reactants, is_elong=False):
        dna_strings = [r for r in reactants if 'dna' in r.lower()]
        reactant_strings = [r for r in reactants if 'dna' not in r.lower()]
        area_len = 1
        for dna_string in dna_strings:
            proteins = self.get_reacting_protein(dna_string)
            area_len *= len(self.determine_dna_idx(dna_string, proteins=proteins))
        return self.gille_pool.h(reactant_strings) * area_len

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
        for r in reactants:
            if 'dna' in r.lower():
                proteins = self.get_reacting_protein(r)
                if proteins is not None:
                    area = self.determine_dna_idx(r, proteins=proteins)
                    for p in proteins:
                        if np.any(self.state[area, self.protein_to_idx[p]] <= 0):
                            self.determine_dna_idx(r, proteins=proteins)
                            self.h(self.rules[rs_idx][mu].reactants)
                    pos = np.random.choice(area)
                    for p in proteins:
                        self.state[pos, self.protein_to_idx[p]] -= 1
            else:
                self.gille_pool.reduce(r)
        for p in products:
            if 'dna' in p.lower():
                proteins = self.get_reacting_protein(p)
                if proteins is not None:
                    area = self.determine_dna_idx(self.get_reacting_dna(p))
                    pos = np.random.choice(area)
                    for prot in proteins:
                        self.state[pos, self.protein_to_idx[prot]] += 1
            else:
                self.gille_pool.increase(p)

        self.reaction_prob()

    def simulate(self, max_iter=10):
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
    rules_pool = [
        # TODO WHY DOESN'T THIS WORK?
        # Rule(
        #     reactants=[Protein.POL2, Protein.RAD26],
        #     products=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
        #     c=4e-3
        # ),
        # Rule(
        #     reactants=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
        #     products=[Protein.POL2, Protein.RAD26],
        #     c=4e-3
        # )
    ]

    gille_proteins = Protein.get_types_gillespie()
    np.random.seed(0)
    np.random.shuffle(gille_proteins)
    print(gille_proteins)
    concentrations_pool = {gp: 1000 for gp in gille_proteins}
    concentrations_pool['_'.join(sorted([Protein.POL2, Protein.RAD26]))] = 1000

    # #############################################
    # Rules define association/dissociation behaviour between proteins and DNA
    # #############################################
    rules_random = [
        Rule(reactants=['dna', gp], products=['dna_%s' % gp], c=4e-3)
        for gp in gille_proteins
    ]
    rules_random.extend([
        Rule(reactants=['dna_%s' % gp], products=['dna', gp], c=4e-3)
        for gp in gille_proteins
    ])

    rules_dna = [
        # Rad3 associating to the core promoter
        Rule(reactants=['dna_cp', Protein.RAD3], products=['dna_cp_%s' % Protein.RAD3], c=8e-3),
        Rule(reactants=['dna_cp_%s' % Protein.RAD3], products=[Protein.RAD3], c=1e-2),
        # Pol2(:Rad26) associating to the TSS if rad3 present at the core promoter but swiftly moving it to the
        # beginning of the transcript
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, 'dna_tss', Protein.POL2],
            products=['dna_cp_%s' % Protein.RAD3, 'dna_transcript_%s' % Protein.POL2],
            c=8e-3
        ),
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, 'dna_tss', '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=[
                'dna_cp_%s' % Protein.RAD3,
                'dna_transcript_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))
            ],
            c=8e-3
        ),
        # Dissociation from the TTS
        Rule(reactants=['dna_tts_%s' % Protein.POL2], products=['dna_tts', Protein.POL2], c=1e-2),
        Rule(
            reactants=['dna_tts_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=['dna_tts', '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=1e-2
        ),
    ]

    # #############################################
    # Elongation
    # #############################################
    rules_elongation = [
        Rule(
            reactants=['dna_%s_%s' % (i, Protein.POL2)], products=['dna_%s_%s' % (i + 1, Protein.POL2)], c=10)
        for i in range(DEFAULT_DNA_SPEC_1DIM['tss'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ]
    rules_elongation.extend([
        Rule(
            reactants=['dna_%s_%s' % (i, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
            products=['dna_%s_%s' % (i + 1, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
            c=10
        )
        for i in range(DEFAULT_DNA_SPEC_1DIM['tss'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ])

    # Random dissociation
    rules_elongation.extend([
        Rule(
            reactants=['dna_%s_%s' % (i, Protein.POL2)], products=[Protein.POL2], c=1)
        for i in range(DEFAULT_DNA_SPEC_1DIM['tss'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ])
    rules_elongation.extend([
        Rule(
            reactants=['dna_%s_%s' % (i, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
            products=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=1
        )
        for i in range(DEFAULT_DNA_SPEC_1DIM['tss'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ])

    # Dissociation from the TTS
    rules_elongation.extend([
        Rule(
            reactants=['dna_%s_%s' % (i, Protein.POL2)], products=[Protein.POL2], c=15)
        for i in range(DEFAULT_DNA_SPEC_1DIM['tts'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ])
    rules_elongation.extend([
        Rule(
            reactants=['dna_%s_%s' % (i, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
            products=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=15
        )
        for i in range(DEFAULT_DNA_SPEC_1DIM['tts'][0], DEFAULT_DNA_SPEC_1DIM['tts'][1] - 1)
    ])

    # #############################################
    # Damage response
    # #############################################
    # Damage recognition TC-NER
    rules_elongation.extend([
        Rule(
            reactants=['dna_lesion', 'dna_%s_%s' % (DEFAULT_CPD[0] - 1, Protein.POL2)],
            products=['dna_lesion_%s' % Protein.POL2],
            c=1e-2
        ),
        Rule(
            reactants=['dna_lesion', 'dna_%s_%s' % (DEFAULT_CPD[0] - 1, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
            products=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=1e-2
        ),
    ])

    damage_response = [
        # ####
        # OVERWRITING DEFAULT W/OUT UV
        # ####
        # Only random Rad3 association
        # TODO IS TWO DNA PARTS PROBLEM?
        Rule(reactants=['dna_lesion', 'dna_cp', Protein.RAD3], products=['dna_cp_%s' % Protein.RAD3], c=0.),
        Rule(reactants=['dna_lesion', 'dna_cp_%s' % Protein.RAD3], products=[Protein.RAD3], c=40.),

        # Damage recognition GG-NER
        Rule(
            reactants=['dna_lesion', Protein.RAD4],
            products=['dna_lesion_%s' % Protein.RAD4],
            c=1e-2
        ),
        Rule(
            reactants=['dna_lesion_%s' % Protein.RAD4],
            products=['dna_lesion', Protein.RAD4],
            c=5e-3
        ),

        # Association / dissociation Rad26 TC-NER
        Rule(
            reactants=['dna_lesion_%s' % Protein.POL2, Protein.RAD26],
            products=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=1e-2
        ),
        Rule(
            reactants=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=['dna_lesion_%s' % Protein.POL2, Protein.RAD26],
            c=5e-3
        ),

        # # Dissociation Rad3 from core promoter when lesion is found
        # Rule(
        #     reactants=['dna_lesion', 'dna_cp_%s' % Protein.RAD3],
        #     products=[Protein.RAD3],
        #     c=1
        # ),

        # Recruitment Rad3 TC-NER
        Rule(
            reactants=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26])), Protein.RAD3],
            products=[
                'dna_lesion_%s' % '_'.join(sorted([Protein.RAD26, Protein.RAD3])),
                'dna_before_%s' % Protein.POL2  # Backtracking
            ],
            c=9e-3
        ),
        Rule(
            reactants=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26])), Protein.RAD3],
            products=[
                'dna_lesion_%s' % '_'.join(sorted([Protein.RAD26, Protein.RAD3])),
                Protein.POL2  # Removal
            ],
            c=9e-3
        ),

        # Recruitment Rad3 GG-NER
        Rule(
            reactants=['dna_lesion_%s' % Protein.RAD4, Protein.RAD3],
            products=['dna_lesion_%s' % '_'.join(sorted([Protein.RAD4, Protein.RAD3]))],
            c=9e-3
        ),

        # Dissociation of Rad3 from lesion
        Rule(
            reactants=['dna_lesion_%s' % Protein.RAD3],
            products=[Protein.RAD3],
            c=5e-3
        ),

    ]
    rules_dna.extend(damage_response)

    rad3 = []
    pol2 = []
    rad26 = []
    for t in range(10):
        print('%s' % t)
        gille_pool = PoolGillespie(protein_conc=concentrations_pool, rules=rules_pool)
        gille_dna = DNAGillespie(
            gille_pool,
            protein_names=gille_proteins,
            rules=[rules_random, rules_dna, rules_elongation]
        )
        for i in range(2000):
            if i == 500:
                print('##################### UV RADIATION')
                gille_dna.dna_spec['lesion'] = DEFAULT_CPD
                gille_dna.dna_spec['before'] = [DEFAULT_DNA_SPEC_1DIM['transcript'][0], DEFAULT_CPD[0]]
                gille_dna.reaction_prob()
            print(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD4]])
            print(gille_dna.t)
            print('\n')
            gille_dna.simulate()
            if i % 100 == 0 and i > 500:
                plt.plot(smooth(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD3]], 3), label='Rad3')
                plt.plot(smooth(gille_dna.state[:, gille_dna.protein_to_idx[Protein.POL2]], 3), label='Pol2')
                plt.plot(smooth(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD26]], 3), label='Rad26')
                plt.plot(smooth(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD4]], 3), label='Rad4')
                plt.xlabel('DNA Position')
                plt.ylabel('#Molecules')
                plt.title('Smoothed ChIP-seq Simulation')
                plt.legend(loc='upper right')
                plt.show()

        rad3.append(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD3]])
        pol2.append(gille_dna.state[:, gille_dna.protein_to_idx[Protein.POL2]])
        rad26.append(gille_dna.state[:, gille_dna.protein_to_idx[Protein.RAD26]])

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

