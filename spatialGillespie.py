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

LENGTH = 100
DEFAULT_DNA_SPEC_1DIM = {
    'cp': [0, 10],
    'tss': [15, 20],  # Set TSS into transcript
    'transcript': [15, 90],
    'tts': [90, LENGTH],
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
        self.rules = rules
        self.elong_speed = elong_speed
        self.a = [np.zeros(len(r)) for r in self.rules]

        self.t = 0

        self.protein_to_idx = {prot: num for num, prot in enumerate(self.protein_names)}
        self.reaction_prob()

    def determine_dna_idx(self, dna_string, proteins=None):
        area = []
        is_except_string = '!' in dna_string
        p_idc = np.asarray([self.protein_to_idx[p] for p in proteins]) if proteins is not None else None
        if 'dna' == dna_string.lower():
            interact_mask = np.all(self.state[:, p_idc] > 0, axis=1)
            if is_except_string:
                interact_mask = ~interact_mask
            return np.arange(self.size)[interact_mask] if p_idc is not None else np.arange(self.size)

        pos = dna_string.split('_')[1]
        try:
            pos = int(pos)
            if p_idc is not None:
                do_interact = np.all(self.state[pos, p_idc] > 0)
                if is_except_string:
                    do_interact = ~do_interact
                area = [pos] if do_interact else []
            else:
                area = [pos] if not is_except_string else []
            return area
        except ValueError:
            pass
        if not area:
            for key in self.dna_spec.keys():
                if key.lower() in dna_string.lower():
                    if not self.dna_spec[key]:
                        return [] if not is_except_string else None
                    border_start = self.dna_spec[key][::2]
                    border_end = self.dna_spec[key][1::2]
                    for s, e in zip(border_start, border_end):
                        if proteins is not None:
                            interact_mask = np.all(self.state[s:e][:, p_idc] > 0, axis=1)
                        else:
                            interact_mask = np.ones(e - s).astype('bool')
                        if is_except_string:
                            interact_mask = ~interact_mask
                        area_range = np.arange(s, e)[interact_mask]
                        area.extend(area_range)
                    break

        return list(np.unique(area))

    def get_reacting_protein(self, reactant_org):
        reactant = reactant_org.split('!')[-1]
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
            if split[1] not in self.dna_spec.keys():
                return 'dna'
        except IndexError:
            # Know that there's nothing after dna because of IndexError
            if split[0] == 'dna':
                return 'dna'

        for key in self.dna_spec.keys():
            if key.lower() in reactant.lower():
                return 'dna_%s' % key

    def h(self, reactants, is_elong=False):
        dna_strings = [r for r in reactants if 'dna' in r.lower()]
        reactant_strings = [r for r in reactants if 'dna' not in r.lower() and '!' not in r.lower()]

        dna_react = 1
        for dna_string in dna_strings:
            proteins = self.get_reacting_protein(dna_string)
            dna_area = self.determine_dna_idx(self.get_reacting_dna(dna_string), proteins=proteins)
            if dna_area is None:
                continue
            if proteins is not None and len(dna_area) > 0:
                p_idc = [self.protein_to_idx[p] for p in proteins]
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
        for r in reactants:
            if 'dna' in r.lower():
                proteins = self.get_reacting_protein(r)
                if proteins is not None:
                    dna_seg = self.get_reacting_dna(r)
                    area = self.determine_dna_idx(dna_seg, proteins=proteins)
                    try:
                        pos = np.random.choice(area)
                    except:
                        self.h(reactants)
                    for p in proteins:
                        self.state[pos, self.protein_to_idx[p]] -= 1
            else:
                self.gille_pool.reduce(r)
        for p in products:
            if 'dna' in p.lower():
                proteins = self.get_reacting_protein(p)
                if proteins is not None:
                    dna_seg = self.get_reacting_dna(p)
                    area = self.determine_dna_idx(dna_seg)
                    pos = np.random.choice(area)
                    for prot in proteins:
                        self.state[pos, self.protein_to_idx[prot]] += 1
            else:
                self.gille_pool.increase(p)

        self.reaction_prob()

    def simulate(self, max_iter=10, random_power=5):
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

        num_pol2_tss = np.sum(self.state[
                self.dna_spec['tss'][0]:self.dna_spec['tss'][1]
            ][:, self.protein_to_idx[Protein.POL2]]
        )
        elong_update = int(self.elong_speed * max_tau * num_pol2_tss)
        while elong_update > 0:
            pol2_state = self.state[
                            self.dna_spec['transcript'][0]:self.dna_spec['transcript'][1]
                         ][:, self.protein_to_idx[Protein.POL2]]
            rad26_state = self.state[
                         self.dna_spec['transcript'][0]:self.dna_spec['transcript'][1]
                         ][:, self.protein_to_idx[Protein.RAD26]]

            if pol2_state.sum() == 0:
                break

            p_idx = np.random.choice(
                [self.protein_to_idx[Protein.POL2], self.protein_to_idx[Protein.RAD26]],
                p=np.asarray([pol2_state.sum(), rad26_state.sum()]) / (pol2_state.sum() + rad26_state.sum())
            )
            state = self.state[self.dna_spec['transcript'][0]:self.dna_spec['transcript'][1]][:, p_idx]
            # upd_pos = np.argmax(state) + self.dna_spec['transcript'][0]
            upd_pos = np.random.choice(
                np.arange(self.dna_spec['transcript'][0], self.dna_spec['transcript'][1]),
                p=(state**random_power)/(state**random_power).sum()  # To the power of 5 is engineered
                                                                     # and doesn't represent a biological feature
            )
            upd_step = int(np.random.uniform(0, self.dna_spec['tts'][1] - upd_pos))
            if upd_step == 0:
                break
            self.state[upd_pos, p_idx] -= 1
            if upd_pos + upd_step < self.dna_spec['transcript'][1]:
                self.state[upd_pos + upd_step, p_idx] += 1
            elong_update -= np.minimum(upd_step, self.dna_spec['transcript'][1] - upd_pos)

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
    # Pivotal definitions
    # Time unit = minute
    # Considering genes with high transcription rate
    gille_proteins = Protein.get_types_gillespie()
    elong_speed = 120
    num_prot = 1e5
    chip_norm = 1e5
    random_chip = 2.6
    rad3_cp_chip = 6.8
    pol2_trans_chip = 7.01
    rad26_trans_chip = 3.5

    random_asso = random_chip / (chip_norm * LENGTH)
    random_disso = 1. / float(LENGTH)
    rad3_cp_asso = rad3_cp_chip / float(chip_norm * (DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]))
    rad3_cp_disso = 1. / (DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0])

    pol2_trans_c = pol2_trans_chip / float(
        chip_norm
        * (
                DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]
                + DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0]
        )
    )
    pol2_disso = rad3_cp_chip * 1. / (DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0])
    rad26_asso = rad26_trans_chip / float(
        num_prot
        * (
                DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]
                + DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0]
        )
    )
    rad26_disso = rad3_cp_chip * 1. / (DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0])

    rules_pool = []
    np.random.seed(0)
    np.random.shuffle(gille_proteins)
    print(gille_proteins)
    concentrations_pool = {gp: num_prot for gp in gille_proteins}
    # num_complex = 100
    # concentrations_pool['_'.join(sorted([Protein.POL2, Protein.RAD26]))] = num_complex
    # #############################################
    # Rules define association/dissociation behaviour between proteins and DNA
    # #############################################
    rules_random = [
        Rule(reactants=['dna', gp], products=['dna_%s' % gp], c=random_asso)
        for gp in gille_proteins
    ]
    rules_random.extend([
        Rule(reactants=['dna_%s' % gp], products=['dna', gp], c=random_disso)
        for gp in gille_proteins
    ])

    rules_dna = [
        # Rad3 associating to the core promoter
        Rule(reactants=['dna_cp', Protein.RAD3], products=['dna_cp_%s' % Protein.RAD3], c=rad3_cp_asso),
        Rule(reactants=['dna_cp_%s' % Protein.RAD3], products=[Protein.RAD3], c=rad3_cp_disso),
        # Pol2(:Rad26) associating to the TSS if rad3 present at the core promoter but swiftly moving it to the
        # beginning of the transcript
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, 'dna_tss', Protein.POL2],
            products=['dna_cp_%s' % Protein.RAD3, 'dna_tss_%s' % Protein.POL2],
            c=pol2_trans_c
        ),
        Rule(
            reactants=['dna_tss_%s' % Protein.POL2],
            products=['dna_tss', Protein.POL2],
            c=pol2_disso
        ),
        Rule(
            reactants=['dna_cp_%s' % Protein.RAD3, 'dna_tss', Protein.RAD26],
            products=['dna_cp_%s' % Protein.RAD3, 'dna_tss_%s' % Protein.RAD26],
            c=rad26_asso
        ),
        Rule(
            reactants=['dna_tss_%s' % Protein.RAD26],
            products=['dna_tss', Protein.RAD26],
            c=rad26_disso
        )
    ]

    # #############################################
    # Damage response
    # #############################################
    # Damage recognition TC-NER
    # rules_elongation.extend([
    #     Rule(
    #         reactants=['dna_lesion', 'dna_%s_%s' % (DEFAULT_CPD[0] - 1, Protein.POL2)],
    #         products=['dna_lesion_%s' % Protein.POL2],
    #         c=.8
    #     ),
    #     Rule(
    #         reactants=['dna_lesion', 'dna_%s_%s' % (DEFAULT_CPD[0] - 1, '_'.join(sorted([Protein.POL2, Protein.RAD26])))],
    #         products=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
    #         c=.8
    #     ),
    # ])

    # damage_response = [
    #     # No more association to the cp and only random Rad3 association
    #     Rule(reactants=['dna_lesion', 'dna_cp_%s' % Protein.RAD3], products=[Protein.RAD3], c=0.5),
    #
    #     # Damage recognition GG-NER
    #     Rule(
    #         reactants=['dna_lesion', '!dna_lesion_%s' % Protein.POL2, Protein.RAD4],
    #         products=['dna_lesion_%s' % Protein.RAD4],
    #         c=1e-2
    #     ),
    #     Rule(
    #         reactants=['dna_lesion_%s' % Protein.RAD4],
    #         products=['dna_lesion', Protein.RAD4],
    #         c=4e-3
    #     ),
    #
    #     # Association / dissociation Rad26 TC-NER
    #     Rule(
    #         reactants=['dna_lesion_%s' % Protein.POL2, Protein.RAD26],
    #         products=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
    #         c=4e-3
    #     ),
    #     Rule(
    #         reactants=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26]))],
    #         products=['dna_lesion_%s' % Protein.POL2, Protein.RAD26],
    #         c=4e-3
    #     ),
    #
    #     # Recruitment Rad3 TC-NER
    #     Rule(
    #         reactants=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26])), Protein.RAD3],
    #         products=[
    #             'dna_lesion_%s' % '_'.join(sorted([Protein.RAD26, Protein.RAD3])),
    #             'dna_before_%s' % Protein.POL2  # Backtracking
    #         ],
    #         c=1e-3
    #     ),
    #     Rule(
    #         reactants=['dna_lesion_%s' % '_'.join(sorted([Protein.POL2, Protein.RAD26])), Protein.RAD3],
    #         products=[
    #             'dna_lesion_%s' % '_'.join(sorted([Protein.RAD26, Protein.RAD3])),
    #             Protein.POL2  # Removal
    #         ],
    #         c=1e-3
    #     ),
    #
    #     # Recruitment Rad3 GG-NER
    #     Rule(
    #         reactants=['dna_lesion_%s' % Protein.RAD4, Protein.RAD3],
    #         products=['dna_lesion_%s' % '_'.join(sorted([Protein.RAD4, Protein.RAD3]))],
    #         c=1e-3
    #     ),
    #
    #     # Dissociation of Rad3 from lesion
    #     Rule(
    #         reactants=['dna_lesion_%s' % Protein.RAD3],
    #         products=[Protein.RAD3],
    #         c=2e-3
    #     ),
    #
    # ]
    # rules_dna.extend(damage_response)

    rad3 = []
    pol2 = []
    rad26 = []
    for t in range(10):
        print('%s' % t)
        gille_pool = PoolGillespie(protein_conc=concentrations_pool, rules=rules_pool)
        gille_dna = DNAGillespie(
            gille_pool,
            protein_names=gille_proteins,
            rules=[rules_random, rules_dna],
            elong_speed=elong_speed
        )
        i = 0
        while True:
            i += 1
            # if i == 1000:
            #     print('##################### UV RADIATION')
            #     gille_dna.dna_spec['lesion'] = DEFAULT_CPD
            #     gille_dna.dna_spec['before'] = [DEFAULT_DNA_SPEC_1DIM['transcript'][0], DEFAULT_CPD[0]]
            #     gille_dna.reaction_prob()
            print(gille_dna.t)
            print('\n')
            gille_dna.simulate(max_iter=100, random_power=5)
            if i % 500 == 0:
                print('Mean Pol2 Transcript: %s' % np.mean(gille_dna.state[15:90, gille_dna.protein_to_idx[Protein.POL2]]))
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

