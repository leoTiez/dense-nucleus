#!/usr/bin/env python3
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt

from datahandler.seqDataHandler import smooth
from modules.rules import *
from modules.proteins import Protein
from modules.utils import validate_dir
from modules.abstractClasses import Gillespie


class Lesion:
    def __init__(self, start, end):
        """
        Lesion class which implements the location in the state of a lesion. This permits the implementation
        of enzymatic steps. The concrete enzymatic steps are implemented in the modules.rules file.
        :param start: Start position of the lesion
        :type start: int
        :param end: End position of the lesion
        :type end: int
        """
        self.start = start
        self.end = end
        self.state = CPD_STATES['new']

    def update_state_to(self, new_state):
        """
        Updates the state to a new state, which permits the application of rules that are dependent on particular
        states.
        :param new_state: Name of the new state that is defined in the CPD_STATES constant dictionary in the
        modules.rules file.
        :type new_state: str
        :return: None
        """
        self.state = CPD_STATES[new_state]


class PoolGillespie(Gillespie):
    def __init__(self, protein_conc, rules):
        """
        Gillespie algorithm that simulates chemical reactions in a well-mixed solution. No notion of space
        involved
        :param protein_conc: Protein concentrations passed as a dictionary, where the keys represent the name of
        the protein and the value the number of proteins
        :type protein_conc: dict
        :param rules:
        """
        self.state = protein_conc
        self.rules = rules
        self.t = 0
        self.a = np.zeros(len(rules))
        self.reaction_prob()

    def h(self, reactants):
        """
        Determines the possible interaction combiantions according to the Gillespie paper.
        :param reactants: The names of the proteins. Only two reactants are permitted at the momemtn
        :type reactants: list(string)
        :return: Number of possible combinations
        """
        for r in reactants:
            if r not in self.state:
                return 0
        if len(reactants) == 0:
            return 1
        elif len(reactants) == 1:
            return self.state[reactants[0]]
        elif len(reactants) == 2:
            if reactants[0] == reactants[1]:
                return .5 * self.state[reactants[0]] * (self.state[reactants[0]] - 1)
            else:
                return self.state[reactants[0]] * self.state[reactants[1]]
        else:
            raise ValueError('Reactions with more than 2 reactants not supported')

    def reaction_prob(self):
        """
        Calculate and update the reaction probabilities based on the reaction probability in an infinitesimal step
        (given in the rules) and the number of possible interaction combinations.
        :return: None
        """
        for i in range(len(self.rules)):
            reactants = self.rules[i].reactants
            self.a[i] = self.h(reactants) * self.rules[i].c

    def get_state(self, reactant):
        """
        Getter function for the state of a particular reactant/protein
        :param reactant: Name of the reactant
        :type reactant: str
        :return: Number of proteins in the solution
        """
        return self.state[reactant] if reactant in self.state else 0

    def _change_reactant_delta(self, reactant, delta):
        """
        Update the state by a given number of proteins that are either added or removed from the pool
        :param reactant: Name of the reactant
        :type reactant: str
        :param delta: The difference that is added to the current state. Negative values therefore represent
        the number of particles that are removed from the pool
        :type delta: int
        :return: None
        """
        if reactant not in self.state:
            self.state[reactant] = 0
        self.state[reactant] = np.maximum(0, self.state[reactant] + delta)

    def reduce(self, reactant):
        """
        Reduce the number of available particles in the pool by one.
        :param reactant: Name of the reactant
        :type reactant: str
        :return: None
        """
        self._change_reactant_delta(reactant, -1)

    def increase(self, reactant):
        """
        Increase the number of available particles in the pool by one.
        :param reactant: Name of the reactant
        :type reactant: str
        :return: None
        """
        self._change_reactant_delta(reactant, +1)

    def simulate(self):
        """
        Simulate a single interaction in the solution
        :return: The time it took for the reaction to occur
        """
        r1, r2 = np.random.random(2)
        a0 = np.sum(self.a)
        if a0 == 0:
            return 0
        tau = 1./a0 * np.log(1./r1)
        self.t += tau
        mu = np.searchsorted([np.sum(self.a[:i]) for i in range(1, self.a.size + 1)], a0 * r2)
        reactants = self.rules[mu].reactants
        products = self.rules[mu].products

        for r in reactants:
            if r not in self.state:
                self.state[r] = 0
            self.state[r] = np.maximum(0, self.state[r] - 1)
        for p in products:
            if p not in self.state:
                self.state[p] = 0
            self.state[p] += 1

        self.reaction_prob()
        return tau

    def plot(self, save_plot=False, save_prefix=''):
        """
        Plot the state as a bar plot
        :param save_plot: If the flag is true, the plots are saved instead of being displayed
        :type save_plot: bool
        :param save_prefix: String that can be used as an identifier for the graph if it is saved
        :type save_prefix: str
        :return: None
        """
        names, state = zip(*self.state.items())
        plt.bar(np.arange(len(state)), state, tick_label=names)
        plt.title("System's State")
        plt.ylabel('#Particles')
        if not save_plot:
            plt.show()
        else:
            path = validate_dir('figures')
            plt.savefig('%s/%s_protein_pool.png' % (path, save_prefix))


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
        """
        Class to simulate the Gillespie algorithm with a notion of space. The DNA molecule is represented as a
        one-dim array. The DNA molecule interacts w/ a well-mixed solution which represents the core.
        :param gille_pool: Gillespie pool object representing the nucleus surrounding the DNA
        :type gille_pool: PoolGillespie
        :param size: Length of the DNA molecule
        :type size: int
        :param protein_names: List with protein names
        :type protein_names: list(str)
        :param dna_spec: DNA specification define the areas where interaction profiles can change. Areas are passed
        as a dictionary where the key represents the name of the DNA area and the value is a tuple or a list of length
        2 defining the start and the end positions.
        :type dna_spec: dict(str, list(int))
        :param rules: A list with lists of rules. Due to the fact that the lists are nested, it's possible to run
        several rule sets independently in the simulation. This can be used if reactions are frequent.
        :type rules: list(list(Rule))
        :param elong_speed: Elongation speed of Pol2 Usually assumed to be base pairs per minute but that is dependent
        on the usage
        :type elong_speed: int
        """
        self.gille_pool = gille_pool
        self.size = size
        self.protein_names = protein_names
        num_species = len(self.protein_names)

        self.state = np.zeros((size, num_species))

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
        """
        Setter function for new rules, e.g. when the cell is radiated and the interaction profiles change.
        Interaction probabilities (or number of expected interactions per time unit) are updated with the new rules.
        :param rules: Nested list with rules (see docstring of the constructor)
        :type rules: list(list(Rule))
        :param elong_speed: New elongation speed of Pol2
        :type elong_speed: int
        :return: None
        """
        self.rules = rules
        self.elong_speed = elong_speed
        self.a = [np.zeros(len(r)) for r in self.rules]

    def add_lesion(self, start, end):
        """
        Add a lesion to the dna. A lesion is represented as an object which can go through different enzymatic states
        (see constructor of Lesion)
        :param start: Start position
        :type start: int
        :param end: End position
        :type end: int
        :return: None
        """
        self.lesions.append(Lesion(start, end))

    def _determine_dna_idx(self, dna_react='', dna_prod='', proteins=None):
        """
        Determine the indices/base pairs in the DNA molecule where proteins can interact based on the rules.
        Rules can be determined with conditions of presence or absence (see constructor if Rule) of other proteins
        or the state of a lesion. DNA with which proteins interact is passed as either dna_react (reaction condition)
        or dna_prod (reaction product) but never both.
        All entities must be connected with an underscore. If it is dependent on absence of a protein,
        an exclamation mark is added to the beginning. Example:
        dna_react = dna_cp
        proteins = ['rad3', '!pol2']
        :param dna_react: DNA segment on which the reaction is conditioned
        :type dna_react: str
        :param dna_prod: DNA segment with which the chemicals effectively bind
        :type dna_prod: str
        :param proteins: List with proteins
        :type proteins: list(str)
        :return: Positions on the DNA or which the conditions defined in the rules are met
        """
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
        """
        Determine the DNA indices/base pairs which are part of the reactants
        :param dna_string: string which defines the dna segment (e.g. dna or dna_transcript)
        :type dna_string: str
        :param proteins: List of proteins which interact with the DNA or on which the reaction is conditioned
        :type proteins: list(str)
        :return: Indices on the DNA for which the conditions are met
        """
        return self._determine_dna_idx(dna_react=dna_string, dna_prod='', proteins=proteins)

    def determine_dna_idx_prod(self, dna_string, proteins):
        """
        Determine the DNA indices/base pairs which are part of the products
        :param dna_string: string which defines the dna segment (e.g. dna or dna_transcript)
        :type dna_string: str
        :param proteins: List of proteins which interact with the DNA or on which the reaction is conditioned
        :type proteins: list(str)
        :return: Indices on the DNA for which the conditions are met
        """
        return self._determine_dna_idx(dna_react='', dna_prod=dna_string, proteins=proteins)

    def get_reacting_protein(self, reactant_org):
        """
        Get the proteins for which the rule is defined
        :param reactant_org: Rule string which defines the interacting proteins and DNA segments
        :type reactant_org: str
        :return: List with proteins
        """
        reactant = reactant_org
        split = reactant.split('_')
        proteins = [protein for protein in split if protein.strip('!') in self.protein_names]
        return proteins if proteins else None

    def get_reacting_dna(self, reactant_org):
        """
        Get the DNA segment with which the proteins interact.
        :param reactant_org: Rule string which defines the interacting proteins and DNA segments
        :type reactant_org: str
        :return: String defining the DNA segment
        """
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
        """
        Get the lesion type from the reactants
        :param reactants: Rule string which defines the interacting proteins and DNA segments
        :type reactants: str
        :return: Type of lesion as a string
        """
        split = reactants.split('_')
        try:
            if split[1] in self.protein_names:
                return ''
            else:
                return split[1]
        except IndexError:
            return ''

    def h(self, reactants):
        """
        Determine the number of all possible combinations how proteins/DNA can interact w/ each other
        :param reactants: List with reactants
        :type reactants: list(str)
        :return: Number of all possible interaction combinations
        """
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
        """
        Calculate and update the reaction probabilities (or the number of expected reactions per time unit)
        :return: None
        """
        for r in range(len(self.rules)):
            for i in range(len(self.rules[r])):
                reactants = self.rules[r][i].reactants
                self.a[r][i] = self.h(reactants) * self.rules[r][i].c

    def _sample_reaction(self, rs_idx=0):
        """
        Sample a reaction and reaction time
        :param rs_idx: Rule set index
        :type rs_idx: int
        :return: Time step and the index of the reaction in the given rule set
        """
        a = self.a[rs_idx]
        a0 = np.sum(a)
        if a0 == 0:
            return np.inf, -1
        r1, r2 = np.random.random(2)
        tau = 1./a0 * np.log(1./r1)
        mu = np.searchsorted([np.sum(a[:i]) for i in range(1, a.size + 1)], a0 * r2)
        return tau, mu

    def _update(self, mu, rs_idx):
        """
        Update the state of the system given a reaction according to which the interactions change. This function
        also updates the reaction probabilities.
        :param mu: Index of the reaction
        :type mu: int
        :param rs_idx: Index of the rule set
        :type rs_idx: int
        :return: None
        """
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
                                            self.state[:, self.protein_to_idx[Protein.ACTIVE_POL2]] < 1,
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
                            update_mask = [prot_temp in dna_interact_dict[prev_lesion_state] for prot_temp in proteins]
                            if any(update_mask):
                                update_idx = np.where(update_mask)[0]
                                self.state[
                                    dna_interact_dict[prev_lesion_state][proteins[update_idx[0]]],
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

        for num in range(len(self.lesions)):
            if self.lesions[num].state == CPD_STATES['removed']:
                del self.lesions[num]
        self.reaction_prob()

    def simulate(self, max_iter=10):
        """
        Simulate a reaction in the system. As there are several rule sets possible, the sample times can be different.
        Therefore, if a rule of one rule set takes much less time than the reaction of another rule set, more reactions
        are sampled until no more reaction can occur within the time frame the longer reaction took. This function
        also elongates active Pol2 along the genome.
        :param max_iter: Maximum number of reactions which can be sampled from a rule set which was quicker than a
        rule from another rule set.
        :type max_iter: int
        :return: The reaction time
        """
        def update_pol2(s=None, e=None):
            """
            Update the Pol2 state dependent on active and inactive Pol2 that is associated to the DNA
            :param s: Start index
            :type s: int
            :param e: End index
            :type e: int
            :return: None
            """
            s = s if s is not None else self.dna_spec['tss'][0]
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

        p_idx = self.protein_to_idx[Protein.ACTIVE_POL2]
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

    def plot(self, proteins, colors, smoothing=3, save_plot=False, save_prefix=''):
        """
        Plot the state occupancy level on the DNA
        :param proteins: Proteins that are plotted
        :type proteins: list(str)
        :param colors: Colors that are used
        :type colors: list(str)
        :param smoothing: The number of values that are used for smoothing the graph
        :type smoothing: int
        :param save_plot: If True, the plot is saved instead of displayed
        :type save_plot: bool
        :param save_prefix: Identifier that is added to the beginning of the saved plot name
        :type save_prefix: str
        :return: None
        """
        for p, c in zip(proteins, colors):
            if p != Protein.POL2 and p != Protein.ACTIVE_POL2:
                plt.plot(smooth(self.get_protein_state(p), smoothing), label=p, color=c)
            else:
                plt.plot(smooth(
                    self.get_protein_state(Protein.POL2) + self.get_protein_state(Protein.ACTIVE_POL2),
                    smoothing
                ), label='Pol2', color=c)

        plt.xlabel('DNA Position')
        plt.ylabel('#Molecules')
        plt.title('Smoothed ChIP-seq Simulation')
        plt.legend(loc='upper right')
        if not save_plot:
            plt.show()
        else:
            path = validate_dir('figures')
            plt.savefig('%s/%s_singe_cell_state.png' % (path, save_prefix))

    def get_protein_state(self, protein, start=None, end=None):
        """
        Get the occupancy levels of a particular protein
        :param protein: Protein name
        :type protein: str
        :param start: Start position
        :type start: int
        :param end: End position
        :type end: int
        :return: Occupancy levels of the given protein
        """
        start = start if start is not None else 0
        end = end if end is not None else self.state.shape[0]
        return self.state[start:end, self.protein_to_idx[protein]].copy()
