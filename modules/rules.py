#!/usr/bin/env python3
# ####################################################
# Pivotal definitions
# Time unit = minute
# ####################################################
from modules.proteins import *
import numpy as np

# CONSTANTS
LENGTH = 100
DEFAULT_DNA_SPEC_1DIM = {
    'cp': [0, 10],
    'tss': [10, 15],
    'transcript': [15, 90],
    'tts': [90, LENGTH]
}

CPD_STATES = {
    'new': 0,
    'recognised': 1,
    'opened': 2,
    'incised': 3,
    'replaced': 4,
    'removed': 5
}

DEFAULT_CPD_LENGTH = 5
BACKTRACKING_LENGTH = 5
DEFAULT_CPD = [30, 30 + DEFAULT_CPD_LENGTH]


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


class PoolNoInteract:
    def __init__(self):
        self.rules = []


# High TR
class NoUVHigh:
    def __init__(self, gille_proteins):
        self.gille_proteins = gille_proteins

        self.elong_speed = 1200
        self.chip_norm = 1e5
        self.random_chip = 2.6
        self.rad3_cp_chip = 6.8

        self.pol2_trans_chip = 7.01
        self.rad26_trans_chip = 3.1
        self.disso_const = 10.

        self.random_asso = None
        self.random_disso = None
        self.pol2_random_asso = None
        self._random()

        self.rad3_cp_asso = None
        self.rad3_cp_disso = None
        self._rad3()

        self.pol2_trans_c = None
        self.pol2_disso = None
        self._pol2()

        self.rad26_asso = None
        self.rad26_disso = None
        self._rad26()

        self.rules = [[]]
        self._rules()

    def _random(self, factor=1, pol2_factor=.1):
        self.random_asso = self.random_chip * factor / (self.chip_norm * LENGTH)
        self.random_disso = self.disso_const / float(LENGTH)
        self.pol2_random_asso = self.random_asso * pol2_factor

    def _rad3(self, factor=2):
        self.rad3_cp_asso = self.rad3_cp_chip * factor / float(
            self.chip_norm * (DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]))
        self.rad3_cp_disso = self.disso_const / (DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0])

    def _pol2(self, factor=2):
        self.pol2_trans_c = self.pol2_trans_chip * factor / float(
            self.chip_norm
            * (
                    DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]
                    + DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0]
            )
        )
        self.pol2_disso = self.disso_const / (DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0])

    def _rad26(self, factor=2):
        self.rad26_asso = self.rad26_trans_chip * factor / (
                self.chip_norm * (DEFAULT_DNA_SPEC_1DIM['transcript'][1] - DEFAULT_DNA_SPEC_1DIM['transcript'][0])
        )
        self.rad26_disso = self.disso_const / (
                DEFAULT_DNA_SPEC_1DIM['transcript'][1] - DEFAULT_DNA_SPEC_1DIM['transcript'][0])

    def _rules_random(self):
        rules_random = [
            Rule(reactants=['!dna_%s' % gp, gp], products=['dna_%s' % gp], c=self.random_asso)
            for gp in self.gille_proteins if gp != Protein.POL2
        ]

        rules_random.append(
            Rule(
                reactants=['dna_!%s_!%s' % (Protein.ACTIVE_POL2, Protein.POL2), Protein.POL2],
                products=['dna_!%s_%s' % (Protein.ACTIVE_POL2, Protein.POL2)],
                c=self.pol2_random_asso
            )
        )

        rules_random.extend([
            Rule(reactants=['dna_%s' % gp], products=[gp], c=self.random_disso)
            for gp in self.gille_proteins
        ])

        return rules_random

    def _rules_nouv(self):
        rules_dna = [
            # Rad3 associating to the core promoter
            Rule(
                reactants=['!dna_cp_%s' % Protein.RAD3, Protein.RAD3],
                products=['dna_cp_%s' % Protein.RAD3], c=self.rad3_cp_asso
            ),
            Rule(reactants=['dna_cp_%s' % Protein.RAD3], products=[Protein.RAD3], c=self.rad3_cp_disso),
            # Pol2 associating to the TSS if rad3 present at the core promoter bu
            Rule(
                reactants=['dna_cp_%s' % Protein.RAD3, '!dna_tss_%s' % Protein.ACTIVE_POL2, Protein.POL2],
                products=['dna_cp_%s' % Protein.RAD3, 'dna_tss_%s' % Protein.ACTIVE_POL2],
                c=self.pol2_trans_c
            ),
            # Pol2 dissociation from TSS
            Rule(
                reactants=['dna_tss_%s' % Protein.ACTIVE_POL2],
                products=[Protein.POL2],
                c=self.pol2_disso
            ),
            # Rad26 association to inactive Pol2. Similar to the verification step of whether pausing is due to
            # a lesion
            Rule(
                reactants=[
                    'dna_transcript_%s_!%s' % (Protein.POL2, Protein.RAD26),
                    Protein.RAD26
                ],
                products=['dna_transcript_%s_%s' % (Protein.POL2, Protein.RAD26)],
                c=self.rad26_asso
            ),
            # Dissociation of Rad26 if no inactive Pol2 is present
            Rule(
                reactants=['dna_transcript_!%s_%s' % (Protein.POL2, Protein.RAD26)],
                products=[Protein.RAD26],
                c=self.rad26_disso
            )
        ]
        return rules_dna

    def _rules(self):
        """
        Put rules together. Although possible to use different rule sets, the single cell scale should make
        functional interactions much more likely than random intractions. Overall, interactions are slower
        on a single-cell scale.
        :return None
        """
        self.rules[0].extend(self._rules_random())
        self.rules[0].extend(self._rules_nouv())


class RepairHigh:
    def __init__(self, gille_proteins):
        self.gille_proteins = gille_proteins

        self.elong_speed = 400
        self.chip_norm = 1e5

        self.random_chip = 2.6   # TODO Replace made up value
        self.disso_const = 1.   # TODO Replace made up value
        self.rad3_cpd_chip = 6.1  # TODO Replace made up value
        self.rad26_cpd_chip = 3.1  # TODO Replace made up value
        self.rad4_cpd_chip = 1.7  # TODO Replace made up value
        self.pol2_trans_chip = 7.01  # TODO Replace made up value

        self.random_asso = None
        self.random_disso = None
        self._random()

        self.rad3_cpd_asso_rm = None
        self.rad3_cpd_asso_bt = None
        self.rad3_cpd_asso_gg = None
        self.rad3_cp_disso = None
        self._rad3()

        self.pol2_trans_c = None
        self.pol2_disso = None
        self._pol2()

        self.rad26_cpd_asso = None
        self.rad26_cpd_disso = None
        self._rad26()

        self.rad4_cpd_asso = None
        self._rad4()

        self.rules = [[]]
        self._rules()

    def _random(self, factor=.1, pol2_factor=.0):
        self.random_asso = self.random_chip * factor / (self.chip_norm * LENGTH)
        self.random_disso = self.disso_const / float(LENGTH)
        self.pol2_random_asso = self.random_asso * pol2_factor

    def _rad3(self, factor=1.):
        self.rad3_cpd_asso_rm = self.rad3_cpd_chip * factor / (self.chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))
        self.rad3_cpd_asso_bt = self.rad3_cpd_asso_rm
        self.rad3_cp_disso = self.disso_const / (DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0])
        self.rad3_cpd_asso_gg = self.rad3_cpd_asso_rm

    def _pol2(self, factor=.1):
        self.pol2_trans_c = self.pol2_trans_chip * factor / float(
            self.chip_norm
            * (
                    DEFAULT_DNA_SPEC_1DIM['cp'][1] - DEFAULT_DNA_SPEC_1DIM['cp'][0]
                    + DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0]
            )
        )
        self.pol2_disso = self.disso_const / (DEFAULT_DNA_SPEC_1DIM['tss'][1] - DEFAULT_DNA_SPEC_1DIM['tss'][0])

    def _rad26(self, factor=1.):
        self.rad26_cpd_asso = self.rad26_cpd_chip * factor / (self.chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))
        self.rad26_disso = self.disso_const / (
                DEFAULT_DNA_SPEC_1DIM['transcript'][1] - DEFAULT_DNA_SPEC_1DIM['transcript'][0])

    def _rad4(self, factor=1.):
        self.rad4_cpd_asso = self.rad4_cpd_chip * factor / (self.chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))

    def _rules_random(self):
        """
        Random association/dissociation. Exclude Pol2.
        :return List with random association/dissociation rules
        """
        rules_cpd_random = [
            Rule(reactants=['!dna_%s' % gp, gp], products=['dna_%s' % gp], c=self.random_asso)
            for gp in self.gille_proteins if gp != Protein.POL2
        ]

        rules_cpd_random.extend([
            Rule(reactants=['dna_%s' % gp], products=[gp], c=self.random_disso)
            for gp in self.gille_proteins if gp != Protein.POL2
        ])

        return rules_cpd_random

    def _rules_tc_ner(self):
        """
        TC-NER rules
        :return: List with TC-NER rules
        """
        tc_ner = [
            # Still Pol2 transcription and interaction with TSS but reduced probability
            Rule(
                reactants=['dna_cp_%s' % Protein.RAD3, '!dna_tss_%s' % Protein.ACTIVE_POL2, Protein.POL2],
                products=['dna_cp_%s' % Protein.RAD3, 'dna_tss_%s' % Protein.ACTIVE_POL2],
                c=self.pol2_trans_c
            ),
            Rule(
                reactants=['dna_tss_%s' % Protein.ACTIVE_POL2],
                products=[Protein.POL2],
                c=self.pol2_disso
            ),

            # Same dissociation probability for RAD3 from cp but no further association
            Rule(
                reactants=['dna_cp_%s' % Protein.RAD3],
                products=[Protein.RAD3],
                c=self.rad3_cp_disso
            ),

            # Rad26 association to Pol2
            # TODO is that reasonable? Or only slightly higher association to stalled Pol2
            Rule(
                reactants=[
                    'dna_transcript_%s_!%s' % (Protein.POL2, Protein.RAD26),
                    Protein.RAD26
                ],
                products=['dna_transcript_%s_%s' % (Protein.POL2, Protein.RAD26)],
                c=self.rad26_cpd_asso
            ),
            Rule(
                reactants=[
                    'dna_transcript_%s_!%s' % (Protein.ACTIVE_POL2, Protein.RAD26),
                    Protein.RAD26
                ],
                products=['dna_transcript_%s_%s' % (Protein.ACTIVE_POL2, Protein.RAD26)],
                c=self.rad26_cpd_asso
            ),

            # Create correlation between Rad26 and Pol2 due to higher dissociation of Rad26
            Rule(
                reactants=['dna_transcript_!%s_%s' % (Protein.POL2, Protein.RAD26)],
                products=[Protein.RAD26],
                c=self.rad26_disso
            ),
            Rule(
                reactants=['dna_transcript_!%s_%s' % (Protein.ACTIVE_POL2, Protein.RAD26)],
                products=[Protein.RAD26],
                c=self.rad26_disso
            ),

            # Recruitment Rad26 to lesion in TC-NER
            Rule(
                reactants=['lesion_recognised_%s' % Protein.ACTIVE_POL2, '!lesion_recognised_%s' % Protein.RAD26,
                           Protein.RAD26],
                products=['lesion_recognised_%s' % Protein.ACTIVE_POL2, 'lesion_recognised_%s' % Protein.RAD26],
                c=self.rad26_cpd_asso
            ),

            # Recruitment Rad3 TC-NER
            Rule(
                reactants=[
                    'lesion_recognised_%s' % Protein.ACTIVE_POL2,
                    'lesion_recognised_%s' % Protein.RAD26,
                    '!lesion_recognised_%s' % Protein.RAD3,
                    Protein.RAD3
                ],
                products=['lesion_opened_%s' % Protein.RAD26, 'lesion_opened_%s' % Protein.RAD3, Protein.POL2],
                # Removal
                c=self.rad3_cpd_asso_rm
            ),
            Rule(
                reactants=[
                    'lesion_recognised_%s' % Protein.ACTIVE_POL2,
                    'lesion_recognised_%s' % Protein.RAD26,
                    '!lesion_recognised_%s' % Protein.RAD3,
                    Protein.RAD3
                ],
                products=['dna_before_%s' % Protein.RAD26, 'lesion_opened_%s' % Protein.RAD3,
                          'dna_before_%s' % Protein.POL2  # Backtracking, doesn't move anymore
                          ],
                c=self.rad3_cpd_asso_bt
            ),

            # Continue recruiting Rad3 in the opened state
            Rule(
                reactants=['!lesion_opened_%s' % Protein.RAD3, Protein.RAD3],
                products=['lesion_opened_%s' % Protein.RAD3],
                c=self.rad3_cpd_asso_rm
            ),

            # Continue to remove Pol2
            Rule(
                reactants=[
                    'lesion_opened_%s' % Protein.ACTIVE_POL2,
                    'lesion_opened_%s' % Protein.RAD26,
                    'lesion_opened_%s' % Protein.RAD3
                ],
                products=[
                    'dna_before_%s' % Protein.RAD26,
                    'lesion_opened_%s' % Protein.RAD3,
                    'dna_before_%s' % Protein.POL2  # Backtracking, doesn't move anymore
                ],
                c=self.rad3_cpd_asso_bt
            ),
            Rule(
                reactants=[
                    'lesion_opened_%s' % Protein.ACTIVE_POL2,
                    'lesion_opened_%s' % Protein.RAD26,
                    '!lesion_opened_%s' % Protein.RAD3,
                    Protein.RAD3
                ],
                products=['lesion_opened_%s' % Protein.RAD26, 'lesion_opened_%s' % Protein.RAD3, Protein.POL2],
                # Removal
                c=self.rad3_cpd_asso_rm
            )
        ]

        return tc_ner

    def _rules_gg_ner(self):
        """
        GG-NER
        :return: List with GG-NER rules
        """
        gg_ner = [
            # Lesion recognition
            Rule(
                reactants=['lesion_new', Protein.RAD4],
                products=['lesion_recognised_%s' % Protein.RAD4],
                c=self.rad4_cpd_asso
            ),
            # Recruitment Rad3 GG-NER
            Rule(
                reactants=['lesion_recognised_%s' % Protein.RAD4, Protein.RAD3],
                products=[Protein.RAD4, 'lesion_opened_%s' % Protein.RAD3],
                c=self.rad3_cpd_asso_rm
            )
        ]

        return gg_ner

    def _rules_lesion(self):
        pass

    def _rules(self):
        self.rules[0].extend(self._rules_random())
        self.rules[0].extend(self._rules_tc_ner())
        self.rules[0].extend(self._rules_gg_ner())



