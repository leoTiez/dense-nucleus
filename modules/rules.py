#!/usr/bin/env python3
# ####################################################
# Pivotal definitions
# Time unit = minute
# ####################################################
from modules.proteins import *
import numpy as np

# CONSTANTS
LENGTH = 1000
DEFAULT_DNA_SPEC_1DIM = {
    'cp': [0, 10, 140, 150, 255, 260, 330, 360, 500, 550, 710, 720],
    'tss': [10, 15, 150, 155, 260, 270, 360, 390, 550, 600, 720, 725],
    'transcript': [15, 90, 155, 180, 270, 300, 390, 500, 600, 620, 725, 820],
    'tts': [90, 140, 180, 255, 300, 330, 390, 500, 600, 710, 820, LENGTH]
}

CPD_STATES = {
    'new': 0,
    'recognised': 1,
    'opened': 2,
    'incised': 3,
    'replaced': 4,
    'removed': 5
}

ADD_CHROM_SIZES = {
    'chrI': 0,
    'chrII': 230218,
    'chrIII': 1043402,
    'chrIV': 1360022,
    'chrIX': 2891955,
    'chrM': 3331843,
    'chrV': 3417622,
    'chrVI': 3994496,
    'chrVII': 4264657,
    'chrVIII': 5355597,
    'chrX': 5918240,
    'chrXI': 6663991,
    'chrXII': 7330807,
    'chrXIII': 8408984,
    'chrXIV': 9333415,
    'chrXV': 10117748,
    'chrXVI': 11209039
}

DEFAULT_CPD_LENGTH = 2
BACKTRACKING_LENGTH = 5
DEFAULT_CPD = [30, 125, 140, 200, 501, 750, 755, 900]
NUM_RANDOM_CPD = 2


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

        self.random_chip = .3
        self.rad3_cp_chip = .35

        self.pol2_trans_chip = .4
        self.rad26_trans_chip = .4
        self.disso_const = 1.

        self.random_asso = None
        self.random_disso = None
        self.pol2_random_asso = None
        self._random(pol2_factor=1.)

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
            for gp in self.gille_proteins # if gp != POL2
        ]

        rules_random.append(
            Rule(
                reactants=['dna_!%s_!%s' % (ACTIVE_POL2, POL2), POL2],
                products=['dna_!%s_%s' % (ACTIVE_POL2, POL2)],
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
                reactants=['!dna_cp_%s' % RAD3, RAD3],
                products=['dna_cp_%s' % RAD3], c=self.rad3_cp_asso
            ),
            Rule(reactants=['dna_cp_%s' % RAD3], products=[RAD3], c=self.rad3_cp_disso),
            # Pol2 associating to the TSS if rad3 present at the core promoter bu
            Rule(
                reactants=['dna_cp_%s' % RAD3, '!dna_tss_%s' % ACTIVE_POL2, POL2],
                products=['dna_cp_%s' % RAD3, 'dna_tss_%s' % ACTIVE_POL2],
                c=self.pol2_trans_c
            ),
            # Pol2 dissociation from TSS
            Rule(
                reactants=['dna_tss_%s' % ACTIVE_POL2],
                products=[POL2],
                c=self.pol2_disso
            ),
            # Rad26 association to inactive Pol2. Similar to the verification step of whether pausing is due to
            # a lesion
            Rule(
                reactants=[
                    'dna_transcript_%s_!%s' % (POL2, RAD26),
                    RAD26
                ],
                products=['dna_transcript_%s_%s' % (POL2, RAD26)],
                c=self.rad26_asso
            ),
            # Dissociation of Rad26 if no inactive Pol2 is present
            Rule(
                reactants=['dna_transcript_!%s_%s' % (POL2, RAD26)],
                products=[RAD26],
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

        # TODO TOO SLOW
        self.random_chip = .01   # TODO Replace made up value
        self.disso_const = 1.   # TODO Replace made up value
        self.rad3_cpd_chip = .24  # TODO Replace made up value
        self.rad26_cpd_chip = 0.12  # TODO Replace made up value
        self.rad4_cpd_chip = .12  # TODO Replace made up value
        self.pol2_trans_chip = .12  # TODO Replace made up value
        self.rad2_cpd_chip = 1.8  # TODO Replace made up value
        self.rad10_cpd_chip = 1.8  # TODO Replace made up value
        self.poly_cpd_chip = 1.8  # TODO Replace made up value
        self.ligase_cpd_chip = 1.8  # TODO Replace made up value

        self.random_asso = None
        self.random_disso = None
        self._random(factor=1.6, pol2_factor=1.6)#.1)

        self.rad3_cpd_asso_rm = None
        self.rad3_cpd_asso_bt = None
        self.rad3_cpd_asso_gg = None
        self.rad3_cp_disso = None
        self._rad3(factor=1.6)#.4)

        self.pol2_trans_c = None
        self.pol2_disso = None
        self._pol2(factor=1.6)#.4)

        self.rad26_cpd_asso = None
        self.rad26_cpd_disso = None
        self._rad26(factor=1.6)#.4)

        self.rad4_cpd_asso = None
        self._rad4(factor=1.6)#.4)

        self.rad2_cpd_asso = None
        self.rad10_cpd_asso = None
        self._rad2_rad10(factor=1.6)#.4)

        self.poly_cpd_asso = None
        self.poly_repair_disso = None
        self._dna_polymerase(factor=1.6)#.4)

        self.ligase_cpd_asso = None
        self._dna_ligase(factor=1.6)#.4)

        self.rules = [[]]
        self._rules()

    def _random(self, factor=1., pol2_factor=.0):
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

    def _rad2_rad10(self, factor=1.):
        self.rad2_cpd_asso = self.rad2_cpd_chip * factor / (self.chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))
        self.rad10_cpd_asso = self.rad10_cpd_chip * factor / (self.chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))

    def _dna_polymerase(self, factor=1.):
        self.poly_cpd_asso = self.poly_cpd_chip * factor / (self.chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))
        self.poly_repair_disso = self.poly_cpd_asso

    def _dna_ligase(self, factor=1.):
        self.ligase_cpd_asso = self.ligase_cpd_chip * factor / (self.chip_norm * (DEFAULT_CPD[1] - DEFAULT_CPD[0]))

    def _rules_random(self):
        """
        Random association/dissociation. Exclude Pol2.
        :return List with random association/dissociation rules
        """
        rules_cpd_random = [
            Rule(reactants=['!dna_%s' % gp, gp], products=['dna_%s' % gp], c=self.random_asso)
            for gp in self.gille_proteins #if gp != POL2
        ]

        rules_cpd_random.extend([
            Rule(reactants=['dna_%s' % gp], products=[gp], c=self.random_disso)
            for gp in self.gille_proteins #if gp != POL2
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
                reactants=['dna_cp_%s' % RAD3, '!dna_tss_%s' % ACTIVE_POL2, POL2],
                products=['dna_cp_%s' % RAD3, 'dna_tss_%s' % ACTIVE_POL2],
                c=self.pol2_trans_c
            ),
            Rule(
                reactants=['dna_tss_%s' % ACTIVE_POL2],
                products=[POL2],
                c=self.pol2_disso
            ),

            # Same dissociation probability for RAD3 from cp but no further association
            Rule(
                reactants=['dna_cp_%s' % RAD3],
                products=[RAD3],
                c=self.rad3_cp_disso
            ),

            # Rad26 association to Pol2
            # TODO is that reasonable? Or only slightly higher association to stalled Pol2
            Rule(
                reactants=[
                    'dna_transcript_%s_!%s' % (POL2, RAD26),
                    RAD26
                ],
                products=['dna_transcript_%s_%s' % (POL2, RAD26)],
                c=self.rad26_cpd_asso
            ),
            Rule(
                reactants=[
                    'dna_transcript_%s_!%s' % (ACTIVE_POL2, RAD26),
                    RAD26
                ],
                products=['dna_transcript_%s_%s' % (ACTIVE_POL2, RAD26)],
                c=self.rad26_cpd_asso
            ),

            # Create correlation between Rad26 and Pol2 due to higher dissociation of Rad26
            Rule(
                reactants=['dna_transcript_!%s_%s' % (POL2, RAD26)],
                products=[RAD26],
                c=self.rad26_disso
            ),
            Rule(
                reactants=['dna_transcript_!%s_%s' % (ACTIVE_POL2, RAD26)],
                products=[RAD26],
                c=self.rad26_disso
            ),

            # Recruitment Rad26 to lesion in TC-NER
            Rule(
                reactants=['lesion_recognised_%s' % ACTIVE_POL2, '!lesion_recognised_%s' % RAD26,
                           RAD26],
                products=['lesion_recognised_%s' % ACTIVE_POL2, 'lesion_recognised_%s' % RAD26],
                c=self.rad26_cpd_asso
            ),

            # Recruitment Rad3 TC-NER
            Rule(
                reactants=[
                    'lesion_recognised_%s' % ACTIVE_POL2,
                    'lesion_recognised_%s' % RAD26,
                    '!lesion_recognised_%s' % RAD3,
                    RAD3
                ],
                products=['lesion_opened_%s' % RAD26, 'lesion_opened_%s' % RAD3, POL2],
                # Removal
                c=self.rad3_cpd_asso_rm
            ),
            Rule(
                reactants=[
                    'lesion_recognised_%s' % ACTIVE_POL2,
                    'lesion_recognised_%s' % RAD26,
                    '!lesion_recognised_%s' % RAD3,
                    RAD3
                ],
                products=['dna_before_%s' % RAD26, 'lesion_opened_%s' % RAD3,
                          'dna_before_%s' % POL2  # Backtracking, doesn't move anymore
                          ],
                c=self.rad3_cpd_asso_bt
            ),

            # Continue recruiting Rad3 in the opened state
            Rule(
                reactants=['!lesion_opened_%s' % RAD3, RAD3],
                products=['lesion_opened_%s' % RAD3],
                c=self.rad3_cpd_asso_rm
            ),

            # Continue to remove Pol2
            Rule(
                reactants=[
                    'lesion_opened_%s' % ACTIVE_POL2,
                    'lesion_opened_%s' % RAD26,
                    'lesion_opened_%s' % RAD3
                ],
                products=[
                    'dna_before_%s' % RAD26,
                    'lesion_opened_%s' % RAD3,
                    'dna_before_%s' % POL2  # Backtracking, doesn't move anymore
                ],
                c=self.rad3_cpd_asso_bt
            ),
            Rule(
                reactants=[
                    'lesion_opened_%s' % ACTIVE_POL2,
                    'lesion_opened_%s' % RAD26,
                    '!lesion_opened_%s' % RAD3,
                    RAD3
                ],
                products=['lesion_opened_%s' % RAD26, 'lesion_opened_%s' % RAD3, POL2],
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
                reactants=['lesion_new', RAD4],
                products=['lesion_recognised_%s' % RAD4],
                c=self.rad4_cpd_asso
            ),
            # Recruitment Rad3 GG-NER and Rad4 removal
            Rule(
                reactants=['lesion_recognised_%s' % RAD4, RAD3],
                products=[RAD4, 'lesion_opened_%s' % RAD3],
                c=self.rad3_cpd_asso_rm
            )
        ]

        return gg_ner

    def _rules_incision(self):
        incision = [
            # Recruitment of Rad10
            Rule(
                reactants=[
                    'lesion_opened_%s' % RAD3,
                    'lesion_opened_%s' % RAD2,
                    RAD10
                ],
                products=['lesion_opened_%s' % RAD3, 'lesion_opened_%s' % RAD10],
                c=self.rad10_cpd_asso
            ),
            Rule(
                reactants=[
                    'lesion_opened_%s' % RAD3,
                    'lesion_opened_%s' % RAD2,
                    '!lesion_opened_%s' % RAD10,
                    RAD10
                ],
                products=[
                    'lesion_incised_%s' % RAD3,
                    'lesion_incised_%s' % RAD10,
                    'lesion_incised_%s' % RAD2
                ],
                c=self.rad10_cpd_asso
            ),

            # Recruitment of Rad2
            Rule(
                reactants=[
                    'lesion_opened_%s' % RAD3,
                    '!lesion_opened_%s' % RAD10,
                    RAD2
                ],
                # Don't need to be at the exact same position. Sufficient to be on the lesion
                products=['lesion_opened_%s' % RAD3, 'lesion_opened_%s' % RAD2],
                c=self.rad2_cpd_asso
            ),
            Rule(
                reactants=[
                    'lesion_opened_%s' % RAD3,
                    'lesion_opened_%s' % RAD10,
                    RAD2
                ],
                products=[
                    'lesion_incised_%s' % RAD3,
                    'lesion_incised_%s' % RAD2,
                    'lesion_incised_%s' % RAD10
                ],  # Don't need to be at the exact same position. Sufficient to be on the lesion
                c=self.rad2_cpd_asso
            )
        ]

        return incision

    def _rules_gap_filling(self):
        gap_filling = [
            # Recruitment of DNA Polymerase
            Rule(
                reactants=[
                    'lesion_incised_%s' % RAD3,
                    'lesion_incised_%s' % RAD2,
                    'lesion_incised_%s' % RAD10,
                    DNA_POL
                ],
                products=['lesion_replaced_%s' % DNA_POL, RAD3, RAD2, RAD10],
                c=self.poly_cpd_asso
            ),
            # Removal of repair proteins
            # Recruitment of DNA Polymerase
            Rule(
                reactants=[
                    'lesion_replaced_%s' % RAD3,
                    'lesion_replaced_%s' % RAD2,
                    'lesion_replaced_%s' % RAD10,
                    DNA_POL
                ],
                products=['lesion_replaced_%s' % DNA_POL, RAD3, RAD2, RAD10],
                c=self.poly_repair_disso
            )
        ]
        return gap_filling

    def _rules_sealing(self):
        sealing = [
            Rule(
                reactants=['lesion_replaced_%s' % DNA_POL, DNA_LIG],
                products=['lesion_removed_%s' % DNA_LIG, DNA_POL],
                c=self.ligase_cpd_asso
            )
        ]
        return sealing

    def _rules(self):
        self.rules[0].extend(self._rules_random())
        self.rules[0].extend(self._rules_tc_ner())
        self.rules[0].extend(self._rules_gg_ner())
        self.rules[0].extend(self._rules_incision())
        self.rules[0].extend(self._rules_gap_filling())
        self.rules[0].extend(self._rules_sealing())


class RulesBiMNoUV:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.gille_proteins = Protein.get_types_easy()
        self.elong_speed = 1000
        self.random_asso = None
        self.random_disso = None
        self._prob_random()

        self.gg_asso = None
        self.gg_disso = None
        self._prob_gg()

        self.tc_asso = None
        self.tc_disso = None
        self._prob_tc()

        self.rules = [[]]
        self._rules()

    def _prob_random(self):
        self.random_asso = 9e-3 / float(LENGTH)
        self.random_disso = 1.8e-1

    def _prob_gg(self):
        self.gg_asso = 9e-3 / float(LENGTH)
        self.gg_disso = 1.8e-1

    def _prob_tc(self):
        self.tc_asso = 6e-1 / float(LENGTH)
        self.tc_disso = 4e-1

    def _rules_random(self):
        rules_random = [
            Rule(reactants=['!dna_%s' % gp, gp], products=['dna_%s' % gp], c=self.random_asso)
            for gp in self.gille_proteins
        ]

        rules_random.extend([
            Rule(reactants=['dna_%s' % gp], products=[gp], c=self.random_disso)
            for gp in self.gille_proteins
        ])

        return rules_random

    def _rules_nouv(self):
        rules_dna = [
            # TC association to the TSS or CP
            Rule(
                reactants=['!dna_tss_%s' % TC_COMP, TC_COMP],
                products=['dna_tss_%s' % TC_COMP],
                c=self.tc_asso
            ),
            Rule(
                reactants=['!dna_cp_%s' % TC_COMP, TC_COMP],
                products=['dna_cp_%s' % TC_COMP],
                c=self.tc_asso
            ),
            # TC dissociation from TSS
            Rule(
                reactants=['dna_tts_%s' % TC_COMP],
                products=[TC_COMP],
                c=self.tc_disso
            ),
            # Random association / dissociation GG
            Rule(
                reactants=['!dna_%s' % GG_COMP, GG_COMP],
                products=['dna_%s' % GG_COMP],
                c=self.gg_asso
            ),
            Rule(
                reactants=['dna_%s' % GG_COMP],
                products=[GG_COMP],
                c=self.gg_disso
            ),
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


class RulesBiMolUV:
    def __init__(self, num_particles, require_double=False):
        self.num_particles = num_particles
        self.gille_proteins = Protein.get_types_easy()
        self.elong_speed = 200
        self.random_asso = None
        self.random_disso = None
        self.require_double = require_double
        self._prob_random()

        self.gg_asso = None
        self.gg_disso = None
        self.gg_disso_rep = None
        self._prob_gg()

        self.tc_asso = None
        self.tc_disso = None
        self._prob_tc()

        self.nuc_asso = None
        self.nuc_disso = None
        self._prob_nuc()

        self.rules = [[]]
        self._rules()

    def _prob_random(self):
        self.random_asso = 9e-3 / float(LENGTH)
        self.random_disso = 1.8e-1

    def _prob_gg(self):
        self.gg_asso = 9e-3 / float(LENGTH)
        self.gg_disso = 1.8e-1
        self.gg_rec = 1.
        self.gg_disso_lesion = 4e-2

    def _prob_tc(self):
        self.tc_asso = 1e-1 / float(LENGTH)
        self.tc_disso = 2e-1

    def _prob_nuc(self):
        if self.require_double:
            self.nuc_asso = 1. / float(LENGTH)
        else:
            self.nuc_asso = 5e-1 / float(LENGTH)
        self.nuc_disso = 2e-1
        self.nuc_convert = 1.

    def _rules_random(self):
        rules_random = [
            Rule(reactants=['!dna_%s' % gp, gp], products=['dna_%s' % gp], c=self.random_asso)
            for gp in self.gille_proteins
        ]

        rules_random.extend([
            Rule(reactants=['dna_%s' % gp], products=[gp], c=self.random_disso)
            for gp in self.gille_proteins
        ])

        return rules_random

    def _rules_uv(self):
        rules_dna = [
            Rule(
                reactants=['!dna_tss_%s' % TC_COMP, TC_COMP],
                products=['dna_tss_%s' % TC_COMP],
                c=self.tc_asso
            ),
            # TC dissociation from TSS
            Rule(
                reactants=['dna_tts_%s' % TC_COMP],
                products=[TC_COMP],
                c=self.tc_disso
            ),
            # Lesion recognition GG
            Rule(
                reactants=['!dna_%s' % GG_COMP, GG_COMP],
                products=['dna_%s' % GG_COMP],
                c=self.gg_asso
            ),
            Rule(
                reactants=['lesion_%s' % GG_COMP],
                products=['lesion_recognised_%s' % GG_COMP],
                c=self.gg_rec
            ),
            Rule(
                reactants=['lesion_recognised_%s' % GG_COMP],
                products=['lesion_recognised', GG_COMP],
                c=self.gg_disso_lesion
            ),
            # Reset dissociation for GG when lesion repaired
            Rule(
                reactants=['!lesion_%s' % GG_COMP],
                products=[GG_COMP],
                c=self.gg_disso
            ),
        ]
        return rules_dna

    def _rules_nuc(self):
        if self.require_double:
            rules_nuc = [
                Rule(
                    reactants=['lesion_recognised_%s' % GG_COMP, NUC_COMP],
                    products=['lesion_removed_%s_%s' % (GG_COMP, NUC_COMP)],
                    c=self.nuc_asso
                ),
                Rule(
                    reactants=['lesion_recognised_%s' % TC_COMP, NUC_COMP],
                    products=['lesion_removed_%s_%s' % (TC_COMP, NUC_COMP)],
                    c=self.nuc_asso
                )
            ]
        else:
            rules_nuc = [
                Rule(
                    reactants=['lesion_recognised', NUC_COMP],
                    products=['lesion_removed_%s' % NUC_COMP],
                    c=self.nuc_asso
                )
            ]

        rules_nuc.extend([
            Rule(
                reactants=['lesion_recognised_%s' % NUC_COMP],
                products=['lesion_removed_%s' % NUC_COMP],
                c=self.nuc_convert
            ),
            Rule(
                reactants=['dna_%s' % NUC_COMP],
                products=[NUC_COMP],
                c=self.nuc_disso
            )
        ])

        return rules_nuc

    def _rules(self):
        """
        Put rules together. Although possible to use different rule sets, the single cell scale should make
        functional interactions much more likely than random intractions. Overall, interactions are slower
        on a single-cell scale.
        :return None
        """
        self.rules[0].extend(self._rules_random())
        self.rules[0].extend(self._rules_uv())
        self.rules[0].extend(self._rules_nuc())

