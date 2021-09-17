#!/usr/bin/env python3
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from modules.proteins import Protein
import matplotlib.pyplot as plt
from modules.rules import *
from modules.gillespie import PoolGillespie, DNAGillespie
from datahandler import seqDataHandler as sq


def load_transcript(tss_ratio=.1, tts_ratio=.1):
    curr_dir = os.getcwd()
    tss = pd.read_csv('%s/data/GSE49026_S-TSS.txt' % curr_dir, delimiter='\t', usecols=['chr', 'coordinate', 'ORF'])
    tss.columns = ['chr', 'start', 'ORF']
    pas = pd.read_csv('%s/data/GSE49026_S-PAS.txt' % curr_dir, delimiter='\t', usecols=['coordinate', 'ORF'])
    pas.columns = ['end', 'ORF']
    tss_pas = pd.merge(left=tss, right=pas, left_on='ORF', right_on='ORF')

    minus_trans = {
        'cp': [],
        'tss': [],
        'transcript': [],
        'tts': []
    }
    plus_trans = {
        'cp': [],
        'tss': [],
        'transcript': [],
        'tts': []
    }
    for row in tss_pas.iterrows():
        trans = row[1]
        trans_dict = plus_trans
        offs = ADD_CHROM_SIZES[trans['chr']]
        if trans['end'] < trans['start']:
            trans_dict = minus_trans
            temp = trans['end']
            trans['end'] = trans['start']
            trans['start'] = temp

        trans_dict['cp'].extend([offs + trans['start'] - 500, offs + trans['start']])
        trans_dict['tss'].extend([
            trans_dict['cp'][-1],
            offs + trans['start'] + int((trans['end'] - trans['start']) * tss_ratio)]
        )
        trans_dict['transcript'].extend([trans_dict['tss'][-1], offs + trans['end']])
        trans_dict['transcript'].extend([
            trans_dict['transcript'][-1],
            offs + trans['end'] + int((trans['end'] - trans['start']) * tts_ratio)]
        )

    return minus_trans, plus_trans


def routine_gille_pool():
    """
    Main function for Gillespie algorithm w/out the notion of space
    :return: None
    """
    rules = [
        Rule(
            reactants=[POL2, RAD26],
            products=['_'.join(sorted([POL2, RAD26]))],
            c=.5
        ),
        Rule(
            reactants=['_'.join(sorted([POL2, RAD26]))],
            products=[POL2, RAD26],
            c=.4
        )
    ]

    concentrations = {
        RAD3: 10,
        POL2: 10,
        RAD26: 10,
        IC_RAD3: 5,
        IC_POL2: 5,
        IC_RAD26: 5,
    }

    gille = PoolGillespie(protein_conc=concentrations, rules=rules)
    pol2 = []
    pol2_rad26_complex = []
    for i in range(2000):
        pol2.append(gille.get_state(POL2))
        pol2_rad26_complex.append(gille.get_state('_'.join(sorted([POL2, RAD26]))))
        print(gille.t)
        print('\n')
        gille.simulate()
        if i == 1000:
            gille.plot()

    plt.plot(pol2, label='Pol2')
    plt.plot(pol2_rad26_complex, label='Pol2:Rad26')
    plt.legend(loc='upper right')
    plt.xlabel('Update Steps')
    plt.ylabel('#Particles')
    plt.title('Protein Evolution After %.3f Sec' % gille.t)
    plt.show()


def routine_gille_dna():
    """
    Main function for Gillespie algorithm which combines the interaction of proteins in the nucleus with the DNA.
    It combines a well mixed solution w/ space dependent interaction profiles.
    :return: None
    """
    # Futcher, B., et al. "A sampling of the yeast proteome." Molecular and cellular biology 19.11 (1999): 7357-7368.
    # 4pg protein / cell and 5e7 Molecules/cell
    # Bhargava, Madhu M., and Harlyn O. Halvorson. "Isolation of nuclei from yeast." The Journal of cell biology 49.2 (1971): 423-429.
    # .717pg proteins / nucleus and 9.05 pg/nucleus ==> ratio = .717 / 9.05 approx= .08
    # Both results in line when assuming that haploid cell has also only half of the proteins
    # ==> 1e8 proteins/diploid cell ==> .6 proteins per base pair
    num_prot = int(0.6 * LENGTH)
    num_iter = 3
    radiation_time = 20.
    plot_range = (0, 1000)
    smoothing = 50
    prope_freq = 20
    max_repair_t = 100
    random_lesion = False
    verbosity = 1
    is_easy = True
    uniform_prot = False
    inv_proteins = [RAD3, POL2, RAD26] if not is_easy else [GG_COMP, TC_COMP, NUC_COMP]
    colors = ['tab:orange', 'tab:green', 'tab:cyan']

    num_probes = max_repair_t // prope_freq + 1
    protein_data_mean_minus = {prot: np.zeros((num_probes, plot_range[1] - plot_range[0])) for prot in inv_proteins}
    protein_data_sqe_minus = {prot: np.zeros((num_probes, plot_range[1] - plot_range[0])) for prot in inv_proteins}
    protein_data_mean_plus = {prot: np.zeros((num_probes, plot_range[1] - plot_range[0])) for prot in inv_proteins}
    protein_data_sqe_plus = {prot: np.zeros((num_probes, plot_range[1] - plot_range[0])) for prot in inv_proteins}
    cpd_data_mean_minus = np.zeros((num_probes, plot_range[1] - plot_range[0]))
    cpd_data_sqe_minus = np.zeros((num_probes, plot_range[1] - plot_range[0]))
    cpd_data_mean_plus = np.zeros((num_probes, plot_range[1] - plot_range[0]))
    cpd_data_sqe_plus = np.zeros((num_probes, plot_range[1] - plot_range[0]))
    lesion_states_mean = np.zeros((num_probes, 3 if is_easy else 6))
    lesion_states_sqe = np.zeros((num_probes, 3 if is_easy else 6))

    gille_proteins = Protein.get_types_gillespie() if not is_easy else Protein.get_types_easy()

    if uniform_prot:
        concentrations_pool = {gp: int(num_prot / float(len(gille_proteins))) for gp in gille_proteins}
    else:
        gille_proteins = Protein.get_types_easy()
        concentrations_pool = {GG_COMP: int(num_prot * .3), TC_COMP: int(num_prot * .3), NUC_COMP: int(num_prot * .4)}
    rules_pool = PoolNoInteract()

    if not is_easy:
        gp_sim = deepcopy(gille_proteins)
        gp_sim.append(ACTIVE_POL2)
        nouv_rules = NoUVHigh(gille_proteins=gille_proteins)
        damage_response = RepairHigh(gille_proteins=gille_proteins)
    else:
        gp_sim = deepcopy(gille_proteins)
        nouv_rules = RulesBiMNoUV(num_particles=num_prot)
        damage_response = RulesBiMolUV(num_particles=num_prot)

    cpd_distribution = []
    for t in range(num_iter):
        def update_mean_sqe(values, mean, sqe):
            delta = values - mean
            mean += delta / (t + 1)
            upd_delta = values - mean
            sqe += delta * upd_delta
            return mean, sqe

        def print_associated_proteins(p_idx):
            num_asso_prot = np.sum(gille_plus.state[:, p_idx]) + np.sum(gille_minus.state[:, p_idx])
            prot_name = gille_proteins[p_idx]
            print('Num associated %s: %s' % (prot_name, num_asso_prot))

        if verbosity > 0:
            print('%s' % t)
        gille_pool = PoolGillespie(protein_conc=concentrations_pool, rules=rules_pool.rules)

        # All data is stored [minus, plus, minus, plus ...]
        gille_minus = DNAGillespie(
            gille_pool,
            size=LENGTH,
            dna_spec=DEFAULT_DNA_SPEC_1DIM.copy(),
            protein_names=gp_sim,
            rules=nouv_rules.rules,
            elong_speed=nouv_rules.elong_speed
        )
        gille_plus = DNAGillespie(
            gille_pool,
            size=LENGTH,
            dna_spec=DEFAULT_DNA_SPEC_1DIM.copy(),
            protein_names=gp_sim,
            rules=nouv_rules.rules,
            elong_speed=nouv_rules.elong_speed
        )

        i = 0
        is_radiated = False
        radiation_t = -1

        counter = 0
        num_lesions = 0
        while True:
            i += 1
            if verbosity > 0:
                print('Time point: %.3f min' % gille_minus.t)
                print('\n')
            # Use the same resource pool. To avoid bias, random selection of which DNA strand is simulated first
            if np.random.random() > .5:
                gille_plus.simulate(max_iter=100, is_easy=is_easy)
                gille_minus.simulate(max_iter=100, is_easy=is_easy)
            else:
                gille_minus.simulate(max_iter=100, is_easy=is_easy)
                gille_plus.simulate(max_iter=100, is_easy=is_easy)
            time = np.maximum(gille_plus.t, gille_minus.t)
            gille_minus.t = time
            gille_plus.t = time

            if is_radiated:
                if np.around(time - radiation_t, -1) // prope_freq == counter:
                    if verbosity > 2:
                        print('Gillespiel Pool: ', gille_pool.state)
                        print_associated_proteins(0)
                        print_associated_proteins(1)
                        print_associated_proteins(2)

                    if verbosity > 1:
                        print('Plot +/- Strand')
                        fig, ax = plt.subplots(2, 1, figsize=(8, 7))
                        gille_minus.plot(proteins=inv_proteins, colors=colors, axis=ax[0])
                        gille_plus.plot(proteins=inv_proteins, colors=colors, axis=ax[1])
                        ax[0].set_title('- Strand')
                        ax[1].set_title('+ Strand')
                        fig.suptitle('Smoothed ChIP-seq Simulation')
                        fig.tight_layout()
                        plt.show()

                    for prot in protein_data_mean_minus.keys():
                        protein_data_mean_minus[prot][counter], protein_data_sqe_minus[prot][counter] = update_mean_sqe(
                            gille_minus.get_protein_state(prot, start=plot_range[0], end=plot_range[1]),
                            protein_data_mean_minus[prot][counter],
                            protein_data_sqe_minus[prot][counter]
                        )
                        protein_data_mean_plus[prot][counter], protein_data_sqe_plus[prot][counter] = update_mean_sqe(
                            gille_plus.get_protein_state(prot, start=plot_range[0], end=plot_range[1]),
                            protein_data_mean_plus[prot][counter],
                            protein_data_sqe_plus[prot][counter]
                        )

                    cpd_data_mean_minus[counter], cpd_data_sqe_minus[counter] = update_mean_sqe(
                        gille_minus.get_lesion_state(start=plot_range[0], end=plot_range[1]),
                        cpd_data_mean_minus[counter],
                        cpd_data_sqe_minus[counter]
                    )
                    cpd_data_mean_plus[counter], cpd_data_sqe_plus[counter] = update_mean_sqe(
                        gille_plus.get_lesion_state(start=plot_range[0], end=plot_range[1]),
                        cpd_data_mean_plus[counter],
                        cpd_data_sqe_plus[counter]
                    )

                    lesion_states = np.zeros(3 if is_easy else 6)
                    for lesion_minus, lesion_plus in zip(gille_minus.lesions, gille_plus.lesions):
                        lesion_states[np.minimum(lesion_minus.state, 2 if is_easy else 5)] += 1
                        lesion_states[np.minimum(lesion_plus.state, 2 if is_easy else 5)] += 1
                    repaired = num_lesions - len(gille_minus.lesions) - len(gille_plus.lesions)
                    lesion_states[-1] += repaired
                    lesion_states_mean[counter], lesion_states_sqe[counter] = update_mean_sqe(
                        lesion_states / num_lesions,
                        lesion_states_mean[counter],
                        lesion_states_sqe[counter]
                    )
                    # num_lesions -= repaired
                    counter += 1

                if time - radiation_t > max_repair_t:
                    break
            else:
                if gille_minus.t > radiation_time:
                    if verbosity > 2:
                        print('Gillespiel Pool: ', gille_pool.state)
                        print_associated_proteins(0)
                        print_associated_proteins(1)
                        print_associated_proteins(2)

                    if verbosity > 1:
                        print('Plot +/- Strand')
                        fig, ax = plt.subplots(2, 1, figsize=(8, 7))
                        gille_minus.plot(proteins=inv_proteins, colors=colors, axis=ax[0])
                        gille_plus.plot(proteins=inv_proteins, colors=colors, axis=ax[1])
                        fig.suptitle('Smoothed ChIP-seq Simulation')
                        ax[0].set_title('- Strand')
                        ax[1].set_title('+ Strand')
                        fig.tight_layout()
                        plt.show()

                    if verbosity > 0:
                        print('##################### UV RADIATION')

                    radiation_t = time
                    gille_minus.set_rules(damage_response.rules, damage_response.elong_speed)
                    gille_plus.set_rules(damage_response.rules, damage_response.elong_speed)
                    if random_lesion:
                        cpd_start = np.random.choice(LENGTH, size=(NUM_RANDOM_CPD, 2))
                        num_lesions = NUM_RANDOM_CPD * 2
                    else:
                        cpd_start = DEFAULT_CPD
                        num_lesions = len(DEFAULT_CPD) * 2
                    for cpd in cpd_start:
                        if random_lesion:
                            cpd_distribution.append(np.arange(cpd[0], cpd[0] + DEFAULT_CPD_LENGTH))
                            gille_minus.add_lesion(cpd, cpd + DEFAULT_CPD_LENGTH)
                            cpd_distribution.append(np.arange(cpd[1], cpd[1] + DEFAULT_CPD_LENGTH))
                            gille_plus.add_lesion(cpd, cpd + DEFAULT_CPD_LENGTH)
                        else:
                            cpd_distribution.append(np.arange(cpd, cpd + DEFAULT_CPD_LENGTH))
                            gille_minus.add_lesion(cpd, cpd + DEFAULT_CPD_LENGTH)
                            gille_plus.add_lesion(cpd, cpd + DEFAULT_CPD_LENGTH)
                    gille_minus.reaction_prob()
                    gille_plus.reaction_prob()
                    is_radiated = True

    plt.rcParams.update({'font.size': 8})
    fig_cpd, ax_cpd = plt.subplots(num_probes, 2, figsize=(8, 7))
    # Plot CPD
    for num, (mean_minus, sqe_minus, mean_plus, sqe_plus) in enumerate(zip(cpd_data_mean_minus, cpd_data_sqe_minus,
                                                                           cpd_data_mean_plus, cpd_data_sqe_plus)):
        if num_iter < 2:
            var_minus = np.zeros(plot_range[1] - plot_range[0])
            var_plus = np.zeros(plot_range[1] - plot_range[0])
        else:
            var_minus = sq.smooth(sqe_minus / float(num_iter - 1), smooth_size=smoothing)
            var_plus = sq.smooth(sqe_plus / float(num_iter - 1), smooth_size=smoothing)

        mean_minus = sq.smooth(mean_minus, smooth_size=smoothing)
        ax_cpd[num, 0].plot(np.arange(plot_range[0], plot_range[1]), mean_minus, color='tab:blue')
        ax_cpd[num, 0].fill_between(
            np.arange(plot_range[0], plot_range[1]),
            np.maximum(mean_minus - var_minus, 0),
            mean_minus + var_minus,
            alpha=.2
        )
        mean_plus = sq.smooth(mean_plus, smooth_size=smoothing)
        ax_cpd[num, 1].plot(np.arange(plot_range[0], plot_range[1]), mean_plus, color='tab:blue')
        ax_cpd[num, 1].fill_between(
            np.arange(plot_range[0], plot_range[1]),
            np.maximum(mean_plus - var_plus, 0),
            mean_plus + var_plus,
            alpha=.2
        )
        ax_cpd[num, 0].set_ylabel('Average Profile')
        ax_cpd[num, 0].set_xlabel('Sequence')
        ax_cpd[num, 1].set_xlabel('Sequence')
        ax_cpd[num, 0].set_title('CPD - %s' % (num * prope_freq))
        ax_cpd[num, 1].set_title('CPD + %s' % (num * prope_freq))

    fig_cpd.tight_layout()
    plt.show()

    # Plot proteins
    for protein, c in zip(protein_data_mean_minus.keys(), colors):
        fig_prot, ax_prot = plt.subplots(num_probes, 2, figsize=(8, 7))
        for num, (mean_minus, sqe_minus, mean_plus, sqe_plus) in enumerate(
                zip(protein_data_mean_minus[protein], protein_data_sqe_minus[protein],
                    protein_data_mean_plus[protein], protein_data_sqe_plus[protein])):
            if num_iter < 2:
                var_minus = np.zeros(plot_range[1] - plot_range[0])
                var_plus = np.zeros(plot_range[1] - plot_range[0])
            else:
                var_minus = sq.smooth(sqe_minus / float(num_iter - 1), smooth_size=smoothing)
                var_plus = sq.smooth(sqe_plus / float(num_iter - 1), smooth_size=smoothing)

            mean_minus = sq.smooth(mean_minus, smooth_size=smoothing)
            ax_prot[num, 0].plot(np.arange(plot_range[0], plot_range[1]), mean_minus, color=c)
            ax_prot[num, 0].fill_between(
                np.arange(plot_range[0], plot_range[1]),
                np.maximum(mean_minus - var_minus, 0),
                mean_minus + var_minus,
                alpha=.2,
                color=c
            )
            mean_plus = sq.smooth(mean_plus, smooth_size=smoothing)
            ax_prot[num, 1].plot(np.arange(plot_range[0], plot_range[1]), mean_plus, color=c)
            ax_prot[num, 1].fill_between(
                np.arange(plot_range[0], plot_range[1]),
                np.maximum(mean_plus - var_plus, 0),
                mean_plus + var_plus,
                alpha=.2,
                color=c
            )
            ax_prot[num, 0].set_ylabel('Average Profile')
            ax_prot[num, 0].set_xlabel('Sequence')
            ax_prot[num, 1].set_xlabel('Sequence')
            ax_prot[num, 0].set_title('%s - %s' % (protein, num * prope_freq))
            ax_prot[num, 1].set_title('%s + %s' % (protein, num * prope_freq))

        fig_prot.tight_layout()
        plt.show()

    plt.figure(figsize=(8, 7))
    lesion_types = ['new', 'recognised', 'removed'] if is_easy else CPD_STATES.keys()
    for num, (lesion_mean, lesion_sqe, lt) in enumerate(zip(lesion_states_mean.T, lesion_states_sqe.T, lesion_types)):
        if num_iter < 2:
            var = np.zeros(lesion_mean.size)
        else:
            var = lesion_sqe / float(num_iter - 1)

        plt.plot(np.arange(lesion_mean.size) * prope_freq, lesion_mean, label=lt)
        plt.fill_between(
            np.arange(lesion_mean.size) * prope_freq,
            lesion_mean - var,
            lesion_mean + var,
            alpha=.2
        )

    plt.xlabel('Time(min)')
    plt.ylabel('Average Lesion States')
    plt.title('Lesion Progression')
    plt.legend(loc='upper right')
    plt.show()


def main():
    """
    Main function which either runs the Gillespie algorithm w/out the notion of space, or the spatial dependent
    interaction profiles.
    :return: None
    """
    run_pool = False
    if run_pool:
        routine_gille_pool()
    else:
        routine_gille_dna()


if __name__ == '__main__':
    main()

