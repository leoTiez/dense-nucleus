#!/usr/bin/env python3
from copy import deepcopy
import numpy as np
from modules.proteins import Protein
import matplotlib.pyplot as plt
from modules.rules import *
from modules.gillespie import PoolGillespie, DNAGillespie


def routine_gille_pool():
    """
    Main function for Gillespie algorithm w/out the notion of space
    :return: None
    """
    rules = [
        Rule(
            reactants=[Protein.POL2, Protein.RAD26],
            products=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            c=.5
        ),
        Rule(
            reactants=['_'.join(sorted([Protein.POL2, Protein.RAD26]))],
            products=[Protein.POL2, Protein.RAD26],
            c=.4
        )
    ]

    concentrations = {
        Protein.RAD3: 10,
        Protein.POL2: 10,
        Protein.RAD26: 10,
        Protein.IC_RAD3: 5,
        Protein.IC_POL2: 5,
        Protein.IC_RAD26: 5,
    }

    gille = PoolGillespie(protein_conc=concentrations, rules=rules)
    pol2 = []
    pol2_rad26_complex = []
    for i in range(2000):
        pol2.append(gille.get_state(Protein.POL2))
        pol2_rad26_complex.append(gille.get_state('_'.join(sorted([Protein.POL2, Protein.RAD26]))))
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
    num_prot = 1e5
    num_iter = 10
    radiation_time = 10.
    after_radiation_time = 10.
    random_lesion = False
    plot_single_cell = True
    inv_proteins = [Protein.RAD4, Protein.RAD10, Protein.RAD2, Protein.DNA_POL, Protein.DNA_LIG]
    colors = ['tab:red', 'tab:cyan', 'tab:blue', 'yellow', 'tab:purple']

    gille_proteins = Protein.get_types_gillespie()
    concentrations_pool = {gp: num_prot for gp in gille_proteins}
    gille_proteins_elong = deepcopy(gille_proteins)
    gille_proteins_elong.append(Protein.ACTIVE_POL2)

    rules_pool = PoolNoInteract()
    nouv_rules = NoUVHigh(gille_proteins=gille_proteins)
    damage_response = RepairHigh(gille_proteins=gille_proteins)

    rad3_nouv = []
    pol2_nouv = []
    rad26_nouv = []

    rad3_t0 = []
    pol2_t0 = []
    rad26_t0 = []
    cpd_distribution = []
    for t in range(num_iter):
        print('%s' % t)
        gille_pool = PoolGillespie(protein_conc=concentrations_pool, rules=rules_pool.rules)
        gille_dna = DNAGillespie(
            gille_pool,
            dna_spec=DEFAULT_DNA_SPEC_1DIM.copy(),
            protein_names=gille_proteins_elong,
            rules=nouv_rules.rules,
            elong_speed=nouv_rules.elong_speed
        )
        i = 0
        is_radiated = False
        radiation_t = -1
        while True:
            i += 1
            print('Time point: %.3f min' % gille_dna.t)
            print('\n')
            gille_dna.simulate(max_iter=100)

            if gille_dna.t > radiation_time and not is_radiated:
                if plot_single_cell:
                    gille_dna.plot(proteins=inv_proteins, colors=colors)
                rad3_nouv.append(gille_dna.get_protein_state(Protein.RAD3))
                pol2_nouv.append(
                    gille_dna.get_protein_state(Protein.POL2) + gille_dna.get_protein_state(Protein.ACTIVE_POL2)
                )
                rad26_nouv.append(gille_dna.get_protein_state(Protein.RAD26))
                radiation_t = gille_dna.t
                print('##################### UV RADIATION')
                if random_lesion:
                    cpd_start = np.random.choice(
                        np.arange(DEFAULT_DNA_SPEC_1DIM['transcript'][0], DEFAULT_DNA_SPEC_1DIM['transcript'][1])
                    )
                else:
                    cpd_start = DEFAULT_CPD[0]
                cpd_distribution.append(np.arange(cpd_start, cpd_start + DEFAULT_CPD_LENGTH))
                gille_dna.set_rules(damage_response.rules, damage_response.elong_speed)
                gille_dna.add_lesion(cpd_start, cpd_start + DEFAULT_CPD_LENGTH)
                gille_dna.reaction_prob()
                is_radiated = True

            if radiation_t > 0 and not gille_dna.lesions:
                print(gille_dna.t)
                if plot_single_cell:
                    gille_dna.plot(proteins=inv_proteins, colors=colors)
                rad3_t0.append(gille_dna.get_protein_state(Protein.RAD3))
                pol2_t0.append(
                    gille_dna.get_protein_state(Protein.POL2) + gille_dna.get_protein_state(Protein.ACTIVE_POL2)
                )
                rad26_t0.append(gille_dna.get_protein_state(Protein.RAD26))
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

