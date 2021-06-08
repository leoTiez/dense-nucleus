#!/usr/bin/env python3
from modules.simulations import PetriDish, Nucleus
from modules.proteins import *


def main():
    # nucleus = Nucleus([500, 200, 500, 200, 500, 200], t=.035, animation=False)
    # for t in range(150):
    #     if t == 100:
    #         print('########################### ADD DAMAGE')
    #         nucleus.radiate()
    #     nucleus.update()
    #     nucleus.display()
    #
    # nucleus.to_gif('animations', 'example')

    petri_dish = PetriDish(
        100,
        [(Protein.RAD3, False), (Protein.POL2, False), ('_'.join(sorted([Protein.POL2, Protein.RAD26])), True)],
        animate=True
    )
    for t in range(80):
        print(t)
        if t == 40:
            print('########################### ADD DAMAGE')
            petri_dish.radiate()
        petri_dish.simulate()
        petri_dish.chip(bins=100, time_step=t)

    petri_dish.to_gif('animations', 'slow_elong')


if __name__ == '__main__':
    main()

