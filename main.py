#!/usr/bin/env python3
import os
import multiprocessing
from itertools import combinations

import networkx as nx
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

from modules.proteins import *
from modules.dna import *
from modules.messengers import *


class Nucleus:
    INTERACT_RAD = .015

    def __init__(self, num_proteins, pos_dim=2, t=.2, animation=False):
        plt.ion()
        prot_types = Protein.get_types()
        if not len(prot_types) == len(num_proteins):
            raise ValueError('Pass one value for each protein type to determine how many should be created.')
        self.proteins = [ProteinFactory.create(t_prot, pos_dim)
                         for t_prot, n in zip(prot_types, num_proteins) for _ in range(int(n))]

        self.pos_dim = pos_dim
        if not 0. <= t <= 1.:
            raise ValueError('t must be between 0 and 1 as it represents the magnitude of movement')
        self.t = t
        self.dna = DNA()

        self.core_promoter = (.0, .1)
        self.tss = (.1, .15)
        self.transcript = (.1, .85)

        self._init_core_promoter()
        self._init_tss()
        self._init_transcript()

        self.pos = []
        self.state = None
        self._fetch_pos()
        self.animation = animation
        self.gif = []

    def _init_core_promoter(self):
        """
        Define core promoter
        :return: None
        """
        self.dna.add_event(
            start=self.core_promoter[0],
            stop=self.core_promoter[1],
            target=Protein.RAD3,
            new_prob=.9,
            sc=Condition(Protein.RAD3, 1, is_greater=False),
            tc=Condition(Protein.RAD3, 1, is_greater=True)
        )

        # When Rad3 associated, Pol2 or a Pol2-Rad26 complex can attach to transcription starting site
        self.dna.add_event(
            start=self.core_promoter[0],
            stop=self.core_promoter[1],
            target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
            new_prob=.9,
            sc=Condition(Protein.RAD3, 1, is_greater=True),
            tc=Condition(Protein.RAD3, 1, is_greater=False),
            update_area='%s:%s' % (self.tss[0], self.tss[1])
        )
        self.dna.add_event(
            start=self.core_promoter[0],
            stop=self.core_promoter[1],
            target=Protein.POL2,
            new_prob=.9,
            sc=Condition(Protein.RAD3, 1, is_greater=True),
            tc=Condition(Protein.RAD3, 1, is_greater=False),
            update_area='%s:%s' % (self.tss[0], self.tss[1])
        )

    def _init_tss(self):
        """
        Define transcription starting site
        :return: None
        """

        # When Pol2 is associated, probability increased to stay associated, but interaction probability is
        # set back to 0 when Pol2 dissociates
        self.dna.add_action(
            start=self.tss[0],
            stop=self.tss[1],
            target=Protein.POL2,
            new_prob=.999,
            callback=lambda x: None,
            on_add=[Action(
                Message(
                    target=Protein.POL2,
                    update='%s:%s' % (self.tss[0], self.tss[1]),
                    prob=.999
                ),
                lambda x: None
                )],
            on_del=[Action(
                Message(
                    target=Protein.POL2,
                    update='%s:%s' % (self.tss[0], self.tss[1]),
                    prob=.0
                ),
                lambda x: None
            )]
        )

        # Similar for the Pol2-Rad26 complex but also increase interaction probability for complex
        self.dna.add_action(
            start=self.tss[0],
            stop=self.tss[1],
            target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
            new_prob=.999,
            callback=lambda x: None,
            on_add=[
                Action(
                    Message(
                        target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                        update='%s:%s' % (self.tss[0], self.tss[1]),
                        prob=.999
                    ),
                    lambda x: None
                ),
                Action(
                    Message(Protein.POL2, update=Protein.RAD26, prob=.999),
                    lambda x: None
                ),
            ],
            on_del=[
                Action(
                    Message(
                        target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                        update='%s:%s' % (self.tss[0], self.tss[1]),
                        prob=.0
                    ),
                    lambda x: None
                ),
                Action(
                    Message(Protein.POL2, update=Protein.RAD26, prob=.7),
                    lambda x: None
                ),
            ],
        )

        # self.dna.add_event(
        #     start=self.tss[0],
        #     stop=self.tss[1],
        #     target=Protein.RAD3,
        #     new_prob=.0,
        #     sc=Condition(Protein.POL2, 1, is_greater=True),
        #     tc=Condition(Protein.POL2, 1, is_greater=False),
        #     update_area='%s:%s' % (self.core_promoter[0], self.core_promoter[1])
        # )

        self.dna.add_event(
            start=self.tss[0],
            stop=self.tss[1],
            target=Protein.RAD3,
            new_prob=.0,
            sc=Condition('', 1, is_greater=True),
            tc=Condition('', 1, is_greater=False),
            update_area='%s:%s' % (self.core_promoter[0], self.core_promoter[1])
        )

    def _init_transcript(self):
        """
        Define Transcript
        :return: None
        """
        def pol2_callback(p):
            if isinstance(p, Pol2):
                p.set_position_delta(np.asarray([1.2 * Nucleus.INTERACT_RAD, .0]))

        def complex_callback(p):
            if p.species == '_'.join(sorted([Protein.POL2, Protein.RAD26])):
                p.set_position_delta(np.asarray([1.2 * Nucleus.INTERACT_RAD, .0]))

        # Pol2 and complex is pushed forward along transcript with pol2_callback/complex callback
        # When Pol2/complex associates, the probability of staying on the transcript is increased by is set back to 0
        # in case it dissociates
        self.dna.add_action(
            start=self.transcript[0],
            stop=self.transcript[1],
            target=Protein.POL2,
            new_prob=.999,
            callback=pol2_callback,
            on_add=[Action(
                Message(target=Protein.POL2, update='%s:%s' % (self.transcript[0], self.transcript[1]), prob=.999),
                lambda x: None
            )],
            on_del=[Action(
                Message(target=Protein.POL2, update='%s:%s' % (self.transcript[0], self.transcript[1]), prob=.0),
                lambda x: None
            )],
        )

        self.dna.add_action(
            start=self.transcript[0],
            stop=self.transcript[1],
            target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
            new_prob=.999,
            callback=complex_callback,
            on_add=[
                Action(
                    Message(
                        target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                        update='%s:%s' % (self.transcript[0], self.transcript[1]),
                        prob=.999
                    ),
                    lambda x: None
                ),
                Action(
                    Message(Protein.POL2, update=Protein.RAD26, prob=.999),
                    lambda x: None
                ),
            ],
            on_del=[
                Action(
                    Message(
                        target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                        update='%s:%s' % (self.transcript[0], self.transcript[1]),
                        prob=.0
                    ),
                    lambda x: None
                ),
                Action(
                    Message(Protein.POL2, update=Protein.RAD26, prob=.8),
                    lambda x: None
                ),
            ],
        )
        # When anything associates to the tss (Pol2/complex), Rad3 doesn't need to bind anymore. This is intended to
        # serve as a balance between continuously associating to the core promoter and dissociating.
        self.dna.add_event(
            start=self.transcript[0],
            stop=self.transcript[1],
            target=Protein.RAD3,
            new_prob=.0,
            sc=Condition('', 1, is_greater=True),
            tc=Condition('', 1, is_greater=False),
            update_area='%s:%s' % (self.core_promoter[0], self.core_promoter[1])
        )

    def _fetch_pos(self):
        with multiprocessing.Pool(np.maximum(multiprocessing.cpu_count() - 1, 1)) as parallel:
            results = []
            for p in self.proteins:
                res = parallel.apply_async(p.get_position, args=())
                results.append(res)
            parallel.close()
            parallel.join()
            self.pos = [r.get() for r in results]
        self.state = KDTree(np.asarray(self.pos))

    def global_event(self, target, update, prob):
        for i in range(len(self.proteins)):
            self.proteins[i].clear_messages()
            if self.proteins[i].species == target:
                self.proteins[i].add_message(target, update, prob)

    def global_event_obj(self, m):
        self.global_event(m.target, m.update, m.prob)

    def radiate(self, damage_site_x=None):
        """
        Radiate cell
        :param damage_site_x: Optional add start position of damage
        :return: None
        """
        if damage_site_x is None:
            damage_site_x = np.random.uniform(self.transcript[0], self.transcript[1] - .1)
        damage_site = (damage_site_x, damage_site_x + .1)

        # Stalling of Pol2/Complex
        self.dna.add_action(
            start=damage_site[0],
            stop=damage_site[1],
            target=Protein.POL2,
            new_prob=1.,
            callback=lambda x: None,
            on_add=[Action(
                Message(target=Protein.POL2, update='%s:%s' % (damage_site[0], damage_site[1]), prob=1.),
                lambda x: None
            )],
            on_del=[Action(
                Message(target=Protein.POL2, update='%s:%s' % (damage_site[0], damage_site[1]), prob=.0),
                lambda x: None
            )],
            is_damage=True
        )

        self.dna.add_action(
            start=damage_site[0],
            stop=damage_site[1],
            target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
            new_prob=1.,
            callback=lambda x: None,
            on_add=[Action(
                Message(
                    target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                    update='%s:%s' % (damage_site[0], damage_site[1]),
                    prob=1.
                ),
                lambda x: None
            )],
            on_del=[Action(
                Message(
                    target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                    update='%s:%s' % (damage_site[0], damage_site[1]),
                    prob=.0
                ),
                lambda x: None
            )],
            is_damage=True
        )

        # Shutdown of transcription
        self.dna.add_action(
            start=self.tss[0],
            stop=self.tss[1],
            target=Protein.POL2,
            new_prob=.0,
            callback=lambda x: None,
            on_add=[
                Action(
                    Message(target=Protein.POL2, update='%s:%s' % (self.tss[0], self.tss[1]), prob=.0),
                    lambda x: None
                ),
                Action(
                    Message(target=Protein.POL2, update='%s:%s' % (self.transcript[0], self.transcript[1]), prob=.0),
                    lambda x: None
                ),
            ],
        )

        self.dna.add_action(
            start=self.tss[0],
            stop=self.tss[1],
            target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
            new_prob=.0,
            callback=lambda x: None,
            on_add=[
                Action(
                    Message(
                        target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                        update='%s:%s' % (self.tss[0], self.tss[1]),
                        prob=.0
                    ),
                    lambda x: None
                ),
                Action(
                    Message(
                        target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
                        update='%s:%s' % (self.transcript[0], self.transcript[1]),
                        prob=.0
                    ),
                    lambda x: None
                ),
            ],
        )

        # Remove Rad3 from core promoter
        self.dna.add_action(
            start=self.core_promoter[0],
            stop=self.core_promoter[1],
            target=Protein.RAD3,
            new_prob=.0,
            callback=lambda x: None,
            on_add=[Action(
                Message(target=Protein.RAD3, update='%s:%s' % (self.core_promoter[0], self.core_promoter[1]), prob=.0),
                lambda x: None
            )],
        )

        # Recruitment of Rad26 if not present
        self.dna.add_event(
            start=damage_site[0],
            stop=damage_site[1],
            target=Protein.RAD26,
            new_prob=.9,
            sc=Condition(Protein.POL2, 1, is_greater=True),
            tc=Condition(Protein.RAD26, 1, is_greater=True),
            is_damage=True
        )

        # Recruitment of Rad3 to lesion when Rad26 is present
        self.dna.add_event(
            start=damage_site[0],
            stop=damage_site[1],
            target=Protein.RAD3,
            new_prob=.9,
            sc=Condition(Protein.RAD26, 1, is_greater=True),
            tc=Condition(Protein.RAD3, 2, is_greater=True),
            is_damage=True
        )

        # Recruit no new Rad3 to core promoter
        self.dna.add_event(
            start=self.core_promoter[0],
            stop=self.core_promoter[1],
            target=Protein.RAD3,
            new_prob=.0
        )

        # Global reset
        self.global_event(
            target='_'.join(sorted([Protein.POL2, Protein.RAD26])),
            update='%s:%s' % (self.tss[0], self.tss[1]),
            prob=.0
        )
        self.global_event(
            target=Protein.POL2,
            update='%s:%s' % (self.tss[0], self.tss[1]),
            prob=.0
        )
        self.global_event(
            target=Protein.RAD3,
            update='%s:%s' % (self.core_promoter[0], self.core_promoter[1]),
            prob=.0
        )

    def update(self):
        def handshake(x, y, k):
            m = self.proteins[x].broadcast(k)
            self.proteins[y].add_message_obj(m)

        # Release all unstable connections that aren't associated to the dna
        collapse_mask = np.asarray([x.is_collapsing() and not x.is_associated for x in self.proteins])
        idx = np.arange(len(self.proteins))
        rev_idx = np.flip(idx[collapse_mask])
        for i in rev_idx:
            self.proteins.extend(self.proteins[i].prot_list)
            del self.proteins[i]

        # Dissociate unstable connections
        self.dna.dissociate()

        # Fetch positions
        self._fetch_pos()

        # Update segments
        self.dna.segment_update()

        # DNA:Protein interaction
        for num, seg in enumerate(self.dna):
            seg.act()
            mes = seg.emit()
            neighbor_prots = self.state.query_radius(seg.get_position(), r=Nucleus.INTERACT_RAD)
            neighbor_prots = np.unique([x for n in neighbor_prots for x in n])
            neighbor_prots = sorted(neighbor_prots, reverse=True)
            for i in neighbor_prots:
                if mes and isinstance(self.proteins[i], InfoProtein):
                    [self.proteins[i].add_message_obj(m) for m in mes]
                if self.proteins[i].interact(seg):
                    self.dna.add_protein(self.proteins[i])

        asso_mask = np.asarray([x.is_associated for x in self.proteins])
        idx = np.arange(len(self.proteins))
        success_asso = idx[asso_mask].tolist()

        # Protein:Protein interactions
        adj = np.zeros((len(self.proteins), len(self.proteins)))
        idc = self.state.query_radius(self.pos, r=Nucleus.INTERACT_RAD)
        for num, i in enumerate(idc):
            adj[num, i] = 1
        adj = np.multiply(adj, adj.T)
        interact_graph = nx.from_numpy_matrix(adj)
        pot_inter = sorted(nx.connected_components(interact_graph), reverse=True, key=len)

        complexes = []
        success_inter = set()
        for inter_group in pot_inter:
            if len(inter_group) == 1:
                continue
            interactions = list(combinations(list(inter_group), 2))
            success_deliver = []
            for i, j in interactions:
                if i in success_asso or j in success_asso:
                    continue
                if self.proteins[i].share_info(self.proteins[j]) and i not in success_deliver:
                    success_deliver.append(i)
                    keys_i = list(self.proteins[i].get_message_keys())
                    for k_i in keys_i:
                        handshake(i, j, k_i)
                if self.proteins[j].share_info(self.proteins[i]) and j not in success_deliver:
                    success_deliver.append(i)
                    keys_j = list(self.proteins[j].get_message_keys())
                    for k_j in keys_j:
                        handshake(j, i, k_j)

            for i, j in interactions:
                if i in success_asso or j in success_asso:
                    continue
                if self.proteins[i].interact(self.proteins[j]) or self.proteins[j].interact(self.proteins[i]):
                    if i in success_inter or j in success_inter:
                        continue
                    complexes.append(ProteinComplex([self.proteins[i], self.proteins[j]]))
                    success_inter.add(i)
                    success_inter.add(j)

        self.proteins.extend(complexes)
        still = list(set(success_asso).union(success_inter))
        [self.proteins[i].update_position(self.t) for i in range(len(self.proteins)) if i not in still]
        for si in sorted(list(success_inter), reverse=True):
            del self.proteins[si]

    def display(self):
        results = []
        with multiprocessing.Pool(np.maximum(multiprocessing.cpu_count() - 1, 1)) as parallel:
            for p in self.proteins:
                res = parallel.apply_async(p.get_features, args=())
                results.append(res)

            parallel.close()
            parallel.join()
            results = [r.get() for r in results]

        recruited = [p.get_features() for seg in self.dna for p in seg.proteins]
        if not recruited:
            x_recruited, y_recruited, c_recruited = [], [], []
        else:
            x_recruited = list(map(lambda x: x[0], recruited))
            x_recruited = [x for x_group in x_recruited for x in x_group]
            y_recruited = list(map(lambda x: x[1], recruited))
            y_recruited = [y for y_group in y_recruited for y in y_group]
            c_recruited = list(map(lambda x: x[2], recruited))
            c_recruited = [c for c_group in c_recruited for c in c_group]

        x_all = list(map(lambda x: x[0], results))
        x_all = [x for x_group in x_all for x in x_group]
        y_all = list(map(lambda x: x[1], results))
        y_all = [y for y_group in y_all for y in y_group]
        c_all = list(map(lambda x: x[2], results))
        c_all = [c for c_group in c_all for c in c_group]
        edge_all = list(map(lambda x: x[3], results))
        edge_all = [edge for edge_group in edge_all for edge in edge_group]

        plt.scatter(x_all, y_all, s=5e3 * Nucleus.INTERACT_RAD, c=c_all, edgecolors=edge_all)
        plt.plot(np.linspace(0, 1, 10), [0.5] * 10, linewidth=7.0, color='black')
        for seg in self.dna:
            x, y = seg.get_position().T
            plt.plot(x, y, linewidth=5.0)
        plt.scatter(
            x_recruited,
            y_recruited,
            s=5e3 * Nucleus.INTERACT_RAD,
            c=c_recruited,
            edgecolor='red',
            hatch=r'//',
            zorder=5
        )
        figure = plt.gcf()
        figure.canvas.flush_events()
        figure.canvas.draw()
        if self.animation:
            image = np.frombuffer(figure.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
            self.gif.append(image)
        plt.cla()

    def to_gif(self, path, save_prefix):
        curr_dir = os.getcwd()
        Path('%s/%s' % (curr_dir, path)).mkdir(exist_ok=True, parents=True)
        imageio.mimsave("%s/%s_nucleus_ani.gif" % (path, save_prefix), self.gif, fps=5)


def main():
    nucleus = Nucleus([500, 200, 500, 200, 500, 200], t=.035, animation=True)
    for t in range(150):
        if t == 100:
            print('########################### ADD DAMAGE')
            nucleus.radiate()
        nucleus.update()
        nucleus.display()

    nucleus.to_gif('animations', 'example')


if __name__ == '__main__':
    main()

