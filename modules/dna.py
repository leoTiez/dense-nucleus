#!/usr/bin/env python3
import numpy as np
from modules.abstractClasses import *
from modules.messengers import Message, Event, Action
from rtree import index


class DNASegment(AbstractDNASegment):
    SEGMENT_UNIT = .02

    def __init__(self, start, stop, event=None, action=None, on_add=None, on_del=None, is_damage=False):
        self.start = start
        self.stop = stop
        self.species = '%s:%s' % (self.start, self.stop)
        if event is not None:
            if not isinstance(event, AbstractEvent):
                raise ValueError('Passed event is not of type Event')
            self.events = [event]
        else:
            self.events = []

        if action is not None:
            if not isinstance(action, AbstractAction):
                raise ValueError('Passed action is not of type Action')
            self.actions = [action]
        else:
            self.actions = []

        self.on_add = on_add
        self.on_del = on_del
        self.is_damage = is_damage
        self.proteins = []

    def _apply_action(self, a, i):
        self.proteins[i].add_message_obj(a.message)
        a.callback(self.proteins[i])

    def get_position(self):
        x = np.arange(self.start, self.stop, DNASegment.SEGMENT_UNIT)
        y = np.repeat(.5, x.size)
        return np.dstack((x, y))[0]

    def add_event(self, event):
        if not isinstance(event, AbstractEvent):
            raise ValueError('Passed event is not of type Event')
        self.events.append(event)

    def add_action(self, action):
        if not isinstance(action, AbstractAction):
            raise ValueError('Passed action is not of type Event')
        self.actions.append(action)

    def add_protein(self, p):
        self.proteins.append(p)
        if self.on_add is not None:
            self._apply_action(self.on_add, -1)

    def del_protein(self, p):
        if self.on_del is not None:
            i = self.proteins.index(p)
            self._apply_action(self.on_del, i)
        self.proteins.remove(p)

    def dissociate(self):
        dissoc_prot = []
        for p in self.proteins:
            x_pos, y_pos = p.get_position()
            if not p.interact(self) or not self.start <= x_pos <= self.stop or not .48 <= y_pos <= .52:
                dissoc_prot.append(p)
                self.del_protein(p)

        return dissoc_prot

    def emit(self):
        messages = []
        delete = set()
        for num, e in enumerate(self.events):
            if e.sc is not None:
                if e.sc(self.proteins):
                    e.sc = None
                    messages.append(e.message)
            else:
                if e.tc is None:
                    messages.append(e.message)
                else:
                    if not e.tc(self.proteins):
                        messages.append(e.message)
                    else:
                        delete.add(num)

        if not messages:
            print('Emit no messages')
        return messages

    def act(self):
        def apply_action():
            for i in range(len(self.proteins)):
                self._apply_action(a, i)

        if not self.proteins:
            return

        for num, a in enumerate(self.actions):
            if a.sc is not None:
                if a.sc(self.proteins):
                    apply_action()
            else:
                if a.tc is None:
                    apply_action()
                else:
                    if not a.tc(self.proteins):
                        apply_action()


class DNA:
    def __init__(self):
        self.dna_segments = {}
        p = index.Property()
        p.dimension = 2
        self.segment_tree = index.Index(interleaved=True, properties=p)
        self.damage_tree = index.Index(interleaved=True, properties=p)
        self.id_map = {}

    def __iter__(self):
        return self.dna_segments.values().__iter__()

    def _get_overlap(self, p):
        pos = p.get_position()[0]
        return self.segment_tree.intersection((pos, .5, pos + DNASegment.SEGMENT_UNIT, .5))

    def _get_damage_overlap(self, p):
        pos = p.get_position()[0]
        return list(self.damage_tree.intersection((pos, .5, pos + DNASegment.SEGMENT_UNIT, .5)))

    def add_event(
            self,
            start,
            stop,
            target=None,
            new_prob=None,
            sc=None,
            tc=None,
            on_add=None,
            on_del=None,
            is_damage=False,
            update_area=None
    ):
        key = '%s:%s' % (start, stop)
        if target is not None and new_prob is not None:
            event = Event(Message(target, key if update_area is None else update_area, new_prob), sc=sc, tc=tc)
        else:
            event = None
        self.add_segment(
            start,
            stop,
            event=event,
            update_area=update_area,
            on_add=on_add,
            on_del=on_del,
            is_damage=is_damage
        )

    def add_action(
            self,
            start,
            stop,
            target=None,
            new_prob=None,
            sc=None,
            tc=None,
            callback=None,
            on_add=None,
            on_del=None,
            is_damage=False,
            update_area=None
    ):
        key = '%s:%s' % (start, stop)
        if target is not None and new_prob is not None:
            action = Action(
                Message(target, key if update_area is None else update_area, new_prob),
                callback=callback,
                sc=sc,
                tc=tc
            )
        else:
            action = None

        self.add_segment(
            start,
            stop,
            action=action,
            update_area=update_area,
            on_add=on_add,
            on_del=on_del,
            is_damage=is_damage
        )

    def add_segment(
            self,
            start,
            stop,
            event=None,
            action=None,
            update_area=None,
            on_add=None,
            on_del=None,
            is_damage=False
    ):
        key = '%s:%s' % (start, stop)
        if key not in self.dna_segments.keys():
            self.segment_tree.insert(len(self.dna_segments.keys()), (start, .49, stop, .51))
            self.id_map[len(self.dna_segments.keys())] = key
            if is_damage:
                self.damage_tree.insert(len(self.dna_segments.keys()), (start, .49, stop, .51))
            if event is not None:
                self.dna_segments[key] = DNASegment(
                    start,
                    stop,
                    event=event,
                    on_add=on_add,
                    on_del=on_del,
                    is_damage=is_damage
                )
            if action is not None:
                self.dna_segments[key] = DNASegment(
                    start,
                    stop,
                    on_add=on_add,
                    on_del=on_del,
                    action=action,
                    is_damage=is_damage
                )
        else:
            if event is not None:
                self.dna_segments[key].add_event(event)
            if action is not None:
                self.dna_segments[key].add_action(action)
            if on_add is not None:
                self.dna_segments[key].on_add = on_add
            if on_del is not None:
                self.dna_segments[key].on_del = on_del

        if update_area is not None:
            values = update_area.split(':')
            update_start = float(values[0])
            update_stop = float(values[1])
            if update_area not in self.dna_segments.keys():
                self.segment_tree.insert(len(self.dna_segments.keys()), (update_start, .49, update_stop, .51))
                self.id_map[len(self.dna_segments.keys())] = update_area
                self.dna_segments[update_area] = DNASegment(update_start, update_stop)

    def get_segments(self):
        return self.dna_segments.values()

    def damage_handling(self, damage_segments, p):
        self.del_protein(p)
        for ds in damage_segments:
            self.dna_segments[self.id_map[ds]].add_protein(p)

    def add_protein(self, p):
        damage_segments = self._get_damage_overlap(p)
        if damage_segments:
            self.damage_handling(damage_segments, p)
        else:
            segments = self._get_overlap(p)
            for seg in segments:
                if p not in self.dna_segments[self.id_map[seg]].proteins:
                    self.dna_segments[self.id_map[seg]].add_protein(p)

    def del_protein(self, p):
        segments = self._get_overlap(p)
        for seg in segments:
            try:
                self.dna_segments[self.id_map[seg]].del_protein(p)
            except:
                pass

    def dissociate(self):
        dissoc_prot = []
        for key in self.dna_segments.keys():
            dissoc_prot.extend(self.dna_segments[key].dissociate())

        dissoc_prot = list(set(dissoc_prot))
        [self.del_protein(p) for p in dissoc_prot]
        return dissoc_prot

    def segment_update(self):
        for key in self.dna_segments.keys():
            proteins = self.dna_segments[key].proteins
            for p in proteins:
                self.add_protein(p)
