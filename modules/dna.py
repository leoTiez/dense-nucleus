#!/usr/bin/env python3
import numpy as np
from modules.abstractClasses import *
from modules.messengers import Message, Event, Action
from rtree import index


class DNASegment(AbstractDNASegment):
    SEGMENT_UNIT = .02

    def __init__(self, start, stop, event=None, action=None, on_add=None, on_del=None, is_damage=False):
        """
        DNA segments are parts of the DNA molecule that can emit messages (for example to increase/decrease interaction
        profiles of proteins/complexes to facilitate recruitment) and apply actions to associated proteins (e.g.
        updating their interaction profiles or apply a function to them). This enables the definition of particular
        areas on the DNA that have a particular purpose (for example the core promoter or the transcription starting
        site).
        :param start: Start on the DNA. It is required to be value between 0 and 1 with 0 representing the most left
        position and 1 the most right.
        :type start: float
        :param stop: End on the DNA. It is required to be value between 0 and 1 with 0 representing the most left
        position and 1 the most right. The start must be lower than the stop.
        :type stop: float
        :param event: Default event that is emitted by the segment
        :type event: Event
        :param action: Default action that is applied
        :type action: Action
        :param on_add: List of actions that are applied when proteins newly associate.
        :type on_add: list of Action
        :param on_del: List of actions that are applied when proteins dissociate
        :type on_del: list of Action
        :param is_damage: True if the segment represents DNA damage. False otherwise
        :type is_damage: bool
        """
        if start >= stop:
            raise ValueError('Passed start position is after the stop position.')
        self.start = start
        self.stop = stop
        self.species = '%s:%s' % (self.start, self.stop)

        self.events = {}
        if event is not None:
            if not isinstance(event, AbstractEvent):
                raise ValueError('Passed event is not of type Event')
            self.events[event.message.target] = event

        self.actions = {}
        if action is not None:
            if not isinstance(action, AbstractAction):
                raise ValueError('Passed action is not of type Action')
            self.actions[action.message.target] = action

        self.on_add = on_add
        self.on_del = on_del
        self.is_damage = is_damage
        self.proteins = []

    def _apply_action(self, a, i):
        """
        Apply action to protein. Sends first message and then applies callback defined in action.
        :param a: Action to apply
        :type a: Action
        :param i: Protein index in list self.proteins
        :param i: int
        :return: None.
        """
        if self.proteins[i].species == a.message.target:
            self.proteins[i].add_message_obj(a.message)
            a.callback(self.proteins[i])

    def get_position(self):
        """
        Get positions of DNA segment for easy retrieval of nearby proteins.
        :return: numpy.array with coordinates along which the segment expands
        """
        x = np.arange(self.start, self.stop, DNASegment.SEGMENT_UNIT)
        y = np.repeat(.5, x.size)
        return np.dstack((x, y))[0]

    def add_event(self, event):
        """
        Add new event that is emitted by the segment
        :param event: Event to emit
        :type event: Event
        :return: None
        """
        if not isinstance(event, AbstractEvent):
            raise ValueError('Passed event is not of type Event')
        self.events[event.message.target] = event

    def add_action(self, action):
        """
        Add new action that is applied to the associated proteins
        :param action: Action to apply
        :type action: Action
        :return: None
        """
        if not isinstance(action, AbstractAction):
            raise ValueError('Passed action is not of type Event')
        self.actions[action.message.target] = action

    def add_protein(self, p):
        """
        Add new protein to list that is now associated to the DNA segment. Applies all actions from the on_add list
        :param p: Protein to be associated
        :type p: Protein
        :return: None
        """
        self.proteins.append(p)
        p.is_associated = True
        if self.on_add is not None:
            [self._apply_action(on_add, -1) for on_add in self.on_add]

    def del_protein(self, p):
        """
        Removes dissociating proteins from list. Applies all actions from the on_del list
        :param p: Protein to be removed
        :type p: Protein
        :return: None
        """
        if p not in self.proteins:
            return
        if self.on_del is not None:
            i = self.proteins.index(p)
            [self._apply_action(on_del, i) for on_del in self.on_del]
        self.proteins.remove(p)

    def dissociate(self, p):
        """
        Dissociates protein if there is no stable interaction anymore or it has left the physical interaction range
        with the DNA segment
        :param p: Protein which is to be checked whether it is remaining at the DNA segment
        :type p: Protein
        :return: True if dissociated, False otherwise
        """
        x_pos, y_pos = p.get_position()
        if not p.interact(self) or not self.start <= x_pos <= self.stop or not .48 <= y_pos <= .52:
            self.del_protein(p)
            return True
        return False

    def emit(self):
        """
        Send out messages if starting condition is met but the termination condition is not
        :return: List of messages
        """
        messages = []
        for num, e in enumerate(self.events.values()):
            if e.tc is None or not e.tc(self.proteins):
                if e.sc is not None:
                    if e.sc(self.proteins):
                        messages.append(e.message)
                else:
                    messages.append(e.message)

        if not messages:
            pass
            # print('Emit no messages')
        return messages

    def act(self):
        """
        Apply action to associated proteins
        :return: None
        """
        def apply_action():
            """
            Apply current action to all proteins in the list
            :return: None
            """
            for i in range(len(self.proteins)):
                self._apply_action(a, i)

        if not self.proteins:
            return

        for key, a in self.actions.items():
            if a.sc is not None:
                if a.sc(self.proteins):
                    apply_action()
            else:
                if a.tc is None:
                    apply_action()
                else:
                    if not a.tc(self.proteins):
                        apply_action()

    def associated_proteins(self, prot_type, is_complex=False):
        positions = []
        for p in self.proteins:
            if is_complex:
                if p.species == prot_type:
                    positions.append(p.get_position()[0])
            else:
                if isinstance(p, AbstractProtein):
                    if p.species == prot_type:
                        positions.append(p.get_position()[0])
                elif isinstance(p, AbstractProteinComplex):
                    species = [cp.species for cp in p.prot_list]
                    if prot_type in species:
                        positions.append(p.get_position()[0])
                else:
                    raise ValueError('Associated protein is neither of type Protein nor of type ProteinComplex')

        return positions


class DNA:
    def __init__(self):
        """
        DNA molecule. Keeps track of all DNA segments and assigns/removes proteins to/from the segment they belong to.
        The DNA has one initial default segment which consists of the whole strand.
        """
        self.dna_segments = {}
        p = index.Property()
        p.dimension = 2
        self.id_map = {}
        self.id_damage_map = {}
        # Init whole dna
        self.add_empty_segments(.0, 1.)

    def __iter__(self):
        """
        Iteration over all segments in the DNA
        :return: DNA segment iterator
        """
        return self.dna_segments.values().__iter__()

    @staticmethod
    def _is_in_area(area_key, pos):
        boundaries = area_key.split(':')
        return float(boundaries[0]) <= pos <= float(boundaries[1])

    def _get_overlap(self, p):
        """
        Get all segments with which the protein overlaps
        :param p: Protein
        :type p: Protein
        :return: List with segment ids
        """
        pos = p.get_position()[0]
        return [i for i in self.id_map.keys() if self._is_in_area(self.id_map[i], pos)]

    def _get_damage_overlap(self, p):
        """
        Get all damage with which the protein overlaps
        :param p: Protein
        :type p: Protein
        :return: List with damage segment ids
        """
        pos = p.get_position()[0]
        return [i for i in self.id_damage_map.keys() if self._is_in_area(self.id_damage_map[i], pos)]

    def _damage_handling(self, damage_segments, p):
        """
        Damage handling when protein is associated to damage segments. Delete protein from existing segments and add
        protein to damage segments.
        :param damage_segments: List with damage segment ids the protein is now associated to
        :type damage_segments: list int
        :param p: Protein
        :type p: Protein
        :return: None
        """
        self.del_protein(p)
        for ds in damage_segments:
            self.dna_segments[self.id_map[ds]].add_protein(p)

    def add_empty_segments(self, start, stop):
        """
        Add empty segment to DNA. It is recommended to use add_event or add_action instead to make sure that a segment
        serves a purpose. However, this function can be useful to govern general behaviour
        :param start: Starting position
        :type start: float
        :param stop: Ending positon
        :type stop: float
        :return: None
        """
        self.id_map[len(self.dna_segments.keys())] = '%s:%s' % (start, stop)
        self.dna_segments['%s:%s' % (start, stop)] = DNASegment(start, stop)

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
        """
        Add new event emitter to DNA. If the DNA segment is already present, the event is added to the segment. If
        DNA segment does not exist, a new segment is created.
        :param start: Starting position
        :type start: float
        :param stop: Ending postion
        :type stop: float
        :param target: Target (protein or complex) which is supposed to change its interaction profile through emitted
        message
        :type target: str
        :param new_prob: New interaction probability
        :type new_prob: float
        :param sc: Starting condition when the event is started to be emitted
        :type sc: Condition
        :param tc: Termination condition when the event is not emitted anymore
        :type tc: Condition
        :param on_add: List of actions that are applied when proteins newly associate.
        :type on_add: list of Action
        :param on_del: List of actions that are applied when proteins dissociate
        :type on_del: list of Action
        :param is_damage: True if segment is damage. This is only important if the segment is created newly. But it is
        recommended to be passed also if segment exists.
        :type is_damage: bool
        :param update_area: Events can be emitted to update interaction profile for another DNA segment. None if
        this feature is not used.
        :type update_area: str
        :return: None
        """
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
        """
        Add new action to DNA. If the DNA segment is already present, the action is added to the segment. If
        DNA segment does not exist, a new segment is created.
        :param start: Starting position
        :type start: float
        :param stop: Ending postion
        :type stop: float
        :param target: Target (protein or complex) which is supposed to change its interaction profile through emitted
        message
        :type target: str
        :param new_prob: New interaction probability
        :type new_prob: float
        :param sc: Starting condition when the event is started to be emitted
        :type sc: Condition
        :param tc: Termination condition when the event is not emitted anymore
        :type tc: Condition
        :param callback: Callback function that is applied to proteins when applying function.
        :type callback: function
        :param on_add: List of actions that are applied when proteins newly associate.
        :type on_add: list of Action
        :param on_del: List of actions that are applied when proteins dissociate
        :type on_del: list of Action
        :param is_damage: True if segment is damage. This is only important if the segment is created newly. But it is
        recommended to be passed also if segment exists.
        :type is_damage: bool
        :param update_area: Events can be emitted to update interaction profile for another DNA segment. None if
        this feature is not used.
        :type update_area: str
        :return: None
        """
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
        """
        Add a new segment to the DNA. It is recommended to use add_action or add_event instead. However, if event/
        action is already creatde, it can be passed right away through this function.
        :param start: Starting position
        :type start: float
        :param stop: Ending postion
        :type stop: float
        :param event: Event to emit
        :type event: Event
        :param action: Action to emit
        :type action: Action
        :param update_area: Events can be emitted to update interaction profile for another DNA segment. None if
        this feature is not used.
        :type update_area: str
        :param on_add: List of actions that are applied when proteins newly associate.
        :type on_add: list of Action
        :param on_del: List of actions that are applied when proteins dissociate
        :type on_del: list of Action
        :param is_damage: True if segment is damage. This is only important if the segment is created newly. But it is
        recommended to be passed also if segment exists.
        :type is_damage: bool
        :return: None
        """
        start = float(start)
        stop = float(stop)
        key = '%s:%s' % (start, stop)
        if key not in self.dna_segments.keys():
            self.id_map[len(self.dna_segments.keys())] = key
            if is_damage:
                self.id_damage_map[len(self.dna_segments.keys())] = key
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

        if update_area is not None and action is None:
            values = update_area.split(':')
            update_start = float(values[0])
            update_stop = float(values[1])
            if update_area not in self.dna_segments.keys():
                self.add_empty_segments(update_start, update_stop)

    def get_segments(self):
        """
        Getter for all segments
        :return: Created segments
        """
        return self.dna_segments.values()

    def add_protein(self, p):
        """
        Add protein to correct segments
        :param p: Protein
        :type p: Protein
        :return: None
        """
        damage_segments = self._get_damage_overlap(p)
        if damage_segments:
            self._damage_handling(damage_segments, p)
        else:
            segments = self._get_overlap(p)
            for seg in segments:
                if p not in self.dna_segments[self.id_map[seg]].proteins:
                    self.dna_segments[self.id_map[seg]].add_protein(p)

    def del_protein(self, p):
        """
        Delete protein from DNA segments
        :param p: Protein
        :type p: Protein
        :return: None
        """
        for key in self.dna_segments.keys():
            try:
                self.dna_segments[key].del_protein(p)
            except KeyError:
                pass

    def dissociate(self):
        """
        Dissociate proteins from segments if they do not form stable connections anymore
        :return: List with proteins that dissociated from the DNA
        """
        dissoc_prot = set()
        remain_prot = set()
        for key in self.dna_segments.keys():
            proteins = list(reversed(self.dna_segments[key].proteins))
            for p in proteins:
                if p in dissoc_prot or p in remain_prot:
                    continue
                segments = self._get_overlap(p)
                p_dissoc = [self.dna_segments[self.id_map[seg]].dissociate(p) for seg in segments]
                if all(p_dissoc):
                    dissoc_prot.add(p)
                    p.is_associated = False
                else:
                    remain_prot.add(p)

        [self.del_protein(p) for p in list(dissoc_prot)]
        return list(dissoc_prot)

    def segment_update(self):
        """
        Update proteins and to which segments they are associated
        :return: None
        """
        for key in self.dna_segments.keys():
            proteins = self.dna_segments[key].proteins
            for p in proteins:
                self.add_protein(p)
