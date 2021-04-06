#!/usr/bin/env python3
import numpy as np
from modules.abstractClasses import *
from modules.messengers import Message


class DNASegment(AbstractDNASegment):
    SEGMENT_UNIT = .01

    def __init__(self, start, stop, message, sc=None, tc=None):
        self.start = start
        self.stop = stop
        self.species = '%s:%s' % (self.start, self.stop)
        self.message = message
        if sc is not None:
            if not isinstance(sc, AbstractEvent):
                raise ValueError('Start condition is not None and not an event.')
        if tc is not None:
            if not isinstance(tc, AbstractEvent):
                raise ValueError('Termination condition is not None and not an event.')
        self.sc = sc
        self.tc = tc
        self.proteins = []

    def get_position(self):
        x = np.arange(self.start, self.stop, DNASegment.SEGMENT_UNIT)
        y = np.repeat(.5, x.size)
        return np.dstack((x, y))[0]

    def add_protein(self, p):
        self.proteins.append(p)

    def del_protein(self, p):
        self.proteins.remove(p)

    def dissociate(self):
        dissoc_prot = []
        for p in self.proteins:
            if not p.interact(self):
                dissoc_prot.append(p)
                self.del_protein(p)

        return dissoc_prot

    def emit(self):
        if self.sc is not None:
            if self.sc(self.proteins):
                self.sc = None
                return self.message
        else:
            if self.tc is None:
                return self.message
            else:
                if not self.tc(self.proteins):
                    return self.message


class DNA:
    def __init__(self):
        self.dna_segments = []

    def __iter__(self):
        return self.dna_segments.__iter__()

    def add_segment(self, start, stop, target, new_prob, sc=None, tc=None):
        message = Message(target, '%s:%s' % (start, stop), new_prob)
        self.dna_segments.append(DNASegment(start, stop, message, sc=sc, tc=tc))

