#!/usr/bin/env python3
import numpy as np
from modules.abstractClasses import *


class DNASegment(AbstractDNASegment):
    SEGMENT_UNIT = .01

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.species = '%s:%s' % (self.start, self.stop)
        self.proteins = []
        self.messages = {}
        self.cond = {}

    def get_position(self):
        x = np.arange(self.start, self.stop, DNASegment.SEGMENT_UNIT)
        y = np.repeat(.5, x.size)
        return np.dstack((x, y))

    def add_protein(self, p):
        self.proteins.append(p)

    def del_protein(self, p):
        self.proteins.remove(p)

    def add_event(self, target, number):
        pass

