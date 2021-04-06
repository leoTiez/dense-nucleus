#!/usr/bin/env python3
import numpy as np

from modules.abstractClasses import *


class Message:
    def __init__(self, target, update, prob):
        self.target = target
        self.update = update

        assert 0 <= prob <= 1
        self.prob = prob

    def __ne__(self, other):
        return '%s:%s\tProbability:%s' % (self.target, self.update, self.prob)


class Condition(AbstractEvent):
    def __init__(self, prot_type, num, is_greater=True):
        self.prot_type = prot_type
        self.num = num
        self.is_greater = is_greater

    def __call__(self, *args, **kwargs):
        proteins = args[0]
        mask = np.asarray([isinstance(x, self.prot_type) for x in proteins])
        if self.is_greater:
            if mask.sum() >= self.num:
                return True
            else:
                return False
        else:
            if mask.sum() <= self.num:
                return True
            else:
                False
