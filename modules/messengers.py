#!/usr/bin/env python3
import numpy as np

from modules.abstractClasses import *


class Message:
    def __init__(self, target, update, prob):
        self.target = target
        self.update = update

        assert 0 <= prob <= 1
        self.prob = prob

    def __str__(self):
        return '%s:%s\tProbability:%s' % (self.target, self.update, self.prob)


class Condition(AbstractEvent):
    def __init__(self, prot_type, num, is_greater=True):
        self.prot_type = prot_type
        self.num = num
        self.is_greater = is_greater

    def __call__(self, *args, **kwargs):
        proteins = args[0]
        mask = []
        for x in proteins:
            if isinstance(x, AbstractProteinComplex):
                mask.extend([p.species == self.prot_type or self.prot_type == '' for p in x.prot_list])
            else:
                mask.append(x.species == self.prot_type or self.prot_type == '')
        mask = np.asarray(mask)
        if self.is_greater:
            if mask.sum() >= self.num:
                return True
            else:
                return False
        else:
            if mask.sum() < self.num:
                return True
            else:
                return False


class Event(AbstractEvent):
    def __init__(self, message, sc=None, tc=None):
        if not isinstance(message, Message):
            raise ValueError('Passed message is not of type Message')
        if sc is not None:
            if not isinstance(sc, Condition):
                raise ValueError('Passed starting condition is not of type Condition')
        if tc is not None:
            if not isinstance(tc, Condition):
                raise ValueError('Passed termination condition is not of type Condition')
        self.message = message
        self.sc = sc
        self.tc = tc


class Action(AbstractAction):
    def __init__(self, message, callback, sc=None, tc=None):
        if not isinstance(message, Message):
            raise ValueError('Passed message is not of type Message')
        if sc is not None:
            if not isinstance(sc, Condition):
                raise ValueError('Passed starting condition is not of type Condition')
        if tc is not None:
            if not isinstance(tc, Condition):
                raise ValueError('Passed termination condition is not of type Condition')
        self.callback = callback
        self.message = message
        self.sc = sc
        self.tc = tc

