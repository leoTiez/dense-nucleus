#!/usr/bin/env python3
from modules.abstractClasses import *


class Message:
    def __init__(self, target, update, prob):
        # target_not_valid = not isinstance(target, AbstractProtein) and not isinstance(target, AbstractProteinComplex)
        # update_not_valid = not isinstance(update, AbstractProtein) \
        #                    and not isinstance(update, AbstractProteinComplex)\
        #                    and not isinstance(update, AbstractDNASegment)
        #
        # if target_not_valid or update_not_valid:
        #     raise ValueError('Passed message target is not accepted.')

        self.target = target
        self.update = update

        assert 0 <= prob <= 1
        self.prob = prob

    def __ne__(self, other):
        return '%s:%s\tProbability:%s' % (self.target, self.update, self.prob)


class Condition:
    def __init__(self, prot_type, num):
        type_not_valid = not isinstance(prot_type, AbstractProtein)\
                         and not isinstance(prot_type, AbstractProteinComplex)
        if type_not_valid:
            raise ValueError('Passed protein type is not accepted.')
        self.prot_type = prot_type
        self.num = num


class Event:
    def __init__(self, message, start_cond, end_cond):
        self.message = message,
        self.start_cond = start_cond
        self.end_cond = end_cond

