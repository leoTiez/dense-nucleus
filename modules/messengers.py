#!/usr/bin/env python3
import numpy as np

from modules.abstractClasses import *


class Message:
    def __init__(self, target, update, prob):
        """
        Message class used for updating interaction probabilities. Every message consists three parts. The target
        defines the protein that updates their interaction profiles. The update represents the DNA/protein/protein
        complex with which the target protein has now a different interaction probability. And the prob is the new
        interaction probability which can be any value between (and including) 0 and 1.
        :param target: target protein which updates the interaction profile
        :type target: str
        :param update: DNA/protein/protein complex with which the target protein changes interaction probability
        :param prob: New interaction probability
        """
        self.target = target
        self.update = update

        assert 0 <= prob <= 1
        self.prob = prob

    def __str__(self):
        """
        To string method
        :return: string with target, update and probability
        """
        return 'Target:%s\tUpdate:%s\tProbability:%s' % (self.target, self.update, self.prob)


class Condition(AbstractEvent):
    def __init__(self, prot_type, num, is_greater=True):
        """
        A general description of a condition that must be met before starting with emitting a message or applying
        an action (see DNA and DNASegment class).
        :param prot_type: Protein type for which the condition must be met. Can be a protein, protein complex or empty
        string (if condition is not protein specific).
        :type prot_type: str
        :param num: Threshold number of proteins
        :type num: int
        :param is_greater: If true, the number of proteins on the DNA segment (see DNASegment class) must be equal or
        larger than num (the threshold). Otherwise condition is fulfilled if less proteins than num are associated.
        :type is_greater: bool
        """
        self.prot_type = prot_type
        self.num = num
        self.is_greater = is_greater

    def __call__(self, *args, **kwargs):
        """
        When condition is called, it expects a list of proteins to be passed as args. It returns True if condition
        is met, and False otherwise
        :param args: args[0] is assumed to be a list of proteins
        :param kwargs: None
        :return: True if condition is met, False otherwise.
        """
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
        """
        Events are messages that can be emitted by a DNASegment based on starting and terminating conditions
        :param message: The message that is to be emitted
        :type message: Message
        :param sc: Starting condition. When fulfilled, message can be emitted
        :type sc: Condition
        :param tc: Termination condition. When fulfilled, message is not emitted anymore
        :type tc: Condition
        """
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
        """
        Actions are messages and functions that are applied to proteins that are associated to a DNASegment.
        :param message: Message that is sent to proteins associated to the DNASegment. They can be used to increase or
        decrease interaction probability with other DNASegments and/or proteins
        :type message: Message
        :param callback: Function that is applied to the proteins associated to the DNASegment. This can be used to
        achieve a certain protein dynamic. For example, Pol2 is pushed forward.
        :type callback: function
        :param sc: Starting condition. When fulfilled, message can be emitted
        :type sc: Condition
        :param tc: Termination condition. When fulfilled, message is not emitted anymore
        :type tc: Condition
        """
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

