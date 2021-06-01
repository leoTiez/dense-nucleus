#!/usr/bin/env python3
from abc import ABC, abstractmethod


class AbstractProtein(ABC):
    pass


class AbstractProteinComplex(ABC):
    pass


class AbstractDNASegment(ABC):
    pass


class AbstractEvent(ABC):
    pass


class AbstractAction(ABC):
    pass


class Gillespie(ABC):
    @abstractmethod
    def h(self, reactants):
        pass

    @abstractmethod
    def reaction_prob(self):
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def plot(self):
        pass

