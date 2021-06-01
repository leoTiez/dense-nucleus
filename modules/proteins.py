#!/usr/bin/env python3
from abc import ABC
from itertools import combinations
import numpy as np

from modules.abstractClasses import *
from modules.messengers import *


class Protein(AbstractProtein, ABC):
    RAD3 = 'rad3'
    POL2 = 'pol2'
    RAD26 = 'rad26'
    RAD4 = 'rad4'
    RAD10 = 'rad10'
    RAD2 = 'rad2'
    DNA_POL = 'poly'
    DNA_LIG = 'cdc9'

    ACTIVE_POL2 = 'active pol2'

    IC_RAD3 = 'ic rad3'
    IC_POL2 = 'ic pol2'
    IC_RAD26 = 'ic rad26'

    COLORS = {
        RAD3: 'orange',
        IC_RAD3: 'grey',  # 'lightsalmon'
        POL2: 'green',
        IC_POL2: 'grey',  # 'springgreen'
        RAD26: 'cyan',
        IC_RAD26: 'grey',  # 'lightblue'
    }

    def __init__(self, species, p_inter, p_info, color, pos_dim=2):
        """
        Abstract Protein class.
        :param species: Species name. It is assumed that the species is one of the predefined protein types (see
        get_types), although this is no requirement
        :type species: str
        :param p_inter: Dictionary with the interaction probabilities. Keys are the proteins/complexes/DNA segments
        that protein can interact with
        :type p_inter: dict
        :param p_info: Dictionary with probabilities for sharing information. This is separated from p_inter to enable
        signalling cascades: information can be shared, although no stable physical complex is created
        :type p_info: dict
        :param color: The colour in which the protein is displayed during the simulation
        :type color: str
        :param pos_dim: Number of spatial dimensions that are used to determine the position.
        :type pos_dim: int
        """
        self.color = color
        self.species = species
        self.position = np.random.random(pos_dim)
        self.p_inter = p_inter
        self.p_info = p_info
        self.is_associated = False
        self.messages = {}

    def _tossing(self, p, prob_type='inter'):
        """
        Simulated coin tossing. Determines whether protein:protein or protein:DNA interaction happens or information
        is shared.
        :param p: Protein for which interaction is determined
        :type p: Protein, ProteinComplex, DNASegment
        :param prob_type: The type of probability that is used for determining interaction/communication. If
        prob_type is inter, the interaction probability is chosen (physical interaction). If info is passed, the
        information probability is selected (signalling cascade)
        :return: True if interaction happens. False otherwise.
        """
        species = p.species
        toss = np.random.random()
        if prob_type == 'inter':
            if species in self.p_inter.keys():
                return self.p_inter[species] >= toss
            else:
                return False
        elif prob_type == 'info':
            if species in self.p_inter.keys():
                return self.p_info[species] >= toss
            else:
                return False
        else:
            raise ValueError('Probability type %s is not understood' % prob_type)

    def _update_p_inter(self, species, p_update):
        """
        Update interaction probability. Information probability cannot be updated.
        :param species: Species with which the interaction probability is changed
        :type species: str
        :param p_update: New probability
        :type p_update: float
        :return: None
        """
        # print('UPDATE INTERACTION PROBABILITY %s, %s, %s.' % (self.species, species, p_update))
        self.p_inter[species] = p_update

    def set_position_delta(self, delta):
        """
        Position update by delta.
        :param delta: Position update
        :type delta: numpy.array
        :return: None
        """
        self.position += delta
        self.position = np.maximum(0, self.position)
        self.position = np.minimum(1, self.position)

    def get_position(self):
        """
        Getter for protein position
        :return: Protein position
        """
        return self.position

    def update_position(self, mag):
        """
        Update position as random movement.
        :param mag: Magnitude. Defines the maximum difference between old position and new position per dimension
        :type mag: float
        :return: None
        """
        self.position += mag * np.random.uniform(-1, 1, len(self.position))
        self.position = np.maximum(0, self.position)
        self.position = np.minimum(1, self.position)

    def interact(self, protein):
        """
        Determine whether to interact with other proteins.
        :param protein: Protein to interact with
        :type protein: Protein
        :return: True if protein interact, False otherwise
        """
        if isinstance(protein, Protein):
            return self._tossing(protein, prob_type='inter')
        elif isinstance(protein, ProteinComplex):
            return all(map(lambda x: self._tossing(x, prob_type='inter'), protein.prot_list))
        elif isinstance(protein, AbstractDNASegment):
            return self._tossing(protein, prob_type='inter')
        else:
            raise ValueError('Passed protein is not of class Protein or Protein Complex')

    def share_info(self, protein):
        """
        Determine whether proteins share information
        :param protein: Protein to send information to
        :type protein: Protein
        :return: True if information is shared, False otherwise
        """
        if isinstance(protein, Protein):
            return self._tossing(protein, prob_type='info')
        elif isinstance(protein, ProteinComplex):
            return all(map(lambda x: self._tossing(x, prob_type='info'), protein.prot_list))
        else:
            raise ValueError('Passed protein is not of class Protein or Protein Complex')

    def add_message(self, target, update, prob):
        """
        Add new message. If protein is target or update, it updates its own interaction profile. This is intended
        to keep interaction fairly symmetrical. If protein is neither target nor update, message is added to existing
        messages.
        :param target: Protein which is supposed to update is interaction profile
        :type target: str
        :param update: Protein/Complex/DNASegment for which the interaction profile changes
        :type update: str
        :param prob: New probability
        :type prob: float
        :return: None
        """
        if self.species == target:
            self._update_p_inter(update, prob)
        elif self.species == update:
            self._update_p_inter(target, prob)
        else:
            self.messages['%s:%s' % (target, update)] = Message(target, update, prob)

    def add_message_obj(self, m):
        """
        Wrapper function for add_message to pass a Message object instead.
        :param m: Message
        :type m: Message
        :return: None
        """
        self.add_message(m.target, m.update, m.prob)

    def del_message(self, key):
        """
        Delete message
        :param key: Key identifying the message. This is a string composed of target:update
        :type key: str
        :return: None
        """
        self.messages.pop(key)

    def clear_messages(self):
        """
        Remove all messages and reset dictionary
        :return: None
        """
        self.messages = {}

    def get_message_keys(self):
        """
        Get all keys for which messages are available
        :return: List of keys
        """
        return self.messages.keys()

    def broadcast(self, key):
        """
        Broadcast a message. Retrieve the message and delete it afterwards. There messages do not exponentially
        replicate. Every message that is delivered to a new protein is deleted.
        :param key: Key identifying the message. This is a string composed of target:update
        :type key: str
        :return: Message
        """
        m = self.messages[key]
        self.del_message(key)
        return m

    @staticmethod
    def is_collapsing():
        """
        Protein complexes can collapse into their respective proteins. Since the protein itself cannot further collapse,
        this function returns always False
        :return: False
        """
        return False

    def get_features(self):
        """
        Getter function for retrieving plotting features
        :return: List with x coordinate, list with y coordinate, list with colour, list with edge colour. Values are
        returned in list to permit equal handling in the nucleus for proteins and protein complexes.
        """
        c = self.color
        x = self.position[0]
        y = self.position[1]
        edge = 'limegreen' if self.messages.keys() else 'white'
        return [x], [y], [c], [edge]

    @staticmethod
    def get_types():
        """
        Get pre-defined protein types
        :return: List with pre-defined protein types
        """
        return [Protein.RAD3, Protein.IC_RAD3, Protein.POL2, Protein.IC_POL2, Protein.RAD26, Protein.IC_RAD26]

    @staticmethod
    def get_types_gillespie():
        """
        Get pre-defined protein types for the Gillespie algorithm
        :return: List with pre-defined protein types
        """
        return [Protein.RAD3, Protein.POL2, Protein.RAD26, Protein.RAD4,
                Protein.RAD10, Protein.RAD2, Protein.DNA_POL, Protein.DNA_LIG]


class ProteinComplex(AbstractProteinComplex):
    def __init__(self, prot_list):
        """
        Proteins can interact to form complexes. The protein complex class bundles these proteins together to act as
        one unit. It is able to use the ProteinComplex like a Protein
        :param prot_list: List with proteins
        :type prot_list: list
        """
        def flatten_list():
            """
            Flattens list if list with lists is passed. This can happen when protein creates complex with another
            complex
            :return: Flattened list
            """
            proteins = []
            for p in prot_list:
                if isinstance(p, ProteinComplex):
                    proteins.extend(p.prot_list)
                elif isinstance(p, Protein):
                    proteins.append(p)
                else:
                    raise ValueError('Not all proteins are valid to form a complex')
            return proteins

        self.prot_list = flatten_list()
        self.position = self.prot_list[0].get_position()
        self.is_associated = all([x.is_associated for x in self.prot_list])
        self.species = '_'.join(sorted([x.species for x in self.prot_list]))
        self.p_complex = {}
        self._init_broadcast()

    def _init_broadcast(self):
        """
        Initial internal broadcasting of messages to make sure that all proteins share the same information. If messages
        are contradictory, a majority vote takes the information most proteins have.
        :return: None
        """
        temp_mes = {}
        for p in self.prot_list:
            for k, m in p.messages.items():
                if k not in temp_mes.keys():
                    temp_mes[k] = []
                temp_mes[k].append(m)

        keys = temp_mes.keys()
        for k in keys:
            if len(temp_mes[k]) == 1:
                temp_mes[k] = temp_mes[k][0]
            else:
                probs = np.asarray(list(map(lambda x: x.prob, temp_mes[k])))
                unique, counts = np.unique(probs, return_counts=True)
                unique, _ = zip(*sorted(zip(unique, counts), reverse=True, key=lambda x: x[1]))
                temp_mes[k] = Message(temp_mes[k][0].target, temp_mes[k][0].update, unique[0])

        [self.add_message_obj(m) for m in temp_mes.values()]

    def _update_p_inter(self, species, p_update):
        """
        Update interaction probability for protein complex. They can be specifically set only for the complex
        :param species: The type of protein/protein complex/DNA segment with which the protein complex has now
        a changed interaction profile
        :type species: str
        :param p_update: New probability
        :type p_update: float
        :return: Npne
        """
        # print('UPDATE INTERACTION PROBABILITY %s, %s, %s.' % (self.species, species, p_update))
        self.p_complex[species] = p_update

    def set_position_delta(self, delta):
        """
        Update the new position about delta
        :param delta: Update to new position
        :type delta: numpy.array
        :return: None
        """
        [x.set_position_delta(delta) for x in self.prot_list]
        self.position = self.prot_list[0].get_position()

    def get_position(self):
        """
        Getter for protein position
        :return: Protein position
        """
        return self.position

    def update_position(self, mag):
        """
        Update position as random movement. The more proteins are in the complex, the less the complex moves.
        :param mag: Magnitude. Defines the maximum difference between old position and new position per dimension
        :type mag: float
        :return: None
        """
        delta = (1. / float(len(self.prot_list))) * mag * np.random.uniform(-1, 1, len(self.position))
        self.set_position_delta(delta)

    def interact(self, protein):
        """
        Determine whether to interact with other proteins.
        :param protein: Protein to interact with
        :type protein: Protein
        :return: True if protein interact, False otherwise
        """
        if protein.species in self.p_complex.keys():
            toss = np.random.random()
            return self.p_complex[protein.species] >= toss
        return all([x.interact(protein) for x in self.prot_list])

    def share_info(self, species):
        """
        Determine whether proteins share information
        :param protein: Protein to send information to
        :type protein: Protein
        :return: True if information is shared, False otherwise
        """
        # TODO ANY OR ALL BETTER FOR SHARING INFO?
        return all([x.share_info(species) for x in self.prot_list])

    def add_message(self, target, update, prob):
        """
        Add new message. If protein is target or update, it updates its own interaction profile. This is intended
        to keep interaction fairly symmetrical. If protein is neither target nor update, message is added to existing
        messages.
        :param target: Protein which is supposed to update is interaction profile
        :type target: str
        :param update: Protein/Complex/DNASegment for which the interaction profile changes
        :type update: str
        :param prob: New probability
        :type prob: float
        :return: None
        """
        if self.species == target:
            self._update_p_inter(update, prob)
        elif self.species == update:
            self._update_p_inter(target, prob)
        else:
            [x.add_message(target, update, prob) for x in self.prot_list]

    def add_message_obj(self, m):
        """
        Wrapper function for add_message to pass a Message object instead.
        :param m: Message
        :type m: Message
        :return: None
        """
        self.add_message(m.target, m.update, m.prob)

    def del_message(self, key):
        """
        Delete message for all proteins
        :param key: Key identifying the message. This is a string composed of target:update
        :type key: str
        :return: None
        """
        try:
            [x.del_message(key) for x in self.prot_list]
        except KeyError:
            # print('Different number of messages in complex')
            pass

    def clear_messages(self):
        """
       Remove all messages and reset dictionary for all proteins.
       :return: None
       """
        [x.clear_messages() for x in self.prot_list]

    def get_message_keys(self):
        """
        Get all keys for which messages are available
        :return: List of keys
        """
        return self.prot_list[0].messages.keys()

    def is_collapsing(self):
        """
        Verify whether proteins still form a stable complex.
        :return: True if complex is collapsing and the proteins do not form a stable complex anymore. False otherwise.
        """
        interactions = combinations(range(len(self.prot_list)), 2)
        results = []
        for i, j in interactions:
            results.append(self.prot_list[i].interact(self.prot_list[j]))
        return not all(results)

    def broadcast(self, key):
        """
        Broadcast a message. Retrieve the message and delete it afterwards. There messages do not exponentially
        replicate. Every message that is delivered to a new protein is deleted.
        :param key: Key identifying the message. This is a string composed of target:update
        :type key: str
        :return: Message
        """
        m = self.prot_list[0].messages[key]
        self.del_message(key)
        return m

    def get_features(self):
        """
        Getter function for retrieving plotting features
        :return: List with x coordinate, list with y coordinate, list with colours, list with edge colours.
        """
        c = list(map(lambda p: p.color, self.prot_list))
        x = list(map(lambda p: p.position[0], self.prot_list))
        y = list(map(lambda p: p.position[1], self.prot_list))
        edge = 'blue'
        return x, y, c, len(self.prot_list) * [edge]


class InfoProtein(Protein):
    """
    Information protein class. Signalling happens predominantly through these proteins. The default assumes that
    every protein has one information protein through which they receive their messages (although that information
    sharing is not limited to them).
    """
    pass


class Rad3(Protein):
    def __init__(self, p_inter, p_info, pos_dim):
        """
        Rad3 protein class. Although p_inter and p_info can be passed, it is usually assumed that the default values
        are used.
        :param p_inter: Interaction profile
        :type p_inter: dict
        :param p_info: Sharing information probability
        :type p_info: dict
        :param pos_dim: Number of spatial dimensions.
        :type pos_dim: int
        """
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05,
                '%s:%s' % (.0, 1.): .05,  # Some random interaction with the DNA
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .3,
                Protein.IC_RAD3: .9,
                Protein.POL2: .1,
                Protein.IC_POL2: .1,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .1,
            }
        super().__init__(Protein.RAD3, p_inter, p_info, Protein.COLORS[Protein.RAD3], pos_dim)


class InfoRad3(InfoProtein):
    def __init__(self, p_inter, p_info, pos_dim):
        """
        Rad3 information protein class. Although p_inter and p_info can be passed, it is usually assumed that the
        default values are used.
        :param p_inter: Interaction profile
        :type p_inter: dict
        :param p_info: Sharing information probability
        :type p_info: dict
        :param pos_dim: Number of spatial dimensions.
        :type pos_dim: int
        """
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05,
                '%s:%s' % (.0, 1.): .05,  # Some random interaction with the DNA
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .9,
                Protein.IC_RAD3: .9,
                Protein.POL2: .1,
                Protein.IC_POL2: .9,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .9,
            }
        super().__init__(Protein.IC_RAD3, p_inter, p_info, Protein.COLORS[Protein.IC_RAD3], pos_dim)


class Pol2(Protein):
    def __init__(self, p_inter, p_info, pos_dim):
        """
        Pol2 protein class. Although p_inter and p_info can be passed, it is usually assumed that the default values
        are used.
        :param p_inter: Interaction profile
        :type p_inter: dict
        :param p_info: Sharing information probability
        :type p_info: dict
        :param pos_dim: Number of spatial dimensions.
        :type pos_dim: int
        """
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .1,
                Protein.IC_POL2: .05,
                Protein.RAD26: .7,
                Protein.IC_RAD26: .05,
                '%s:%s' % (.0, 1.): .005,  # Some random interaction with the DNA
            }

        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .1,
                Protein.POL2: .3,
                Protein.IC_POL2: .9,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .1,
            }
        super().__init__(Protein.POL2, p_inter, p_info, Protein.COLORS[Protein.POL2], pos_dim)


class InfoPol2(InfoProtein):
    def __init__(self, p_inter, p_info, pos_dim):
        """
        Pol2 information protein class. Although p_inter and p_info can be passed, it is usually assumed that the
        default values are used.
        :param p_inter: Interaction profile
        :type p_inter: dict
        :param p_info: Sharing information probability
        :type p_info: dict
        :param pos_dim: Number of spatial dimensions.
        :type pos_dim: int
        """
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05,
                '%s:%s' % (.0, 1.): .05,  # Some random interaction with the DNA
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .9,
                Protein.POL2: .9,
                Protein.IC_POL2: .9,
                Protein.RAD26: .1,
                Protein.IC_RAD26: .9,
            }
        super().__init__(Protein.IC_POL2, p_inter, p_info, Protein.COLORS[Protein.IC_POL2], pos_dim)


class Rad26(Protein):
    def __init__(self, p_inter, p_info, pos_dim):
        """
        Rad26 protein class. Although p_inter and p_info can be passed, it is usually assumed that the default values
        are used.
        :param p_inter: Interaction profile
        :type p_inter: dict
        :param p_info: Sharing information probability
        :type p_info: dict
        :param pos_dim: Number of spatial dimensions.
        :type pos_dim: int
        """
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .7,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .1,
                '%s:%s' % (.0, 1.): .05,  # Some random interaction with the DNA
            }
        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .1,
                Protein.POL2: .1,
                Protein.IC_POL2: .1,
                Protein.RAD26: .3,
                Protein.IC_RAD26: .9,
            }
        super().__init__(Protein.RAD26, p_inter, p_info, Protein.COLORS[Protein.RAD26], pos_dim)


class InfoRad26(InfoProtein):
    def __init__(self, p_inter, p_info, pos_dim):
        """
        Rad26 information protein class. Although p_inter and p_info can be passed, it is usually assumed that the
        default values are used.
        :param p_inter: Interaction profile
        :type p_inter: dict
        :param p_info: Sharing information probability
        :type p_info: dict
        :param pos_dim: Number of spatial dimensions.
        :type pos_dim: int
        """
        if p_inter is None:
            p_inter = {
                Protein.RAD3: .05,
                Protein.IC_RAD3: .05,
                Protein.POL2: .05,
                Protein.IC_POL2: .05,
                Protein.RAD26: .05,
                Protein.IC_RAD26: .05,
                '%s:%s' % (.0, 1.): .05,  # Some random interaction with the DNA
            }

        if p_info is None:
            p_info = {
                Protein.RAD3: .1,
                Protein.IC_RAD3: .9,
                Protein.POL2: .1,
                Protein.IC_POL2: .9,
                Protein.RAD26: .9,
                Protein.IC_RAD26: .9
            }
        super().__init__(Protein.IC_RAD26, p_inter, p_info, Protein.COLORS[Protein.IC_RAD26], pos_dim)


class ProteinFactory:
    @staticmethod
    def create(prot_type, pos_dim, p_inter=None, p_info=None):
        """
        Protein creation factory to produce proteins of any kind.
        :param prot_type: Protein type
        :type prot_type: str
        :param pos_dim: Number of spatial dimensions
        :type pos_dim: int
        :param p_inter: Interaction profile. Although it can be passed, it's recommended to use the default setup and
        to pass None.
        :type p_inter: dict
        :param p_info: Sharing information probability. Although it can be passed, it's recommended to use the default
        setup and to pass None.
        :type p_info: dict
        :return: Protein of the required type
        """
        if prot_type == Protein.RAD3:
            return Rad3(p_inter, p_info, pos_dim)
        elif prot_type == Protein.IC_RAD3:
            return InfoRad3(p_inter, p_info, pos_dim)
        elif prot_type == Protein.POL2:
            return Pol2(p_inter, p_info, pos_dim)
        elif prot_type == Protein.IC_POL2:
            return InfoPol2(p_inter, p_info, pos_dim)
        elif prot_type == Protein.RAD26:
            return Rad26(p_inter, p_info, pos_dim)
        elif prot_type == Protein.IC_RAD26:
            return InfoRad26(p_inter, p_info, pos_dim)
        else:
            raise ValueError('Protein type %s is not supported' % prot_type)

