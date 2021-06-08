# Dense Nucleus Model

Nuclear and cellular processes can be modelled through
predefined programs that are executed through events that are emitted or actions that are applied. Running
several hundred/thousand of these nuclear processes in parallel permits the simulation of a cell culture. 
Associated proteins can be counted to produce distribution signals, similar to ChIP-seq or SLAM-seq data.

## The Model
The underlying assumption is that there is no global knowledge available. Therefore, every protein, protein complex
and DNA segment acts as an independent agent that can randomly interact with other parts in the nucleus. Through these
interactions, proteins (or protein complexes) can either create new protein complexes or interact with the DNA (to be
specific, with particular segments of the DNA). However they can also only share information through sending 
messages. Through these messages it is able to increase or decrease the interaction probabilities for protein:protein
or protein:DNA interactions. In order to make information transfer efficient, it is assumed that the nucleus is
densely packed with proteins that could receive and send information. All proteins move constantly and randomly 
through the space in the nucleus. This is why it is called the Dense Nucleus model.

![Random interactions in the cellular nucleus](animations/random_interaction_dna_example_nucleus_ani.gif)

### Messages
Messages contain information for a target protein how it is supposed to update its interaction profile. For example, 
a message could contain the information that a Rad3 protein is supposed to update its interaction probability with
Pol2 to 0.3 (or 30%). Every protein can carry information. However, they are not equally likely to share information.
This permits the implementation for signalling cascades. For example, Rad3 has a much larger probability to receive 
a message from a Rad3 information protein (an abstract protein that is assumed to be involved in some signalling cascade)
rather than from a Rad26 protein. Yet every protein could share information with every other protein in theory, if 
their probability is not set to zero. When a message is sent, the protein that carried the information previously delete
the message from its own list. This ensures that information does not grow exponentially. If the receiving protein
is either the target protein or the protein with which the interaction profile is changed, it updates its probabilities
and deletes the message.

### Interaction and Sharing Information
Proteins can create complexes or share information when they are physically close enough. Therefore, there must be in
a certain distance to each other. All proteins possess information about how likely it is for them to share information
or to interact with them, i.e. create complexes (or in case of DNA segments associating to them). A coin tossing 
procedure determines whether proteins (DNA segment) interact/share information with each other. If a randomly drawn 
number between 0 and 1 is below their interaction/information probability, they interact to create complexes or share
information. It is distinguished between sharing information (intended to allow signalling cascades) and creating
stable complexes (intended to form a new unit). Usually, sharing information is much more likely than creating stable
protein complexes.

### Protein Complexes
Proteins can form complexes to act as a unit together. They share all the same information, move into the same direction,
and can have specific interaction profiles. For example, the protein complex Pol2:Rad26 might be more likely
to associate to some area of the DNA than either of them alone. However, in every simulation update step it is checked
whether they still form a stable complex. Therefore, they could collapse at any point of time again (if their
interaction probability is not set to 1.).

### DNA Segments
There is one DNA molecule that maintains different segments. It assigns proteins to the correct segments and
updates their states accordingly. Each segment has a starting and a stopping point. Proteins in overlapping segments
are associated to all of these segments.
A DNA segment has two chief roles. Firstly, it can emit events. For example, a DNA segment can emit messages to increase 
interaction probability with it in order to facilitate recruitment. They contain additionally customised conditions,
which define when to start emitting the message and when to stop. Conditions are always dependent on the state of the
DNA segment (i.e. what and how many proteins are recruited). They cannot be made dependent on time (as there is no 
global knowledge about time). Secondly, they can apply actions to associated proteins. Events and actions are almost
equivalent. Both can update and change the interaction profiles of proteins. However, actions are only applied to 
proteins that are already bound to the DNA segment. Moreover, they can apply callback functions. For example, Pol2 
can be pushed forward to simulate transcription.

### DNA Damage
DNA damage is a special type of DNA segment that dominates other segments. Proteins that associate with damage do not
automatically associate to all other overlapping segments. Therefore, whilst the action from the DNA damage are applied,
the actions from all other overlapping segments are not. With DNA damage you can define segments that overwrite the 
behaviour of other segments.

### Global Event
Environmental changes like UV radiation are the only circumstances that can trigger global responses. They are transmitted
through global messages. That means that every protein updates their interaction profile according to the new
definition.

### Cell Culture/Petri dish
All these processes can be run in parallel for several cells at one time. Hence, associated proteins can be anlysed
among several cells in a simulated cell culture. This produces simulated sequencing data like ChIP-seq and SLAM-seq
signals. Note that some processes need some time to appear in the simulated sequencing data. The example below
needs around 20 update steps until the behaviour of Rad3 (association to the core promoter between 0.0 and 0.1) and Pol2
(association to the TSS between 0.1 and 0.15 and successive elongation) is perceivable. After 40 time steps, the cell
is radiated (lesion site between 0.4 and 0.5).  If Pol2(-Rad26) moves on the DNA damage, Pol2(-Rad26) is stalled. Rad26
must be present at the lesion before Rad3 is recruited. That means that if Rad26 is already present (because the
Pol2-Rad26 complex got stalled), Rad3 can be recruited right away. Otherwise, Rad26 needs to be associated first 
before Rad3 can attach. This becomes clear trhough the ChIP-seq peaks between 0.4 and 0.5 that start to build up after 
time step 40.

![Simulated ChIP-seq data](animations/example_chipseq_ani.gif)