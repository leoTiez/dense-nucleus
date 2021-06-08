# Simulation of Sequencing Data
This software is intended simulate sequencing experiments. There are two different implementations
that intend to model the processes in the nucleus on different scales of detail.

Read the [README of the Dense Nucleus Model](README_denseNucleus.md) which implements the position and movement
of every protein around the defined DNA segment. It provides the largest level of detail. However, this also
means that it contains many degrees of freedoms (e.g. hidden variables and hyper parameters) to adjust. 
Moreover, this is computationally heavy. It is not developed any further.

A lighter implementation is a Gillespie algorithm with the notion of space. Read the 
[README here](README_spatialGillespie.md). It has less flexibility but is much quicker. This implementation is
currently developed and further improved.

## Installation
The software requires python (3.6 or higher) and pip to be installed. Run
```commandline
python3 -m pip install -r requirements.txt
```
to install the necessary packages.