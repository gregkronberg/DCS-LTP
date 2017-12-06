"""
test mpi
"""
from mpi4py import MPI
from neuron import h
pc = h.ParallelContext()
print pc.id()[0]
# id = int(pc.id())
# nhost = int(pc.nhost())
# print "I am", id, "of", nhost