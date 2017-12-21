"""
test mpi
"""
from mpi4py import MPI
from neuron import h
pc = h.ParallelContext()
id = int(pc.id())
nhost = int(pc.nhost())
print "I am", id, "of", nhost

def f(x):
	f_class = F()
	print 'working'
	return f_class.f(x)

class F:
	def __init__(self):
		pass
	def f(self, x):
		return x
	# def parallel(self):
	# 	pc = h.ParallelContext()
	# 	id = int(pc.id())
	# 	nhost = int(pc.nhost())
	# 	print "I am", id, "of", nhost
	# 	# F_inst = F()
	# 	# func2 = F_inst.f
	# 	# func=f
	# 	pc.runworker()
	# 	s=0
	# 	for i in range(nhost):
	# 		pc.submit(self.f,i)
	# 	while pc.working():
	# 		s+=pc.pyret()
	# 	print s
		# pc.done()


# F_inst = F()
# func2 = F_inst.f
func=f
pc.runworker()
s=0
for i in range(nhost):
	pc.submit(func, i)
# while pc.working():
# 	s+=pc.pyret()
print s
pc.done()


