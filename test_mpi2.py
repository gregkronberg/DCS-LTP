'''
'''
from mpi4py import MPI
import time
def run_test():
	name=MPI.Get_processor_name()
	size=MPI.Get_size()
	print 'hello i am:', name, size
	time.sleep(1)

if __name__=='__main__':
	run_test()