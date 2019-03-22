# Differentially private Bayesian linear regression 
# Arttu Nieminen 2016-2017, Teppo Niinimäki 2017
# University of Helsinki Department of Computer Science
# Helsinki Institute of Information Technology HIIT

# Choose omega parameters for clipping in each test case using auxiliary data
# Run: python3 clippingomega.py test
# where 
# test = 0 for test cases in tensor A (private data size vs. dimensionality)
# test = 2 for test cases in tensor C (private data size vs. privacy parameter)
# Here it is assumed the optimal privacy budget split is already found and defined in diffpri.py.



#import sys
#import diffpri as dp
import numpy as np
from common import ensure_dir_exists
import batch
import logging
#import pickle

# logging configuration
logging.basicConfig(level=logging.INFO)

#pv_size = [0,100,200,300,400,500,600,700,800]
#pv_size = [800]
pv_size = [875, 885]
dim = list(range(1, 21))
#eps = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
#eps = [1.0, 2.0, np.inf]
eps = [0.5]
mcmc = False	# True -> use uneven privacy budget split


def task(args):
  import diffpri as dp
  n, d, e = args
  logging.info("n = %d, d = %d, e = %s", n, d, e)
  if n == 0 or np.isinf(e): # no pv data -> no clipping
    wx = 0.0
    wy = 0.0
  else:
    wx, wy = dp.omega(n,d,e,mcmc)
  ensure_dir_exists("drugsens_params/clipping")
  with open("drugsens_params/clipping/wx_n%d_d%d_e%s.txt" % (n, d, e), 'w') as f:
    f.write("%s" % wx)
  with open("drugsens_params/clipping/wy_n%d_d%d_e%s.txt" % (n, d, e), 'w') as f:
    f.write("%s" % wy)
  #np.savetxt("drugsens_params/clipping/wx_n%d_d%d_e%s.p" % (n, d, e), wx)
  #np.savetxt("drugsens_params/clipping/wy_n%d_d%d_e%s.p" % (n, d, e), wy)

batch.init(task=task, args_ranges=(pv_size, dim, eps))
batch.main()


#wx = dict()
#wy = dict()
#for n in pv_size:
#  for d in dim:
#    for e in eps:
#      print((n, d, e))
#      if n == 0 or np.isinf(e): # no pv data -> no clipping
#        wx[(n, d, e)] = 0.0
#        wy[(n, d, e)] = 0.0
#      else:
#        w_x,w_y = dp.omega(n,d,e,mcmc)
#        wx[(n, d, e)] = w_x
#        wy[(n, d, e)] = w_y
#ensure_dir_exists("drugsens_params")
#with open("drugsens_params/clipping-bounds.p", 'wb') as f:
#  pickle.dump(wx, f)
#  pickle.dump(wy, f)




# Test cases
#pv_size = [0,100,200,300,400,500,600,700,800]
#dim = [5,10,15,20,25,30,35,40]	# A)
#eps = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]	# C)
#ny = len(pv_size)
#csvpath = '' 	# add path for output csv files
#mcmc = False	# True -> use uneven privacy budget split


'''
if len(sys.argv) > 1:
	test = int(sys.argv[1])
else:
	print('No test specified as a command line argument. Choose 0 or 2.')
	sys.exit()


if test == 0:
	# Test A
	nx = len(dim)
	WX = np.zeros((ny,nx),dtype=np.float)
	WY = np.zeros((ny,nx),dtype=np.float)
	for i in range(len(pv_size)):
		for j in range(len(dim)):
			n_pv = pv_size[i]
			n = n_pv	
			d = dim[j]
			e = 2.0
			if i == 0: # no pv data -> no clipping
				WX[i,j] = 0.0
				WY[i,j] = 0.0
			else:
				w_x,w_y = dp.omega(n,d,e,mcmc)
				WX[i,j] = w_x
				WY[i,j] = w_y
	np.savetxt(csvpath+'A-WX.csv',WX,delimiter=',')
	np.savetxt(csvpath+'A-WY.csv',WY,delimiter=',')

if test == 2:
	# Test C
	nx = len(eps)
	WX = np.zeros((ny,nx),dtype=np.float)
	WY = np.zeros((ny,nx),dtype=np.float)
	for i in range(len(pv_size)):
		for j in range(len(eps)):
			n_pv = pv_size[i]
			n = n_pv	
			d = 10
			e = eps[j]
			if i == 0: # no pv data -> no clipping
				WX[i,j] = 0.0
				WY[i,j] = 0.0
			else:
				w_x,w_y = dp.omega(n,d,e,mcmc)
				WX[i,j] = w_x
				WY[i,j] = w_y
	np.savetxt(csvpath+'C-WX.csv',WX,delimiter=',')
	np.savetxt(csvpath+'C-WY.csv',WY,delimiter=',')
'''