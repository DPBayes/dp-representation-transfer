#!/bin/env python3
# Differentially private Bayesian linear regression 
# Arttu Nieminen 2016-2017
# University of Helsinki Department of Computer Science
# Helsinki Institute of Information Technology HIIT

# GDSC/drug sensitivity data

import sys, os
import numpy as np
from math import erf
from scipy import optimize
from scipy.stats import wishart, spearmanr
import theano.tensor as th
from theano import shared
from pymc3 import Model, Normal, Gamma, MvNormal, find_MAP, NUTS, sample, DensityDist, Deterministic
from pymc3.variational.advi import advi, sample_vp
import warnings
import matplotlib.pyplot as plt 
from matplotlib import cm

'''# Centers and L2-normalises x-data (removes columnwise mean, normalises rows to norm 1)
def xnormalise(x):
	n = x.shape[0]
	d = x.shape[1]
	if n == 0:
		return x
	else:
		z = x-np.dot(np.ones((n,1),dtype=np.float),np.nanmean(x,0).reshape(1,d))
		return np.divide(z,np.dot(np.sqrt(np.nansum(np.power(z,2.0),1)).reshape(n,1),np.ones((1,d),dtype=np.float)))'''

# Centers x-data (removes columnwise mean)
def xnormalise(x):
	n = x.shape[0]
	d = x.shape[1]
	if n == 0:
		return x
	else:
		return x-np.dot(np.ones((n,1),dtype=np.float),np.nanmean(x,0).reshape(1,d))
		

# Centers y-data (removes columnwise mean, except for columns where all samples have / all but one sample has missing drug response(s))
def ynormalise(y):
	n = y.shape[0]
	d = y.shape[1]
	if n == 0:
		return y
	else:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			m = np.nanmean(y,0)
		ind = np.where(np.sum(~np.isnan(y),0)<=1)[0]
		m[ind] = 0.0 # don't center samples of size <= 1
		return y-np.dot(np.ones((n,1),dtype=np.float),m.reshape(1,d))

'''# Clip data by projecting (TODO)
def clip(x,y,B_x,B_y):
	C = np.multiply(np.sign(x),np.minimum(np.abs(x),B_x))
	with np.errstate(invalid='ignore'):
		D = np.multiply(np.sign(y),np.minimum(np.abs(y),B_y))
	return C,D'''

# Clip data
def clip(x,y,B_x,B_y):
	C = np.multiply(np.sign(x),np.minimum(np.abs(x),B_x))
	with np.errstate(invalid='ignore'):
		D = np.multiply(np.sign(y),np.minimum(np.abs(y),B_y))
	return C,D

# Selects drug based on drugid, removes cell lines with missing drug response
def ignoreNaN(xx,yy,drugid):
	ind = np.where(np.isnan(yy[:,drugid]))
	y = np.delete(yy[:,drugid],ind,axis=0)
	x = np.delete(xx,ind,axis=0)
	return x,y

# Non-private sufficient statistics
def nxx(x):
	return np.dot(x.T,x)
def nxy(x,y):
	return np.dot(x.T,y)
def nyy(y):
	return np.dot(y.T,y)

# Precision measure: Spearman's rank correlation coefficient
def precision(y_pred,y_real):
	r = spearmanr(y_pred,y_real)[0]
	if np.isnan(r):
		return 0.0
	else:
		return r

# Precision measure: probabilistic concordance index
def pc(pred,real,sd):
	n = real.shape[0]
	pred = -1.0*pred
	real = -1.0*real
	rho = 0.0
	if sd < 0:
		sd = -1.0*sd
	for i in range(n):
		for j in range(n):
			if i<j:
				c = 0.5
				if pred[i] > pred[j]:
					c = 0.5 * (1+erf((real[i]-real[j])/(2*sd)))
				if pred[i] < pred[j]:
					c = 0.5 * (1+erf((real[j]-real[i])/(2*sd)))
				rho = rho + c
	rho = (2.0/(n*(n-1)))*rho
	return rho

# Choose optimal w_x,w_y for clipping thresholds
def omega(n,d,eps,mcmc,ln=20,p1=0.25,p2=0.70,p3=0.05):
	
	plotting_on = False
	np.random.seed(42)	# for reproducibility

	# Precision parameters (correspond to the means of the gamma hyperpriors)
	l = 1.0
	l0 = 1.0

	l1 = ln
	l2 = ln
	st = np.arange(0.1,2.1,0.1)
	lenC1 = len(st)
	lenC2 = lenC1
	err = np.zeros((lenC1,lenC2),dtype=np.float64)
	
	for i in range(l1):

		# Create synthetic data
		x = np.random.normal(0.0,1.0,(n,d))
		x = xnormalise(x)
		sx = np.std(x,ddof=1)
		b = np.random.normal(0.0,1.0/np.sqrt(l0),d)
		y = np.random.normal(np.dot(x,b),1.0/np.sqrt(l)).reshape(n,1)
		y = ynormalise(y)
		sy = np.std(y,ddof=1)
		
		# Thresholds to be tested
		cs1 = st*sx
		cs2 = st*sy

		for j in range(l2):

			# Generate noise
			if mcmc:
				U = wishart.rvs(d+1,((1.0*d)/(p1*eps))*np.identity(d),size=1)
				V = np.random.laplace(scale=(2.0*d)/(p2*eps),size=d).reshape(d,1)
			else:
				U = wishart.rvs(d+1,((2.0*d)/eps)*np.identity(d),size=1)
				V = np.random.laplace(scale=(4.0*d)/eps,size=d).reshape(d,1)

			for ci1 in range(lenC1):
				c1 = cs1[ci1]
				for ci2 in range(lenC2):
					c2 = cs2[ci2]

					# Clip data
					xc,yc = clip(x,y,c1,c2)

					# Perturbed suff.stats.
					xx = nxx(xc) + U*(c1**2.0)
					xy = nxy(xc,yc) + V*c1*c2

					# Prediction
					prec = l0*np.identity(d) + l*xx
					mean = np.linalg.solve(prec,l*xy)
					pred = np.dot(x,mean)

					# Precision
					rho = precision(pred,y)
					err[ci1,ci2] = err[ci1,ci2] + rho

	# Average
	err = err/float(l1*l2)

	# Choose best
	ind = np.unravel_index(err.argmax(),err.shape)
	w_x = st[ind[0]]
	w_y = st[ind[1]]

	if plotting_on:
		cmap = cm.viridis
		interp = 'spline16'
		cmin = np.min(err)
		cmax = np.max(err)
		ax = plt.subplot(111)
		plt.imshow(err,cmap=cmap,vmin=cmin,vmax=cmax,interpolation=interp,origin='lower',extent=[0,err.shape[1]-1,0,err.shape[0]-1])
		plt.autoscale(False)
		plt.plot(ind[1],ind[0],'rx')
		plt.colorbar()
		plt.xlabel('w_y')
		plt.ylabel('w_x')
		tix = list(st)
		plt.xticks(range(len(st)))
		a = ax.get_xticks().tolist() 
		a = tix
		ax.set_xticklabels(a)
		i = 1
		for label in ax.get_xticklabels():
			if i%2 != 0:
				label.set_visible(False)
			i = i+1
		plt.yticks(range(len(st)))
		a = ax.get_yticks().tolist() 
		a = tix
		ax.set_yticklabels(a)
		i = 1
		for label in ax.get_yticklabels():
			if i%2 != 0:
				label.set_visible(False)
			i = i+1
		plt.title('Average Spearman rank correlation coefficient on a synthetic data set')
		plt.show()

	return w_x,w_y

# Choose optimal budget split between three sufficient statistics
def budgetsplit(n,d,eps,ln=1):

	plotting_on = True
	np.random.seed(42)	# for reproducibility

	# Precision parameters (correspond to the means of the gamma hyperpriors)
	l = 1.0
	l0 = 1.0

	l1 = ln
	l2 = ln
	p = np.arange(0.05,0.95,0.05)
	lenp = len(p)
	minp = 0.03
	maxp = 0.97

	err = np.zeros((lenp,lenp),dtype=np.float64)

	for i in range(l1):

		print('\nLoop over data:',i+1,'/',l1)

		# Create synthetic data
		x = np.random.normal(0.0,1.0,(n,d))
		x = xnormalise(x)
		sx = np.std(x,ddof=1)
		b = np.random.normal(0.0,1.0/np.sqrt(l0),d)
		y = np.random.normal(np.dot(x,b),1.0/np.sqrt(l)).reshape(n,1)
		y = ynormalise(y)
		sy = np.std(y,ddof=1)

		for j in range(l2):

			print('\nLoop over noise:',j+1,'/',l2)
		
			# Generate noise
			W = wishart.rvs(d+1,((1.0*d)/eps)*np.identity(d),size=1)
			L = np.random.laplace(scale=(2.0*d)/eps,size=d).reshape(d,1)
			V = wishart.rvs(2,1.0/(2*eps),size=1)

			counter = 0

			for k in range(lenp):
				for l in range(lenp):

					# Budget split				
					p1 = p[k]
					p2 = p[l]
					p3 = 1.0-p1-p2

					counter = counter + 1
					print('\nTesting',counter,'/',lenp**2,': p1 =',p1,', p2 =',p2,', p3 =',p3)

					# Check if split is sensible
					t1 = minp <= p1 and p1 <= maxp
					t2 = minp <= p2 and p2 <= maxp
					t3 = minp <= p3 and p3 <= maxp

					if not all([t1,t2,t3]):
						print('Split not sensible.')
						continue

					# Clipping omega and thresholds
					w_x,w_y = omega(n,d,eps,True,ln=5,p1=p1,p2=p2,p3=p3)
					print('w_x =',w_x,', w_y =',w_y)
					c1 = sx * w_x 
					c2 = sy * w_y

					# Clip data
					xc,yc = clip(x,y,c1,c2)

					# Perturbed suff.stats.
					xx = nxx(xc) + W*(c1**2.0)/p1
					xy = nxy(xc,yc) + L*c1*c2/p2
					yy = nyy(yc) + V*(c2**2.0)/p3

					# Prediction
					pred = doADVI(n,xx,xy,yy,x)

					# Precision
					rho = precision(pred,y)
					err[k,l] = err[k,l] + rho

					print('Error =',rho)

	# Average
	err = err/float(l1*l2)

	# Choose best
	ind = np.unravel_index(err.argmax(),err.shape)
	p1 = p[ind[0]]
	p2 = p[ind[1]]
	p3 = 1.0 - p1 - p2

	if plotting_on:
		cmap = cm.plasma
		interp = 'spline16'
		cmin = np.min(err)
		cmax = np.max(err)
		ax = plt.subplot(111)
		plt.imshow(err,cmap=cmap,vmin=cmin,vmax=cmax,interpolation=interp,origin='lower',extent=[0,err.shape[1]-1,0,err.shape[0]-1])
		plt.autoscale(False)
		plt.plot(ind[1],ind[0],'rx')
		plt.colorbar()
		plt.xlabel('p2 (for X\'y)')
		plt.ylabel('p1 (for X\'X)')
		tix = list(p)
		plt.xticks(range(lenp))
		a = ax.get_xticks().tolist() 
		a = tix
		ax.set_xticklabels(a)
		i = 1
		for label in ax.get_xticklabels():
			if i%2 != 0:
				label.set_visible(False)
			i = i+1
		plt.yticks(range(lenp))
		a = ax.get_yticks().tolist() 
		a = tix
		ax.set_yticklabels(a)
		i = 1
		for label in ax.get_yticklabels():
			if i%2 != 0:
				label.set_visible(False)
			i = i+1
		plt.title('Privacy budget split')
		print('p1 =',p1)
		print('p2 =',p2)
		print('p3 =',p3)
		plt.show()

	return p1,p2,p3,err

# Compute prediction using ADVI
def doADVI(n,xx,xy,yy,x):
	
	d = xx.shape[0]
	ns = 5000
	seed = 42	# for reproducibility

	# Disable printing
	sys.stdout = open(os.devnull, 'w')

	# Sufficient statistics
	NXX = shared(xx)
	NXY = shared(xy)
	NYY = shared(yy)

	# Define model and perform MCMC sampling
	with Model() as model:

		# Fixed hyperparameters for priors
		b0 = Deterministic('b0',th.zeros((d),dtype='float64'))
		ide = Deterministic('ide',th.eye(d,m=d,k=0,dtype='float64'))

		# Priors for parameters
		l0 = Gamma('l0',alpha=2.0,beta=2.0)
		l = Gamma('l',alpha=2.0,beta=2.0)
		b = MvNormal('b',mu=b0,tau=l0*ide,shape=d)

		# Custom log likelihood
		def logp(xtx,xty,yty):	
			return (n/2.0)*th.log(l/(2*np.pi))+(-l/2.0)*(th.dot(th.dot(b,xtx),b)-2*th.dot(b,xty)+yty)

		# Likelihood
		delta = DensityDist('delta',logp,observed={'xtx':NXX,'xty':NXY,'yty':NYY})

		# Inference
		v_params = advi(n=ns,random_seed=seed)
		trace = sample_vp(v_params, draws=ns,random_seed=seed)	
	
	# Enable printing
	sys.stdout = sys.__stdout__

	# Compute prediction over posterior
	return np.mean([np.dot(x,trace['b'][i]) for i in range(ns)],0)

# Compute prediction by generating samples from the posterior distribution (n=n_train)
def doMCMC(n,nxx,nxy,nyy,x):

	d = nxx.shape[0]
	ns = 2000
	seed = 42	# for reproducibility

	# Disable printing
	sys.stdout = open(os.devnull, 'w')

	# Sufficient statistics
	NXX = shared(nxx)
	NXY = shared(nxy)
	NYY = shared(nyy)

	# Define model and perform MCMC sampling
	with Model() as model:

		# Fixed hyperparameters for priors
		b0 = Deterministic('b0',th.zeros((d),dtype='float64'))
		ide = Deterministic('ide',th.eye(d,m=d,k=0,dtype='float64'))

		# Priors for parameters
		l0 = Gamma('l0',alpha=2.0,beta=2.0)
		l = Gamma('l',alpha=2.0,beta=2.0)
		b = MvNormal('b',mu=b0,tau=l0*ide,shape=d)

		# Custom log likelihood
		def logp(xtx,xty,yty):	
			return (n/2.0)*th.log(l/(2*np.pi))+(-l/2.0)*(th.dot(th.dot(b,xtx),b)-2*th.dot(b,xty)+yty)

		# Likelihood
		delta = DensityDist('delta',logp,observed={'xtx':NXX,'xty':NXY,'yty':NYY})

		# Inference
		print('doMCMC: start NUTS')
		step = NUTS()
		trace = sample(ns,step,progressbar=False,random_seed=seed)		
	
	# Enable printing
	sys.stdout = sys.__stdout__

	# Compute prediction over posterior
	return np.mean([np.dot(x,trace['b'][i]) for i in range(ns)],0)

# Train (dp or np) model using MCMC sampled values for lambdas, return prediction on test data
def predictMCMC(n_train,nxx_pv,nxx_npv,nxy_pv,nxy_npv,nyy_pv,nyy_npv,B_x,B_y,e,x_test,private,p1=0.25,p2=0.70,p3=0.05):
	
	d = nxx_pv.shape[0]

	# Generate noise
	if private:
		W = wishart.rvs(d+1,((d*pow(B_x,2))/(p1*e))*np.identity(d),size=1)
		L = np.random.laplace(scale=(2*d*B_x*B_y)/(p2*e),size=d)
		V = wishart.rvs(2,pow(B_y,2)/(2*p3*e),size=1)
	else:
		W = 0
		L = 0
		V = 0

	return doMCMC(n_train,nxx_npv+nxx_pv+W,nxy_npv+nxy_pv+L,nyy_npv+nyy_pv+V,x_test)

# Train (dp or np) model using guessed values for lambdas, returns prediction on test data
def predictL(nxx_pv,nxx_npv,nxy_pv,nxy_npv,B_x,B_y,e,x_test,private):
	
	l = 1.0
	l0 = 1.0
	d = nxx_pv.shape[0]

	# Generate noise
	if private:
		W = wishart.rvs(d+1,((2*d*pow(B_x,2))/e)*np.identity(d),size=1)
		L = np.random.laplace(scale=(4*d*B_x*B_y)/e,size=d)
	else:
		W = 0.0
		L = 0.0

	# Posterior distribution
	prec = l*(nxx_npv + nxx_pv + W) + l0*np.identity(d)
	mean = np.linalg.solve(prec,l*(nxy_npv + nxy_pv + L))

	# Compute prediction
	return np.dot(x_test,mean)

# Process data for model fitting: splits, dimensionality reduction, 
# normalisation, clipping, dropping missing values.
# Returns sufficient statistics, test data, clipping thresholds, 
# number of trainings samples, and private = True if private data size > 0
def processData(x,y,d,n_test,n_pv,n_npv,pv_max,w_x,w_y,drugid,seed):
	
	n_train = n_pv + n_npv

	# Set rng seed
	np.random.seed(seed)
	np.random.seed(int(np.floor(np.random.rand()*5000)))

	# Test/training split + dimensionality reduction
	ind = np.random.permutation(x.shape[0])
	x_test = x[ind[0:n_test],0:d]
	y_test = y[ind[0:n_test],:]
	x_train = x[ind[n_test:],0:d]
	y_train = y[ind[n_test:],:]

	# Training data: private/non-private split
	x_pv = x_train[0:n_pv,:]
	y_pv = y_train[0:n_pv,:]
	x_npv = x_train[pv_max:pv_max+n_npv,:]
	y_npv = y_train[pv_max:pv_max+n_npv,:]
	
	# Normalise x-data
	x_test = xnormalise(x_test)
	x_npv = xnormalise(x_npv)
	x_pv = xnormalise(x_pv)

	# Normalise y-data
	y_test = ynormalise(y_test)
	y_npv = ynormalise(y_npv)
	y_pv = ynormalise(y_pv)

	# Clip data
	n = np.sum(~np.isnan(y_pv[:,drugid])) # number of private data
	if n == 1: # std not possible to compute => no clipping
		B_x = np.max(np.abs(x_pv))
		B_y = np.nanmax(np.abs(y_pv))
		x_pv,y_pv = clip(x_pv,y_pv,B_x,B_y)
		x_npv,y_npv = clip(x_npv,y_npv,B_x,B_y)
	elif n > 1:
		B_x = w_x * np.nanstd(x_pv,ddof=1) 
		B_y = w_y * np.nanstd(y_pv,ddof=1)
		x_pv,y_pv = clip(x_pv,y_pv,B_x,B_y)
		x_npv,y_npv = clip(x_npv,y_npv,B_x,B_y)
	else: # no pv data => no clipping
		B_x = 0.0
		B_y = 0.0

	# Select drug and drop cell lines with missing response
	x_pv,y_pv = ignoreNaN(x_pv,y_pv,drugid)
	x_npv,y_npv = ignoreNaN(x_npv,y_npv,drugid)
	x_test,y_test = ignoreNaN(x_test,y_test,drugid)
	n_train = x_pv.shape[0] + x_npv.shape[0] # update n_train
	
	# Compute suff.stats
	nxx_pv = nxx(x_pv)
	nxy_pv = nxy(x_pv,y_pv)
	nyy_pv = nyy(y_pv)
	nxx_npv = nxx(x_npv)
	nxy_npv = nxy(x_npv,y_npv)
	nyy_npv = nyy(y_npv)
	
	if x_pv.shape[0] == 0:
		private = False
	else:
		private = True

	return nxx_pv,nxx_npv,nxy_pv,nxy_npv,nyy_pv,nyy_npv,x_test,y_test,B_x,B_y,n_train,private
