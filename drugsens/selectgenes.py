#!/bin/env python3
# Differentially private Bayesian linear regression 
# Arttu Nieminen 2016-2017, Teppo Niinimäki 2017-2018
# University of Helsinki Department of Computer Science
# Helsinki Institute of Information Technology HIIT

# GDSC/drug sensitivity data

# Selects the most relevant genes from gene expression.
# Saves the reducted gene expression, drug response data and drug names into csv files.

import scipy.io as sio
import numpy as np
from operator import itemgetter
import pandas

repr_dims = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]

datapath = 'drugsens_data/'
SelGenes70 = sio.loadmat(datapath+'SelGenes70.mat')['SelGenes70']
GenesMutations = sio.loadmat(datapath+'GenesMutations.mat')['GenesMutations']
MutationCnts = sio.loadmat(datapath+'MutationCnts.mat')['MutationCnts']

#gdsc_data_set = "GDSC_geneexpr_filtered"
#gdsc_data_type = 'rma_gene_expressions'
gdsc_data_set = "GDSC_geneexpr_filtered_redistributed"
gdsc_data_type = 'redistributed_gene_expressions'
geneexpr = pandas.read_hdf("data/%s.h5" % (gdsc_data_set), gdsc_data_type)


# Rank genes based on mutation counts from highest to lowest
#GeneNamesList = [n[0][0] for n in GeneNames]
GenesMutationsList = [n[0][0] for n in GenesMutations]
GeneIndex = []
for row in SelGenes70:
	genes = row[0][0].split('_')
	for gene in genes:
		#if gene in GeneNamesList:
		if gene in geneexpr.columns:
			#i = GeneNamesList.index(gene)
			i = geneexpr.columns.get_loc(gene)
			if gene in GenesMutationsList:
				j = GenesMutationsList.index(gene)
				GeneIndex.append((i,MutationCnts[j][0]))
			else:
				GeneIndex.append((i,0))
GeneIndex_sorted = sorted(GeneIndex,key=itemgetter(1),reverse=True)
RankedGenesInd = [t[0] for t in GeneIndex_sorted]
print('Found',len(RankedGenesInd),'/ 70 preselected genes in gene expression.')
print('Out of them,',np.count_nonzero([t[1] for t in GeneIndex]),'had mutations.')

# Pick out the most important genes from gene expression, in order of importance from high to low
#x = np.array(GeneExpression)[:,RankedGenesInd]
x = geneexpr.as_matrix()[:,RankedGenesInd]

# Save into csv
#np.savetxt('data_repr/repr-70-preselected.csv',x,delimiter=',')
for d in repr_dims:
	filename = 'data_repr/%s-preselected_%d.csv' % (gdsc_data_set, d)
	np.savetxt(filename, x[:,0:d], delimiter=',')
	print("Saved %s" % (filename))
