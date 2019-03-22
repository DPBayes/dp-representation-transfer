'''
Script for normalizing a datamatrix to (0,1). Each column is normalized separately
based on its max absolute value.
Input: data: Data as an observations*dims-matrix. A vector is interpreted as a dim=1 matrix.
'''
import numpy as np

#NOTE: contains a bug, tai luultavammin bugi on kutsuvassa päässä

def normalize(data):
  if len(data.shape) > 1:
    dim = data.shape[1]
    for kDim in range(dim):
      dataMax = np.amax( np.absolute(data[:,kDim]))
      data[:,kDim] = data[:,kDim]/dataMax
  else:
    dataMax = np.amax(np.absolute(data))
    data = data/dataMax
  return data
  
#   dim = x_train.shape[1]  
# 	for kDim in range(dim):
# 		xMax = np.amax([np.amax(np.absolute(x_train[:,kDim])),np.amax(np.absolute(x_test[:,kDim]))])
# 		x_train[:,kDim] = x_train[:,kDim]/xMax
# 		x_test[:,kDim] = x_test[:,kDim]/xMax		
# 	yMax = np.amax([np.amax(np.absolute(y_train)),np.amax(np.absolute(y_test))])
# 	y_train = y_train/yMax
# 	y_test = y_test/yMax


