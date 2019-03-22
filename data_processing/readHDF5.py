'''
Script for reading HDF5 gene expression matrix.
Input:
  filename to the HDF5 file
  asNumpy(boolean)= return a list of numpy arrays (float32) or hdf5s
  isNumeric(boolean)=data is numeric (only affects if asNumpy=True)
Output: return a list of datasets read from the given file.
'''
#Mikko Heikkilä 2016
#Note: with characters itemsize might need fixing if something fishy happens

import numpy as np
import h5py

#voi laajentaa hdf5 kirjoittajaksi jos tarvitaan

def getHDF5data(filename, asNumpy, isNumeric):

  #open file for read-only
  f = h5py.File(filename,'r')
  
  #find objects in the file
  names = list()
  palautettava = list()
  for objectName in f: #testaa toimiiko tämä muilla datoilla
    #print(objectName)
    if asNumpy:
      apu = f.get(objectName)
      if isNumeric:
        apu2 = np.zeros(apu.shape,dtype='float32')
        apu.read_direct(apu2)
        palautettava.append(apu2)
      else:
        apu2 = np.chararray(apu.shape,itemsize='16') #this might not work
        apu.read_direct(apu2)
        palautettava.append(apu2)
    else:
      palautettava.append(f.get(object))
  return palautettava

#if __name__ == '__main__':
  #testi = getHDF5data(filename='data/geneNames_reducedToPADS.h5', asNumpy = True,isNumeric=False)
  #print(testi[0])
  
  