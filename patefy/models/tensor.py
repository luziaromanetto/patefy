import sys
import numpy as np

#!/usr/bin/env python 
# encoding: utf-8
#
#  Patefy project
#  2015: Luzia de Menezes Romanetto
#  Ultima atualizacao 24 de julho de 2016
import json
import sys

import patefy.methods.tucker as TKD

class Tensor(object):
	def __init__(self, data = None):
		if data is None:
			self.data = None
			self.order = None
			self.shape = None
		else:
			self.data = data
			self.shape = data.shape
			self.order = len(data.shape)

class METATensor(Tensor):
	def __init__(self):
		Tensor.__init__(self)
		self.modesName = None
		self.modesDimensionName = None
		self.decomposition = None
		
	def make_decomposition(self, factors, method, options = None ):
		# Chose the method
		if len(factors) != self.order:
			raise ValueError("Invalid number of factors ", len(facts)," vs " ,self.order)
			
		if method == "ALTNTD":
			self.decomposition =  TKD.ALTNTD(self.data, factors)
		elif method == "ALSNTD":
			self.decomposition =  TKD.ALSNTD(self.data, factors)
		elif method == "HOOI":
			self.decomposition =  TKD.HOOI(self.data, factors)
		elif method == "HOOI":
			self.decomposition =  TKD.HOSVD(self.data, factors)
			
		# Make the decomposition
		if options is None:
			self.decomposition()
		else:
			self.decomposition(options)
			
	def read_relational_data(self, directory, order):
		self.order = order

		fileValues = 'vals.tsv'
		fileNames=['mt.tsv'] + ['m'+str(i)+'.tsv' for i in range(1,order) ]
    	
    	# Read the mode name
		self.modesName = []
		fin = open(directory+'/names.tsv');
		for line in fin:
			data = line.rstrip().decode('utf-8').split('\t')
			self.modesName.append( data[1].encode('utf-8') )
		fin.close();
		
		# Read the dimensions name for each mode
		self.shape = []
		self.modesDimensionName = []

		codeConvertion = []
		for fileName in fileNames:
			fin = open(directory+'/'+fileName);

			CodeXName = dict()
			for line in fin:
				data = line.rstrip().decode('utf-8').split('\t')
				CodeXName[int(data[0])] = data[1].encode('utf-8')

			self.shape.append( len(CodeXName) )
			
			keys = CodeXName.keys()
			keys.sort()
			
			featureName = []
			for i in range(len(CodeXName)):
				key = keys[i]
				featureName.append( CodeXName[key] )
				CodeXName[key] = i
				
			codeConvertion.append( CodeXName )
			self.modesDimensionName.append(featureName)
			fin.close()
		
		# Read the entries
		fin = open(directory+'/'+fileValues);
		self.data = np.zeros(self.shape);

		for line in fin:
			entries = line.split()
			ids = [ int(v) for v in entries[0:self.order] ]
			ids = [ codeConvertion[i][ids[i]] for i in range(self.order) ]
			
			self.data[tuple(ids)] = float(entries[self.order])

	def write_json(self,fileName):
		print("Saida JSON: "+fileName);
		data=dict()
		T='tensor'
		D='decomposition'
		data[T]=dict()

		data[T]['order']=self.order
		data[T]['shape']=self.shape
		data[T]['modesName']=self.modesName
		data[T]['modesDimensionName']=self.modesDimensionName
		data[T][D]=dict()
		data[T][D]['erro']=self.decomposition.error()
		data[T][D]['R']=self.decomposition.R
		data[T][D]['B']=[x.tolist() for x in self.decomposition.B]
		data[T][D]['C']=[x.tolist() for x in self.decomposition.C]
		with open(fileName,'w') as f:
			json.dump(data,f) 

