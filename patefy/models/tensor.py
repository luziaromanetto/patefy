import sys
import numpy as np

#!/usr/bin/env python 
# encoding: utf-8
#
#  Patefy project
#  2015: Luzia de Menezes Romanetto
#  Ultima atualizacao 24 de julho de 2016
import json

import patefy.methods.tucker as NTD

class Tensor(object):
	def __init__(self):
		self.order = None
		self.shape = None
		self.modesName = None
		self.modesDimensionName = None
		self.data = None
		self.decomposition = None
    
	def tkd(self, factors, rest):
		self.decomposition =  NTD.Alttucker(self.data, factors, rest)
		self.decomposition()

	def read_relational_data(self, directory, fileValues, fileNames):
		self.order = len(fileNames)

		self.modesName = []
		fin = open(directory+'/names.tsv');
		for line in fin:
			data = line.rstrip().decode('utf-8').split('\t')
			self.modesName.append( data[1].encode('utf-8') )
		fin.close();

		# Read the mode name
		self.shape = []
		self.modesDimensionName = []

		#modesDimensionCode = []
		for fileName in fileNames:
			fin = open(directory+'/'+fileName);

			featureName = []
			#featureCode = []
			for line in fin:
				data = line.rstrip().decode('utf-8').split('\t')
				featureName.append( data[1].encode('utf-8') )
				#featureCode.append( int(data[0]) )

			self.shape.append( len(featureName ) )
			self.modesDimensionName.append(featureName)
			#modesDimensionCode.append(featureCode)
			fin.close()
		
		# Read the entries
		fin = open(directory+'/'+fileValues);
		self.data = np.zeros(self.shape);

		print self.shape
		for line in fin:
			entries = line.split()
			ids = [ int(v) for v in entries[0:self.order] ]
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
		data[T][D]['R']=self.decomposition.R
		data[T][D]['B']=[x.tolist() for x in self.decomposition.B]
		data[T][D]['C']=[x.tolist() for x in self.decomposition.C]
		with open(fileName,'w') as f:
			json.dump(data,f) 

