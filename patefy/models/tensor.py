#!/usr/bin/env python 
# encoding: utf-8
#
#  Patefy project
#  2015: Luzia de Menezes Romanetto
import sys
import numpy as np

import json
from StringIO import StringIO

import patefy.methods.tucker as mtkd
import patefy.models.tucker as ttkd

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
            self.decomposition =  mtkd.ALTNTD(self.data, factors)
        elif method == "ALSNTD":
            self.decomposition =  mtkd.ALSNTD(self.data, factors)
        elif method == "HOOI":
            self.decomposition =  mtkd.HOOI(self.data, factors)
        elif method == "HOOI":
            self.decomposition =  mtkd.HOSVD(self.data, factors)
        
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
        self.decomposition.track_unique_paths()

        data[T]['order']=self.order
        data[T]['shape']=self.shape
        data[T]['modesName']=self.modesName
        data[T]['modesDimensionName']=self.modesDimensionName
        data[T][D]=dict()
        data[T][D]['erro']=self.decomposition.error()
        data[T][D]['R']=self.decomposition.R
        data[T][D]['B']=[x.tolist() for x in self.decomposition.B]
        data[T][D]['C']=[x.tolist() for x in self.decomposition.C]
        data[T][D]['uniquePaths'] = self.decomposition.uniquePaths
        data[T][D]['pathDistances'] = [x.tolist() for x in self.decomposition.pathDistances] if self.decomposition.pathDistances is not None else 0
        data[T][D]['pathProjection'] = [[x.tolist() for x in el] for el in self.decomposition.pathProjection] if self.decomposition.pathProjection is not None else 0
        
        with open(fileName,'w') as f:
            json.dump(data,f) 
        

    def read_json(self, fileName):
        with open(fileName,'r') as f:
            data = json.load(f)
        
        T='tensor'
        D='decomposition'
        
        self.data = None
        self.order = data[T]['order']
        self.shape = data[T]['shape']
        self.modesName = data[T]['modesName']
        self.modesDimensionName = data[T]['modesDimensionName']
        
        error = data[T][D]['erro']
        R = data[T][D]['R']
        B = [np.asarray(b) for b in data[T][D]['B']]
        C = np.asarray(data[T][D]['C'])
        paths = data[T][D]['uniquePaths']
        
        self.decomposition = ttkd.TKD()
        self.decomposition.C = C
        self.decomposition.B = B
        self.decomposition.R = R
        self.decomposition.N = self.order
        self.decomposition.err = error
        self.decomposition.uniquePaths = paths
        
