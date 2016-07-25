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
    order = None
    shape = None
    modesName = None
    modesDimensionName = None
    
    data = None
    
    decomposition = None
    
    def tkd(self, factors, rest):
        self.decomposition =  NTD.Alttucker(self.data, factors, rest)
        self.decomposition()

    def writeJSON(self,fileName):
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
            json.dump(data,f)  # sort_keys=True, indent=4, separators=(',', ': ')) #-remove comment for pretty print
	
