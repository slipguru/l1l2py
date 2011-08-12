## This code is written by Salvatore Masecchia <salvatore.masecchia@unige.it>
## and Annalisa Barla <annalisa.barla@unige.it>
## Copyright (C) 2010 SlipGURU -
## Statistical Learning and Image Processing Genoa University Research Group
## Via Dodecaneso, 35 - 16146 Genova, ITALY.
##
## This file is part of L1L2Py.
##
## L1L2Py is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## L1L2Py is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with L1L2Py. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from nose.tools import *
from l1l2py.tools import *
from l1l2py.tests import _TEST_DATA_PATH

class TestKCVTools(object):

    def setup(self):
        data = np.loadtxt(_TEST_DATA_PATH)
        self.X = data[:,:-1]
        self.Y = data[:,-1]

    def test_data(self):
        assert_equals((30, 40), self.X.shape)
        assert_equals((30, ), self.Y.shape)



    


        
      



    
