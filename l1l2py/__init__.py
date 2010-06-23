"""
L1L2Py (l1l2py) is a package to perform feature selection by means of l1l2
regularization with double optimization.

Main functions
--------------
l1l2py.minimal_model
    Aims at finding the minimal (in terms of number of selected variables)
    model within a k-fold cross validation loop.
l1l2py.nested_models
    Identifies the set of relevant lists of variables for increasing
    correlation among them.
l1l2py.model_selection
    Runs the previous two stages sequentially.

Available modules
------------------
l1l2py.algorithms
    Efficiently implements the underlying optimization algorithms.
l1l2py.tools
    Implements miscellaneous useful tools.

"""
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

from _core import *
import algorithms
import tools

from _version import version as __version__
