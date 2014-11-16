# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.stdlib cimport malloc,free,calloc,rand,RAND_MAX

import numpy as np
from cython.parallel import prange
from libc.math cimport exp
cimport cython
cimport numpy as np
cimport openmp
from libcpp cimport bool
from libc.stdio cimport printf
np.import_array()


cdef extern from 'energy.h':
  cdef cppclass Energy[termType]:
    Energy(int nLabel, int nVar, int nHigher)
    void SetUnaryCost(termType*)
    void SetPairCost(int *pairs, termType*)
    int SetOneHOP(int n, int* ind, termType* weights, termType* gamma, termType Q)
    
cdef extern from 'expand.h':
  cdef cppclass AExpand[termType]:
    AExpand(Energy[termType]*, int)  
    termType minimize(int*, termType* ee)
  
def minimize(float[:,::1] unary,
             int[:,::1] pair_index,
             float[:] pair_value,
             int[:] ns,
             int[:,::1] inds,
             float[:,::1] gammas,
             float[:] Qs,
             int max_iter = 10):

    cdef int n_label = unary.shape[1]
    cdef int n_var = unary.shape[0]
    cdef int n_pair = pair_index.shape[0]
    cdef int n_higher = inds.shape[0]
    Energy[float]* energy = new Energy[float](n_label, n_var,n_pair, n_higher)

    energy.SetUnaryCost(unary)
    energy.SetPairCost(pair_index, pair_value)
    cdef int i

    for i in range(n_higher):
        pass

    AExpand[float]* expand = new AExpand[flaot](energy, max_iter)

    cdef int [:] solution = np.zeros(n_var, dtype=np.int)
    cdef float[:] ee = np.zeros(3)
    expand->minimize(solution,ee)              
    del energy 

    return solution
