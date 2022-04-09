import math

from numpy import *
import numpy as np
import numpy.linalg as la

import array_tools
import decomposition


class KronRLS(object):
    
    
    def __init__(self, **kwargs):
        Y = kwargs["train_labels"]
        Y = array_tools.as_labelmatrix(Y)
        self.Y = Y
        if kwargs.has_key('kmatrix1'):
            K1 = mat(kwargs['kmatrix1'])
            K2 = mat(kwargs['kmatrix2'])
            self.K1, self.K2 = K1, K2
            self.kernelmode = True
        else:
            X1 = mat(kwargs['xmatrix1'])
            X2 = mat(kwargs['xmatrix2'])
            self.X1, self.X2 = X1, X2
            self.kernelmode = False
        if kwargs.has_key("regparam"):
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 1.
        self.trained = False
    
    
    def createLearner(cls, **kwargs):
        learner = cls(**kwargs)
        return learner
    createLearner = classmethod(createLearner)
    
    
    def train(self):
        if self.kernelmode:
            self.solve_kernel(self.regparam)
        else:
            self.solve_linear(self.regparam)
    
    
    def solve_kernel(self, regparam):
        self.regparam = regparam
        K1, K2 = self.K1, self.K2
        Y = self.Y.reshape((K1.shape[0], K2.shape[0]), order='F')
        #assert self.Y.shape == (self.K1.shape[0], self.K2.shape[0]), 'Y.shape!=(K1.shape[0],K2.shape[0]). Y.shape=='+str(Y.shape)+', K1.shape=='+str(self.K1.shape)+', K2.shape=='+str(self.K2.shape)
        if not self.trained:
            self.trained = True
            evals1, V  = la.eigh(K1)
            evals1 = mat(evals1).T
            V = mat(V)
            self.evals1 = evals1
            self.V = V
            
            evals2, U = la.eigh(K2)
            evals2 = mat(evals2).T
            U = mat(U)
            self.evals2 = evals2
            self.U = U
            self.VTYU = V.T * self.Y * U
        
        newevals = 1. / (self.evals1 * self.evals2.T + regparam)
        
        self.A = multiply(self.VTYU, newevals)
        self.A = self.V * self.A * self.U.T
        self.model = KernelPairwiseModel(self.A)
    
    
    def solve_linear(self, regparam):
        self.regparam = regparam
        X1, X2 = self.X1, self.X2
        Y = self.Y.reshape((X1.shape[0], X2.shape[0]), order='F')
        if not self.trained:
            self.trained = True
            svals1, V, rsvecs1 = decomposition.decomposeDataMatrix(X1.T)
            self.svals1 = svals1.T
            self.evals1 = multiply(self.svals1, self.svals1)
            self.V = V
            self.rsvecs1 = mat(rsvecs1)
            
            if X1.shape == X2.shape and (X1 == X2).all():
                svals2, U, rsvecs2 = svals1, V, rsvecs1
            else:
                svals2, U, rsvecs2 = decomposition.decomposeDataMatrix(X2.T)
            self.svals2 = svals2.T
            self.evals2 = multiply(self.svals2, self.svals2)
            self.U = U
            self.rsvecs2 = mat(rsvecs2)
            
            self.VTYU = V.T * Y * U
        
        kronsvals = self.svals1 * self.svals2.T
        
        newevals = divide(kronsvals, multiply(kronsvals, kronsvals) + regparam)
        self.W = multiply(self.VTYU, newevals)
        self.W = self.rsvecs1.T * self.W * self.rsvecs2
        self.model = LinearPairwiseModel(self.W)
    
    
    def solve_linear_conditional_ranking(self, regparam):
        self.regparam = regparam
        X1, X2 = self.X1, self.X2
        Y = self.Y.reshape((X1.shape[0], X2.shape[0]), order='F')
        if not self.trained:
            self.trained = True
            svals1, V, rsvecs1 = decomposition.decomposeDataMatrix(X1.T)
            self.svals1 = svals1.T
            self.evals1 = multiply(self.svals1, self.svals1)
            self.V = V
            self.rsvecs1 = mat(rsvecs1)
            
            qlen = X2.shape[0]
            onevec = (1./math.sqrt(qlen))*mat(ones((qlen,1)))
            C = mat(eye(qlen))-onevec*onevec.T
            
            svals2, U, rsvecs2 = decomposition.decomposeDataMatrix(X2.T * C)
            self.svals2 = svals2.T
            self.evals2 = multiply(self.svals2, self.svals2)
            self.U = U
            self.rsvecs2 = mat(rsvecs2)
            
            self.VTYU = V.T * Y * C * U
        
        kronsvals = self.svals1 * self.svals2.T
        
        newevals = divide(kronsvals, multiply(kronsvals, kronsvals) + regparam)
        self.W = multiply(self.VTYU, newevals)
        self.W = self.rsvecs1.T * self.W * self.rsvecs2
        self.model = LinearPairwiseModel(self.W)
    
    
    def imputationLOO(self):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P = X1 * self.W * X2.T
        else:
            P = self.K1 * self.A * self.K2.T
        
        newevals = multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        #for i in range(self.V.shape[0]):
            #cache = Vsqr[i] * newevals.T
            #for j in range(self.U.shape[0]):
            #    ccc = (cache * Usqr[j].T)[0, 0]
            #    loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
            #    #loopred[i, j] = P[i, j]
        ccc = Vsqr * newevals.T * Usqr.T
        loopred = multiply(1. / (1. - ccc), P - multiply(ccc, self.Y))
        return loopred
    
    
    def compute_ho(self, row_inds, col_inds):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P_ho = X1[row_inds] * self.W * X2.T[:, col_inds]
        else:
            P_ho = self.K1[row_inds] * self.A * self.K2.T[:, col_inds]
        
        newevals = multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        
        rowcount = len(row_inds)
        colcount = len(col_inds)
        hosize = rowcount * colcount
        
        VV = mat(zeros((rowcount * rowcount, self.V.shape[1])))
        UU = mat(zeros((colcount * colcount, self.U.shape[1])))
        
    
    def nested_imputationLOO(self, outer_row_coord, outer_col_coord,):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P = X1 * self.W * X2.T
        else:
            P = self.K1 * self.A * self.K2.T
        P_out = P[outer_row_coord, outer_col_coord]
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        newevals = multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        d = (Vsqr[outer_row_coord] * newevals.T * Usqr[outer_col_coord].T)[0, 0]
        dY_out = d * Y_out
        
        Vox = multiply(self.V, self.V[outer_row_coord])
        Uoy = multiply(self.U, self.U[outer_col_coord])
        
        cache = Vsqr * newevals.T * Usqr.T
        crosscache = Vox * newevals.T * Uoy.T
        
        loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        a = cache
        bc = crosscache
        invdetGshift = divide(1., ((1. - a) * (1. - d) - multiply(bc, bc)))
        invGshift_1 = invdetGshift * (1. - d)
        invGshift_2 = multiply(invdetGshift, bc)
        Y = self.Y
        temp1 = P - (multiply(a, Y) + bc * Y_out)
        temp2 = P_out - (multiply(bc, Y) + dY_out)
        loopred = multiply(invGshift_1, temp1) + multiply(invGshift_2, temp2)
        return loopred
    
    
    def nested_imputationLOO_BU(self, outer_row_coord, outer_col_coord):
        P = self.K1.T * self.A * self.K2
        P_out = P[outer_row_coord, outer_col_coord]
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        newevals = multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        d = (Vsqr[outer_row_coord] * newevals.T * Usqr[outer_col_coord].T)[0, 0]
        dY_out = d * Y_out
        
        Vox = multiply(self.V, self.V[outer_row_coord])
        Uoy = multiply(self.U, self.U[outer_col_coord])
        
        cache = Vsqr * newevals.T * Usqr.T
        crosscache = Vox * newevals.T * Uoy.T
        
        loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        for i in range(self.V.shape[0]):
            #cache = Vsqr[i] * newevals.T
            #crosscache = Vox[i] * newevals.T
            jinds = range(self.U.shape[0])
            if i == outer_row_coord:
                jinds.remove(outer_col_coord)
            for j in jinds:
                #a = (cache * Usqr[j].T)[0, 0]
                a = cache[i, j]
                #bc = (crosscache * Uoy[j].T)[0, 0]
                bc = crosscache[i, j]
                #G = mat([[a, bc], [bc, d]])
                #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
                #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
                invdetGshift = 1. / ((1. - a) * (1. - d) - bc * bc)
                invGshift_1 = invdetGshift * (1. - d)
                invGshift_2 = invdetGshift * bc
                #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
                #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
                #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
                Yij = self.Y[i, j]
                temp1 = P[i, j] - (a * Yij + bc * Y_out)
                temp2 = P_out - (bc * Yij + dY_out)
                loopred[i, j] = invGshift_1 * temp1 + invGshift_2 * temp2
                #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
                #loopred[i, j] = P[i, j]
        return loopred
    
    
    def prepareLooCaches(self):
        
        #Hirvee hakkerointi
        if not hasattr(self, "Vsqr"):
            self.Vsqr = multiply(self.V, self.V)
            self.Usqr = multiply(self.U, self.U)
        self.newlooevals = multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        self.P = self.K1.T * self.A * self.K2
        self.newlooevalsUsqr = self.newlooevals.T * self.Usqr.T
        self.Vsqrnewlooevals = self.Vsqr * self.newlooevals.T
        self.Vcache = self.Vsqr * self.newlooevalsUsqr
        self.Ucache = self.Vsqrnewlooevals * self.Usqr.T
        self.diagGcache = self.Vsqr * self.newlooevals.T * self.Usqr.T
    
    
    def nested_imputationLooApproximation(self, outer_row_coord, outer_col_coord):
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        #d = (self.Vsqr[outer_row_coord] * self.newlooevals.T * self.Usqr[outer_col_coord].T)[0, 0]
        ddd = self.diagGcache[outer_row_coord, outer_col_coord]
        dY_out = ddd * Y_out
        one_minus_d = 1. - ddd
        
        #P_col = self.K1.T * (self.A * self.K2[:, outer_col_coord])
        #P_row = (self.K1[outer_row_coord] * self.A) * self.K2
        P_out = self.P[outer_row_coord, outer_col_coord]
        
        #Vox = multiply(self.V, self.V[outer_row_coord])
        #Uoy = multiply(self.U, self.U[outer_col_coord])
        
        #cache = self.Vsqr * self.newlooevals.T * self.Usqr.T
        #crosscache = Vox * self.newlooevals.T * Uoy.T
        #Vcache = self.Vsqr * (self.newlooevalsUsqr[:, outer_col_coord])
        #Vcrosscache = Vox * (self.newlooevals.T * Uoy[outer_col_coord].T)
        Vcrosscache = self.V * multiply(self.V[outer_row_coord].T, self.newlooevalsUsqr[:, outer_col_coord])
        
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        
        VcrosscacheSqr = multiply(Vcrosscache, Vcrosscache)
        
        a = self.Vcache[:, outer_col_coord]
        #bc = (crosscache * Uoy[j].T)[0, 0]
        bc = Vcrosscache
        #G = mat([[a, bc], [bc, d]])
        #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
        #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
        invdetGshift = 1. / ((1. - a) * one_minus_d - VcrosscacheSqr)
        invGshift_1 = invdetGshift * one_minus_d
        invGshift_2 = multiply(invdetGshift, bc)
        #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
        #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
        #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
        Y_j = self.Y[:, outer_col_coord]#self.Y[i, outer_col_coord]
        temp1 = self.P[:, outer_col_coord] - (multiply(a, Y_j) + bc * Y_out)
        temp2 = P_out - (multiply(bc, Y_j) + dY_out)
        loocolumn = multiply(invGshift_1, temp1) + multiply(invGshift_2, temp2)
        #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
        #loopred[i, j] = P[i, j]
        
        #Ucache = self.Vsqrnewlooevals[outer_row_coord] * self.Usqr.T
        Ucrosscache = multiply(self.Vsqrnewlooevals[outer_row_coord], self.U[outer_col_coord]) * self.U.T
        
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        '''
        for j in range(self.Y.shape[1]):
            if j == outer_col_coord: continue
            #a = (cache * self.Usqr[j].T)[0, 0]
            a = self.Ucache[outer_row_coord, j]
            #bc = (crosscache * Uoy[j].T)[0, 0]
            bc = Ucrosscache[0, j]
            #G = mat([[a, bc], [bc, d]])
            #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
            #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
            invdetGshift = 1. / ((1. - a) * one_minus_d - bc * bc)
            invGshift_1 = invdetGshift * one_minus_d
            invGshift_2 = invdetGshift * bc
            #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
            #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
            #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
            Yij = self.Ylist[outer_row_coord][j]#self.Y[outer_row_coord, j]
            temp1 = self.P[outer_row_coord, j] - (a * Yij + bc * Y_out)
            temp2 = P_out - (bc * Yij + dY_out)
            loopred[outer_row_coord, j] = invGshift_1 * temp1 + invGshift_2 * temp2
            #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
            #loopred[i, j] = P[i, j]
        '''
        
        UcrosscacheSqr = multiply(Ucrosscache, Ucrosscache)
        
        #a = (cache * self.Usqr[j].T)[0, 0]
        a = self.Ucache[outer_row_coord]
        #bc = (crosscache * Uoy[j].T)[0, 0]
        bc = Ucrosscache
        #G = mat([[a, bc], [bc, d]])
        #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
        #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
        invdetGshift = 1. / ((1. - a) * one_minus_d - UcrosscacheSqr)
        invGshift_1 = invdetGshift * one_minus_d
        invGshift_2 = multiply(invdetGshift, bc)
        #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
        #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
        #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
        Yi = self.Y[outer_row_coord]#self.Y[outer_row_coord, j]
        temp1 = self.P[outer_row_coord] - (multiply(a, Yi) + multiply(bc, Y_out))
        temp2 = P_out - (multiply(bc, Yi) + dY_out)
        loorow = multiply(invGshift_1, temp1) + multiply(invGshift_2, temp2)
        #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
        #loopred[i, j] = P[i, j]
        
        loocolumn[outer_row_coord] = 0
        loorow[:, outer_col_coord] = 0
        return loocolumn, loorow
    
    
    def getModel(self):
        return self.model

    
class KernelPairwiseModel(object):
    
    def __init__(self, A, kernel = None):
        """Initializes the dual model
        @param A: dual coefficient matrix
        @type A: numpy matrix"""
        self.A = A
        self.kernel = kernel
    
    
    def predictWithKernelMatrices(self, K1pred, K2pred):
        """Computes predictions for test examples.

        Parameters
        ----------
        K1pred: {array-like, sparse matrix}, shape = [n_samples1, n_basis_functions1]
            the first part of the test data matrix
        K2pred: {array-like, sparse matrix}, shape = [n_samples2, n_basis_functions2]
            the second part of the test data matrix
        
        Returns
        ----------
        P: array, shape = [n_samples1, n_samples2]
            predictions
        """
        P = np.array(K1pred * self.A * K2pred.T)
        return P


class LinearPairwiseModel(object):
    
    def __init__(self, W):
        """Initializes the linear model
        @param W: primal coefficient matrix
        @type W: numpy matrix"""
        self.W = W
    
    
    def predictWithDataMatrices(self, X1pred, X2pred):
        """Computes predictions for test examples.

        Parameters
        ----------
        X1pred: {array-like, sparse matrix}, shape = [n_samples1, n_features1]
            the first part of the test data matrix
        X2pred: {array-like, sparse matrix}, shape = [n_samples2, n_features2]
            the second part of the test data matrix
        
        Returns
        ----------
        P: array, shape = [n_samples1, n_samples2]
            predictions
        """
        P = np.array(X1pred * self.W * X2pred.T)
        return P


