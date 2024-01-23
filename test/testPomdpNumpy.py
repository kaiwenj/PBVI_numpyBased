

import sys
sys.path.append('../src/')
import numpy as np

import unittest
from numpy.testing import assert_almost_equal
from ddt import ddt, data, unpack
import pomdpNumpy as targetCode

@ddt
class TestStateEstimator(unittest.TestCase):
    
    def setUp(self):
        self.transitionMatrix=np.array([[[0, 0.5, 0.5], [1, 0, 0]],
                                        [[0.3, 0, 0.7], [0, 1, 0]],
                                        [[0.8, 0.2, 0], [0, 0, 1]]])
        self.observationMatrix=np.array([[[0.5, 0.5, 0], [0, 0, 1]],
                                         [[0.1, 0.9, 0], [0, 1, 0]],
                                         [[0.9, 0.1, 0], [1, 0, 0]]])   
    
    @data((np.array([0.1, 0.5, 0.4]), 0, 1, np.array([0.59949, 0.29847, 0.10204])))
    @unpack
    def testStateEstimatorOrdinary(self, b, a, o, expectedResult):
        se=targetCode.StateEstimator(self.transitionMatrix, self.observationMatrix)
        calculatedResult=se(b, a, o)
        assert_almost_equal(calculatedResult, expectedResult, 5)
        
    @data((np.array([1, 0, 0]), 1, 2, np.array([1, 0, 0])))
    @unpack
    def testStateEstimatorDoNotChange(self, b, a, o, expectedResult):
        se=targetCode.StateEstimator(self.transitionMatrix, self.observationMatrix)
        calculatedResult=se(b, a, o)
        assert_almost_equal(calculatedResult, expectedResult, 5)
        
    @data((np.array([1, 0, 0]), 1, 0, np.array([0, 0, 0])))
    @unpack
    def testStateEstimatorNoSuchObservation(self, b, a, o, expectedResult):
        se=targetCode.StateEstimator(self.transitionMatrix, self.observationMatrix)
        calculatedResult=se(b, a, o)
        assert_almost_equal(calculatedResult, expectedResult, 5)
        
    def tearDown(self):
        pass


@ddt
class TestGetBetaAO(unittest.TestCase):
    
    def setUp(self):
        self.transitionMatrix=np.array([[[0, 0.5, 0.5], [1, 0, 0]],
                                        [[0.3, 0, 0.7], [0, 1, 0]],
                                        [[0.8, 0.2, 0], [0, 0, 1]]])
        self.observationMatrix=np.array([[[0.5, 0.5, 0], [0, 0, 1]],
                                         [[0.1, 0.9, 0], [0, 1, 0]],
                                         [[0.9, 0.1, 0], [1, 0, 0]]])
        self.argmaxAlpha=lambda V, b: V['alpha'][np.argmax(np.dot(V['alpha'], b))]
        self.se=targetCode.StateEstimator(self.transitionMatrix, self.observationMatrix)
    
    @data(({'action':np.array([0, 1, 0]), 'alpha':np.array([[0, 1, 0], [10, 0, 1], [-1, 0, -10]])}, np.array([0.5, 0.5, 0]), 
           0, 1, np.array([10, 0, 1])))
    @unpack
    def testGetBetaAOSuccess(self, V, b, a, o, expectedResult):
        getBetaAO=targetCode.GetBetaAO(self.se, self.argmaxAlpha)
        calculatedResult=getBetaAO(V, b, a, o)
        assert_almost_equal(calculatedResult, expectedResult, 5)
        
    @data(({'action':np.array([0, 1, 0]), 'alpha':np.array([[0, 1, 5], [10, 0, 1], [-1, 0, -10]])}, np.array([1, 1, 1])/3, 
           1, 1, np.array([0, 1, 5])))
    @unpack
    def testGetBetaAOStay(self, V, b, a, o, expectedResult):
        getBetaAO=targetCode.GetBetaAO(self.se, self.argmaxAlpha)
        calculatedResult=getBetaAO(V, b, a, o)
        assert_almost_equal(calculatedResult, expectedResult, 5)
        
    @data(({'action':np.array([0, 1, 0]), 'alpha':np.array([[0, 1, 5], [10, 0, 1], [-1, 0, -10]])}, np.array([0.5, 0.5, 0]), 
           1, 0, np.array([0, 0, 0])))
    @unpack
    def testGetBetaAOFail(self, V, b, a, o, expectedResult):
        getBetaAO=targetCode.GetBetaAO(self.se, self.argmaxAlpha)
        calculatedResult=getBetaAO(V, b, a, o)
        assert_almost_equal(calculatedResult, expectedResult, 5)
        
    def tearDown(self):
        pass


@ddt
class TestGetBetaA(unittest.TestCase):
    
    def setUp(self):
        self.transitionMatrix=np.array([[[0.5, 0.5], [0.5, 0.5], [1, 0]],
                                        [[0.5, 0.5], [0.5, 0.5], [0, 1]]])
        self.rewardMatrix=np.array([[[-100, -100], [10, 10],     [-1, -1]],
                                    [[10, 10],     [-100, -100], [-1, -1]]])
        self.observationMatrix=np.array([[[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]],
                                         [[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]]])
        self.se=targetCode.StateEstimator(self.transitionMatrix, self.observationMatrix)
        self.argmaxAlpha=lambda V, b: V['alpha'][np.argmax(np.dot(V['alpha'], b))]
        self.gamma=1
    
    @data(({'action': np.array([2, 2]), 'alpha': np.array([[0.2, 0.8], [0.6, 0.8]])}, np.array([0.6, 0.4]), 0, np.array([-99.3, 10.7])))
    @unpack    
    def testGetBetaAOneObservation(self, V, b, a, expectedResult):
        getBetaAO=targetCode.GetBetaAO(self.se, self.argmaxAlpha)
        getBetaA=targetCode.GetBetaA(getBetaAO, self.transitionMatrix, self.rewardMatrix, 
                                         self.observationMatrix, self.gamma)
        calculatedResult=getBetaA(V, b, a)
        assert_almost_equal(calculatedResult, expectedResult, 5)
        
    @data(({'action': np.array([2, 2]), 'alpha': np.array([[0.2, 0.8], [0.6, 0.8]])}, np.array([0.6, 0.4]), 2, np.array([-0.4, -0.2])))
    @unpack    
    def testGetBetaATwoObservations(self, V, b, a, expectedResult):
        getBetaAO=targetCode.GetBetaAO(self.se, self.argmaxAlpha)
        getBetaA=targetCode.GetBetaA(getBetaAO, self.transitionMatrix, self.rewardMatrix, 
                                         self.observationMatrix, self.gamma)
        calculatedResult=getBetaA(V, b, a)
        assert_almost_equal(calculatedResult, expectedResult, 5)
                
    def tearDown(self):
        pass


@ddt
class TestBackup(unittest.TestCase):
    
    def setUp(self):
        self.transitionMatrix=np.array([[[0.5, 0.5], [0.5, 0.5], [1, 0]],
                                        [[0.5, 0.5], [0.5, 0.5], [0, 1]]])
        self.rewardMatrix=np.array([[[-100, -100], [10, 10],     [-1, -1]],
                                    [[10, 10],     [-100, -100], [-1, -1]]])
        self.observationMatrix=np.array([[[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]],
                                         [[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]]])
        self.se=targetCode.StateEstimator(self.transitionMatrix, self.observationMatrix)
        self.argmaxAlpha=lambda V, b: V['alpha'][np.argmax(np.dot(V['alpha'], b))]
        self.gamma=1
        self.getBetaAO=targetCode.GetBetaAO(self.se, self.argmaxAlpha)
        self.getBetaA=targetCode.GetBetaA(self.getBetaAO, self.transitionMatrix, self.rewardMatrix, 
                                         self.observationMatrix, self.gamma)
        
    
    @data(({'action':np.array([2, 2]), 'alpha': np.array([[0.2, 0.8], [0.6, 0.8]])},
            np.array([0.95, 0.05]),
            {'action':1, 'alpha':np.array([10.7, -99.3])}))
    @unpack    
    def testBackupOpen(self, V, b, expectedResult):
        backup=targetCode.Backup(self.getBetaA, self.transitionMatrix)
        calculatedResult=backup(V, b)
        self.assertEqual(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'], 5)
   
    @data(({'action':np.array([2, 2]), 'alpha': np.array([[0.2, 0.8], [0.6, 0.8]])},
            np.array([0.4, 0.6]),
            {'action':2, 'alpha':np.array([-0.4, -0.2])}))
    @unpack    
    def testBackupListen(self, V, b, expectedResult):
        backup=targetCode.Backup(self.getBetaA, self.transitionMatrix)
        calculatedResult=backup(V, b)
        self.assertEqual(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'], 5)
            
    def tearDown(self):
        pass
    

@ddt
class TestFurthestB(unittest.TestCase):
    
    @data((np.array([[2, 5], [4, 9]]), np.array([[1, 7], [5, 8]]), np.array([2, 5])))
    @unpack
    def testFurthestB(self, successors, B, expectedResult):
        calculatedResult=targetCode.furthestB(successors, B)
        assert_almost_equal(calculatedResult, expectedResult)
               
    def tearDown(self):
        pass

@ddt
class TestArgmax(unittest.TestCase):
    
    @data(({'action':np.array([2, 10]), 'alpha': np.array([[0.2, 0.8], [0.6, 0.8]])}, np.array([1, 7]), np.array([0.6, 0.8])))
    @unpack
    def testArgmaxAlphaInadmissible(self, V, b, expectedResult):
        calculatedResult=targetCode.argmaxAlpha(V, b)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action':np.array([2, 10]), 'alpha': np.array([[0.8, 0.2], [0.6, 0.8]])}, np.array([7, 1]), np.array([0.8, 0.2])))
    @unpack
    def testArgmaxAlphaInverse(self, V, b, expectedResult):
        calculatedResult=targetCode.argmaxAlpha(V, b)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action':np.array([2, 10]), 'alpha': np.array([[0.8, 0], [-0.8, 50]])}, np.array([7, 1]), np.array([-0.8, 50])))
    @unpack
    def testArgmaxAlphaNegativeAndZero(self, V, b, expectedResult):
        calculatedResult=targetCode.argmaxAlpha(V, b)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action':np.array([2, 10]), 'alpha': np.array([[0.2, 0.8], [0.6, 0.8]])}, np.array([1, 7]), 10))
    @unpack
    def testGetPolicyInadmissible(self, V, b, expectedResult):
        calculatedResult=targetCode.getPolicy(V, b)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action':np.array([2, 10]), 'alpha': np.array([[0.8, 0.2], [0.6, 0.8]])}, np.array([7, 1]), 2))
    @unpack
    def testGetPolicyAlphaInverse(self, V, b, expectedResult):
        calculatedResult=targetCode.getPolicy(V, b)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action':np.array([2, -5]), 'alpha': np.array([[0.8, 0], [-0.8, 50]])}, np.array([7, 1]), -5))
    @unpack
    def testGetPolicyAlphaNegativeAndZero(self, V, b, expectedResult):
        calculatedResult=targetCode.getPolicy(V, b)
        assert_almost_equal(calculatedResult, expectedResult)
               
    def tearDown(self):
        pass        


@ddt
class TestImprove(unittest.TestCase):
    
    @data(({'action': np.array([1, 5]), 'alpha': np.array([[2, 5], [4, 9]])}, np.array([[1, 7], [5, 8]]), 
           lambda V, b: {'action': 15, 'alpha': np.array([6, 10])},
           {'action': np.array([1, 5, 15, 15]), 'alpha': np.array([[2, 5], [4, 9], [6, 10], [6, 10]])}))
    @unpack
    def testImproveInadmissible(self, V, B, backup, expectedResult):
        improve=targetCode.Improve(backup)
        calculatedResult=improve(V, B)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
    
    @data(({'action': np.array([1, 5]), 'alpha': np.array([[2, 5], [4, 9]])}, np.array([[1, 7], [5, 8]]), 
           lambda V, b: {'action': 15, 'alpha': np.array([-6, -10])},
           {'action': np.array([1, 5, 15, 15]), 'alpha': np.array([[2, 5], [4, 9], [-6, -10], [-6, -10]])}))
    @unpack
    def testImproveAdmissible(self, V, B, backup, expectedResult):
        improve=targetCode.Improve(backup)
        calculatedResult=improve(V, B)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
        
    @data(({'action': np.array([1, 5]), 'alpha': np.array([[2, 5], [4, 9]])}, np.array([[1, 7], [5, 8]]), 
           lambda V, b: {'action': 15, 'alpha': np.array([2, 5])},
           {'action': np.array([1, 5]), 'alpha': np.array([[2, 5], [4, 9]])}))
    @unpack
    def testImproveAlreadyExist(self, V, B, backup, expectedResult):
        improve=targetCode.Improve(backup)
        calculatedResult=improve(V, B)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
               
    def tearDown(self):
        pass


@ddt
class TestExpand(unittest.TestCase):
    
    def setUp(self):
        self.observationMatrix=np.array([[[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]],
                                         [[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]]])
        
    @data((np.array([[2, 5], [4, 9]]), lambda b, a, o: np.array([3, 9]),
           lambda successors, B: successors[0],
           np.array([[2, 5], [4, 9], [3, 9], [3, 9]])))
    @unpack
    def testExpand(self, B, se, selectBelief, expectedResult):
        expand=targetCode.Expand(se, self.observationMatrix, selectBelief)
        calculatedResult=expand(B)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((np.array([[2, 5], [4, 9]]), lambda b, a, o: np.array([0, 0]),
           lambda successors, B: successors[0],
           np.array([[2, 5], [4, 9]])))
    @unpack
    def testExpandZero(self, B, se, selectBelief, expectedResult):
        expand=targetCode.Expand(se, self.observationMatrix, selectBelief)
        calculatedResult=expand(B)
        assert_almost_equal(calculatedResult, expectedResult)
               
    def tearDown(self):
        pass


@ddt
class TestEvaluateAction(unittest.TestCase):
    
    @data(({'action': np.array([1, 5]), 'alpha': np.array([[2, 5], [4, 9]])}, np.array([0.5, 0.5]), 
           1, 3.5))
    @unpack
    def testOnlyOneMatch(self, V, b, a, expectedResult):
        calculatedResult=targetCode.evaluateAction(V, b, a)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action': np.array([1, 5, 5, 5]), 'alpha': np.array([[2, 5], [4, 9], [3, 6], [5, 7]])}, np.array([0.5, 0.5]), 
           5, 6.5))
    @unpack
    def testMoreThanOneMatchesSupreme(self, V, b, a, expectedResult):
        calculatedResult=targetCode.evaluateAction(V, b, a)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action': np.array([1, 5, 5, 5]), 'alpha': np.array([[2, 5], [4, 9], [3, 15], [5, 7]])}, np.array([0.5, 0.5]), 
           5, 9))
    @unpack
    def testMoreThanOneMatchesOneOverTheOther(self, V, b, a, expectedResult):
        calculatedResult=targetCode.evaluateAction(V, b, a)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data(({'action': np.array([1, 5, 5, 5]), 'alpha': np.array([[100, 5], [4, 9], [3, 15], [5, 7]])}, np.array([0.9, 0.1]), 
           5, 5.2))
    @unpack
    def testNonUniformBelief(self, V, b, a, expectedResult):
        calculatedResult=targetCode.evaluateAction(V, b, a)
        assert_almost_equal(calculatedResult, expectedResult)


               
    def tearDown(self):
        pass




if __name__ == '__main__':
	unittest.main(verbosity=2)
