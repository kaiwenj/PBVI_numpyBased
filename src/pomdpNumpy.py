

import numpy as np


class StateEstimator(object):
    
    def __init__(self, transitionMatrix, observationMatrix):
        self.transitionMatrix=transitionMatrix
        self.observationMatrix=observationMatrix
        
    def __call__(self, b, a, o):
        stateEstimateAfterTransition=np.dot(self.transitionMatrix[:, a, :].T, b)
        observationCorrection=self.observationMatrix[:, a, o]*stateEstimateAfterTransition
        if observationCorrection.sum()==0:
            return observationCorrection
        bPrime=observationCorrection/observationCorrection.sum()
        return bPrime




    
class PBVI(object):
    
    def __init__(self, improve, expand, getPolicy, V, expansionNumber):
        self.improve=improve
        self.expand=expand
        self.getPolicy=getPolicy
        self.V=V
        self.expansionNumber=expansionNumber
        
    def __call__(self, B):
        V=self.V
        for i in range(self.expansionNumber):
            V=self.improve(V, B)
            B=self.expand(B)
        return V
    
class Improve(object):
    
    def __init__(self, backup):
        self.backup=backup
        
    def __call__(self, V, B):
        VNew=V.copy()
        new=V.copy()
        while True:
            alphaSet=[self.backup(V, b) for b in B]
            new=[alpha for alpha in alphaSet if alpha['alpha'] not in VNew['alpha']]
            if new == []:
                break
            newAlpha=np.array([alpha['alpha'] for alpha in new])
            newAction=np.array([alpha['action'] for alpha in new])
            VNew['alpha']=np.vstack((VNew['alpha'], newAlpha))
            VNew['action']=np.append(VNew['action'], newAction)
        return VNew
        


class Backup(object):
    
    def __init__(self, getBetaA, transitionMatrix):
        self.getBetaA=getBetaA
        self.transitionMatrix=transitionMatrix
            
    def __call__(self, V, b):
        betaA=np.array([self.getBetaA(V, b, a) for a in range(self.transitionMatrix.shape[1])])
        a=np.argmax(np.dot(betaA, b))
        beta=betaA[a]
        return {'action':a, 'alpha':beta}


class GetBetaA(object):
    
    def __init__(self, getBetaAO, transitionMatrix, rewardMatrix, observationMatrix, gamma):
        self.getBetaAO=getBetaAO
        self.transitionMatrix=transitionMatrix
        self.rewardMatrix=rewardMatrix
        self.observationMatrix=observationMatrix
        self.gamma=gamma
        
    def __call__(self, V, b, a):
        oneStepReward=(self.rewardMatrix[:, a, :]*self.transitionMatrix[:, a, :]).sum(axis=1)
        longTermRewardForEachbPrime=np.array([self.getBetaAO(V, b, a, o) for o in range(self.observationMatrix.shape[2])])
        longTermReward=np.dot(self.transitionMatrix[:, a, :], longTermRewardForEachbPrime.T*self.observationMatrix[:, a, :]).sum(axis=1)
        betaA=oneStepReward+self.gamma*longTermReward
        return betaA

class GetBetaAO(object):
    
    def __init__(self, se, argmaxAlpha):
        self.se=se
        self.argmaxAlpha=argmaxAlpha
        
    def __call__(self, V, b, a, o):
        bPrime=self.se(b, a, o)
        if bPrime.sum()==0:
            return bPrime
        betaAO=self.argmaxAlpha(V, bPrime)
        return betaAO
    
class Expand(object):
    
    def __init__(self, se, observationMatrix, selectBelief):
        self.se=se
        self.observationMatrix=observationMatrix
        self.selectBelief=selectBelief
        
    def __call__(self, B):
        BNew=B.copy()
        for b in B:
            successors=[self.se(b, a, o) for a in range(self.observationMatrix.shape[1]) for o in range(self.observationMatrix.shape[2])]
            successors=[element for element in successors if element.sum() != 0]
            if successors != []:
                bNew=self.selectBelief(successors, B)
                BNew=np.vstack((BNew, bNew))
        return BNew

def furthestB(successors, B):
    L1Distance=-np.Inf
    for bNew in successors:
        distance=min(abs(B-bNew).sum(axis=1))
        if distance > L1Distance:
            bFurthest=bNew
            L1Distance=distance
    return bFurthest              


def argmaxAlpha(V, b):
    index=np.argmax(np.dot(V['alpha'], b))
    maxAlpha=V['alpha'][index]
    return maxAlpha


def getPolicy(V, b):
    index=np.argmax(np.dot(V['alpha'], b))
    a=V['action'][index]
    return a    


def evaluateAction(V, b, a):
    VaAlpha=V['alpha'][V['action']==a]
    Q=max(np.dot(VaAlpha, b))
    return Q


    

def main():
    
    transitionMatrix=np.array([[[0.5, 0.5], [0.5, 0.5], [1, 0]],
                               [[0.5, 0.5], [0.5, 0.5], [0, 1]]])
    rewardMatrix=np.array([[[-100, -100], [10, 10],     [-1, -1]],
                           [[10, 10],     [-100, -100], [-1, -1]]])
    observationMatrix=np.array([[[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]],
                                [[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]]])
    
    beliefTransition=StateEstimator(transitionMatrix, observationMatrix)
    getBetaAO=GetBetaAO(beliefTransition, argmaxAlpha)
    
    gamma=0.5
    getBetaA=GetBetaA(getBetaAO, transitionMatrix, rewardMatrix, observationMatrix, gamma)
    backup=Backup(getBetaA, transitionMatrix)    
    improve=Improve(backup)
    
    expand=Expand(beliefTransition, observationMatrix, furthestB)
    
    expansionNumber=3
    V={'action': 2, 'alpha': np.array([[rewardMatrix.min()/(1-gamma) for s in range(transitionMatrix.shape[0])]])}
    pbvi=PBVI(improve, expand, getPolicy, V, expansionNumber)
    
    B=np.array([[0.05*n, 1-0.05*n] for n in range(21)])
    a=[pbvi(b) for b in B]
    print(a)
    

if __name__=="__main__":
    main()    

