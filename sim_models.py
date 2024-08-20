import numpy as np
import itertools

class SimpleLATE:
    def __init__(self, pZ, pXY):
            
        self.pZ = pZ
        self.pXY = pXY
            
        self.f = np.vectorize(self.canonical_f)

    def canonical_f(self, U, *args):
        idx = 0
        for bit in args:
            idx = (idx << 1) | bit
        return (U >> idx) & 1
    
    def samples(self, n = 10000):
        
        pXY = self.pXY.flatten()
        uXY = np.random.choice(pXY.size, n, p=pXY)
        uX, uY = np.unravel_index(uXY, self.pXY.shape)

        uZ = np.random.choice(self.pZ.size, n, p=self.pZ)

        Z = self.f(uZ)
        X = self.f(uX, Z)
        Y = self.f(uY, X)

        return np.column_stack([Z, X, Y])
    
    def compute(self, U, do_var = {}):

        do_vars = do_var.keys()

        Z = self.canonical_f(U[0]) if 'Z' not in do_vars else do_var['Z']
        X = self.canonical_f(U[1], Z) if 'X' not in do_vars else do_var['X']
        Y = self.canonical_f(U[2], X) if 'Y' not in do_vars else do_var['Y']

        return Z, X, Y
    
    def compute_prob(self, U):
        return self.pZ[U[0]] * self.pXY[U[1], U[2]]
    
    def prob(self, event):

        uZ = self.pZ.size
        uX, uY = self.pXY.shape

        U = itertools.product(range(uZ), range(uX), range(uY))

        prob = 0
        for u in U:
            is_event = True
            for t in event:
                Z, X, Y = self.compute(u, do_var=t.do_var)
                var = list(t.vars.keys())[0]
                if t.vars[var] != locals()[var]:
                    is_event = False
                    break
            
            if is_event:
                prob += self.compute_prob(u)
                print(u)

        return prob
    
    def cpd(self, event, condition):

        p_condition = self.prob(condition)
        if(p_condition == 0):
            return -1
        else:
            return self.prob({**event, **condition}) / p_condition

class SimplePostTreatment:
    def __init__(self, pX, pM, pY):
        self.pX = pX
        self.pM = pM
        self.pY = pY

        self.f = np.vectorize(self.canonical_f)

    def canonical_f(self, U, *args):
        idx = 0
        for bit in args:
            idx = (idx << 1) | bit
        return (U >> idx) & 1

    def samples(self, n = 10000):

        uX = np.random.choice(self.pX.size, n, p=self.pX)
        uM = np.random.choice(self.pM.size, n, p=self.pM)
        uY = np.random.choice(self.pY.size, n, p=self.pY)

        X = self.f(uX)
        M = self.f(uM, X)
        Y = self.f(uY, X, M)

        return np.column_stack([X, M, Y])

    def compute(self, U, do_var = {}):
            
        do_vars = do_var.keys()
    
        X = self.canonical_f(U[0]) if 'X' not in do_vars else do_var['X']
        M = self.canonical_f(U[1], X) if 'M' not in do_vars else do_var['M']
        Y = self.canonical_f(U[2], X, M) if 'Y' not in do_vars else do_var['Y']

        return X, M, Y

    def compute_prob(self, U):
        return self.pX[U[0]] * self.pM[U[1]] * self.pY[U[2]]

    def prob(self, event):
        uX = self.pX.size
        uM = self.pM.size
        uY = self.pY.size

        U = itertools.product(range(uX), range(uM), range(uY))

        prob = 0
        for u in U:
            is_event = True
            for t in event:
                X, M, Y = self.compute(u, do_var=t.do_var)
                var = list(t.vars.keys())[0]
                if t.vars[var] != locals()[var]:
                    is_event = False
                    break
            
            if is_event:
                prob += self.compute_prob(u)
                # print(u)

        return prob

    def cpd(self, event, condition):
        p_condition = self.prob(condition)
        if(p_condition == 0):
            return -1
        else:
            return self.prob(event + condition) / p_condition

class PostTreatment:
    def __init__(self, pWX, pM, pY):
        self.pWX = pWX
        self.pM = pM
        self.pY = pY

        self.f = np.vectorize(self.canonical_f)
    
    def canonical_f(self, U, *args):
        idx = 0
        for bit in args:
            idx = (idx << 1) | bit
        return (U >> idx) & 1
    
    def samples(self, n = 10000):

        pWX = self.pWX.flatten()
        uWX = np.random.choice(pWX.size, n, p=pWX)
        uW, uX = np.unravel_index(uWX, self.pWX.shape)

        uM = np.random.choice(self.pM.size, n, p=self.pM)
        uY = np.random.choice(self.pY.size, n, p=self.pY)

        W = self.f(uW)
        X = self.f(uX, W)
        M = self.f(uM, W, X)
        Y = self.f(uY, W, X, M)

        return np.column_stack([W, X, M, Y])
    
    def compute(self, U, do_var = {}):
        
        do_vars = do_var.keys()

        W = self.canonical_f(U[0]) if 'W' not in do_vars else do_var['W']
        X = self.canonical_f(U[1], W) if 'X' not in do_vars else do_var['X']
        M = self.canonical_f(U[2], W, X) if 'M' not in do_vars else do_var['M']
        Y = self.canonical_f(U[3], W, X, M) if 'Y' not in do_vars else do_var['Y']

        return W, X, M, Y
    
    def compute_prob(self, U):
        return self.pWX[U[0], U[1]] * self.pM[U[2]] * self.pY[U[3]]
    
    def prob(self, event):

        uW = self.pWX.shape[0]
        uX = self.pWX.shape[1]
        uM = self.pM.size
        uY = self.pY.size

        U = itertools.product(range(uW), range(uX), range(uM), range(uY))

        prob = 0
        for u in U:
            is_event = True
            for t in event:
                W, X, M, Y = self.compute(u, do_var=t.do_var)
                var = list(t.vars.keys())[0]
                if t.vars[var] != locals()[var]:
                    is_event = False
                    break
            
            if is_event:
                prob += self.compute_prob(u)
                print(u)

        return prob
    
    def cpd(self, event, condition):
        p_condition = self.prob(condition)
        if(p_condition == 0):
            return -1
        else:
            return self.prob({**event, **condition}) / p_condition

class SimpleMediation:
    def __init__(self, pZ, pXY, pM):
        self.pZ = pZ
        self.pXY = pXY
        self.pM = pM

        self.f = np.vectorize(self.canonical_f)

    def canonical_f(self, U, *args):
        idx = 0
        for bit in args:
            idx = (idx << 1) | bit
        return (U >> idx) & 1
    
    def samples(self, n = 10000):
        
        pXY = self.pXY.flatten()
        uXY = np.random.choice(pXY.size, n, p=pXY)
        uX, uY = np.unravel_index(uXY, self.pXY.shape)

        uZ = np.random.choice(self.pZ.size, n, p=self.pZ)
        uM = np.random.choice(self.pM.size, n, p=self.pM)

        Z = self.f(uZ)
        X = self.f(uX, Z)
        M = self.f(uM, X)
        Y = self.f(uY, X, M)

        return np.column_stack([Z, X, M, Y])
    
    def compute(self, U, do_var = {}):
            
        do_vars = do_var.keys()

        Z = self.canonical_f(U[0]) if 'Z' not in do_vars else do_var['Z']
        X = self.canonical_f(U[1], Z) if 'X' not in do_vars else do_var['X']
        M = self.canonical_f(U[2], X) if 'M' not in do_vars else do_var['M']
        Y = self.canonical_f(U[3], X, M) if 'Y' not in do_vars else do_var['Y']

        return Z, X, M, Y
    
    def compute_prob(self, U):
        return self.pZ[U[0]] * self.pXY[U[1], U[3]] * self.pM[U[2]]
    
    def prob(self, event):
        
        uZ = self.pZ.size
        uX, uY = self.pXY.shape
        uM = self.pM.size
    
        U = itertools.product(range(uZ), range(uX), range(uM), range(uY))
    
        prob = 0
        for u in U:
            is_event = True
            for t in event:
                Z, X, M, Y = self.compute(u, do_var=t.do_var)
                var = list(t.vars.keys())[0]
                if t.vars[var] != locals()[var]:
                    is_event = False
                    break
            
            if is_event:
                prob += self.compute_prob(u)
                print(u)
    
        return prob
    
    def cpd(self, event, condition):
            
        p_condition = self.prob(condition)
        if(p_condition == 0):
            return -1
        else:
            return self.prob({**event, **condition}) / p_condition