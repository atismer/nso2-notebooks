from abc import ABC, abstractmethod

class SolveStrategyAbstract(ABC):
    import numpy

    def __init__(self):
        self.res_ = self.numpy.zeros((0,), float)
        
    @abstractmethod
    def do(self, A, b, phi):
        pass
    
    def terminate(self, A, b, phi):
      res = self.numpy.linalg.norm( self.numpy.matmul(A,phi)-b )
      
      self.res_ = self.numpy.append( self.res_, res )
    
      if res<1.e-8:
        print("Convergence")
        return True
      else:
        return False
    
    def reset(self):
      self.res_ = self.numpy.zeros((0,), float)

class LinearEqSys:
    import copy
    import numpy
    
    # constructor
    def __init__(self, A, b, SolveStrategy):
        if not isinstance(A, self.numpy.ndarray):
            print("Wrong type of A")
        if not isinstance(b, self.numpy.ndarray):
            print("Wrong type of b")            
        self.A_ = self.numpy.array(A)
        self.b_ = self.numpy.array(b)
        self.x_ = []
        self.strategy_ = SolveStrategy
    #solve
    def solve(self):
        self.strategy_.reset()
        phi = self.numpy.array(self.b_)
        phi = self.strategy_.do(self.A_, self.b_, phi)
        self.x_ = self.numpy.array( phi )
    #eigenvalues
    def eigenvalues(self):
        return self.numpy.linalg.eig(self.A_)
    
    def strategy(self):
        return self.strategy_

class cg(SolveStrategyAbstract):
    import numpy
    def do(self, A, b, phi):
      d = b-self.numpy.matmul(A,phi)
      r = d
      for n in range(500):
          alpha = self.numpy.dot(d,r) / (self.numpy.dot(d, self.numpy.matmul(A,d)))
          phi_o = phi
          phi = phi + alpha * d
          r_o = r
          r = r - alpha * self.numpy.matmul(A,d)
          beta = self.numpy.dot(r,r) / self.numpy.dot(r_o,r_o)
          d = r + beta * d
          if self.terminate(A,b,phi):
            return phi

class jacobi(SolveStrategyAbstract):
    def do(self, A, b, phi):
        D_inv = 1./self.numpy.diag(A)
        D_inv = D_inv * self.numpy.identity(self.numpy.size(D_inv))
        L = self.numpy.tril(A,k=-1)
        U = self.numpy.triu(A,k=+1)
        for n in range(500):
          phi_o = phi
          phi = self.numpy.matmul( self.numpy.matmul(D_inv,-L-U) , phi) + self.numpy.matmul(D_inv,b)
          if self.terminate(A,b,phi):
            return phi