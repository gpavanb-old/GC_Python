import pandas as pd
import numpy as np
import scipy.optimize as sp

class GroupContribution(object):

    def __init__(self,funcName,*args):

      # Define relative paths
      self.solverInputDir = 'data/Solver_Input/'
      self.compDescDir = 'data/Compound_Descriptions/'
      self.initDataDir = 'data/Init_Data/'

      relDir = './' # '../include/GroupContribution/'

      # Load XLS files
      df = pd.read_excel(relDir + self.solverInputDir + 'gani_prop_table.xlsx')
      # Get excel data in Numpy array - remove first column of names
      self.propMat = np.float64(np.array(df.values[:,1:]))

      df = pd.read_excel(relDir + self.compDescDir + funcName + '.xlsx')
      self.funcMat = np.float64(np.array(df.values[:,1:]))
      
      # Equalize sizes by padding funcMat
      pad_width = self.propMat.shape[1] - self.funcMat.shape[1]
      self.funcMat = np.pad(self.funcMat,((0,0),(0,pad_width)),'constant') 
      
      self.numGroups = self.propMat.shape[1]
      self.num_comp = self.funcMat.shape[0]

      if (len(args)==1):
          initName = args[0];
          df = pd.read_excel(relDir + self.initDataDir + initName + '.xlsx')
          self.x_l_init = np.float64(np.array(df.values[:,1:]))
      else:
          self.x_l_init = np.ones(self.num_comp)

      self.x_l_init = self.x_l_init/sum(self.x_l_init)
           
      self.MW = (1e-3 * np.dot(self.funcMat,self.propMat[9,:]))

      # Calculate critical properties
      TcCoVec = self.propMat[0,:]
      PcCoVec = self.propMat[1,:]
      VcCoVec = self.propMat[2,:]
      omegaCoVec = self.propMat[3,:]
      TbCoVec = self.propMat[12,:]
      
      # Critical volume
      self.VcVec = 1e-3*(-0.00435 + np.dot(self.funcMat,VcCoVec))
      
      # Critical temperature
      self.TcVec = 181.128*np.log(np.dot(self.funcMat,TcCoVec))
      
      # Atmospheric boiling 
      self.TbVec = 204.359*np.log(np.dot(self.funcMat,TbCoVec))
      
      # Critical pressure 
      self.PcVec = 1.3705 + np.power(np.dot(self.funcMat,PcCoVec) + 0.10022,-2.0)
      self.PcVec = self.PcVec * 101325
      
      # Acentric factor
      self.omegaVec = 0.4085 * np.power(np.log(np.dot(self.funcMat,omegaCoVec) + 1.1507),1.0/0.5050)
      
      self.kappaVec = np.zeros(len(self.omegaVec))
      for i in xrange(len(self.omegaVec)):
          if (self.omegaVec[i] <= 0.49):
              self.kappaVec[i] = 0.37464 + 1.54226*self.omegaVec[i] - 0.26992*(self.omegaVec[i]**2)
          else:
              self.kappaVec[i] = 0.379642 + 1.48503*self.omegaVec[i] - \
              0.164423*(self.omegaVec[i]**2) + 0.016666*(self.omegaVec[i]**3)
      
      # Energy interaction parameter
      df = pd.read_csv(relDir + self.solverInputDir + 'a.csv')
      self.a = np.float64(df.values[:,1:])
      
      # Lennard-Jones parameters
      # Tee, Gotoh, Stewart
      self.epsVec = (0.7915 + 0.1693 * self.omegaVec) * self.TcVec
      self.SigmaVec = (1e-10) * (2.3551 - 0.0874 * self.omegaVec)*((101325*self.TcVec/self.PcVec)**(1.0/3.0))
      
      # Standard heat of vaporization
      self.L = 6.829 + np.dot(self.funcMat,self.propMat[4,:])
      self.L = 1e3 * self.L / self.MW
            
    def c_l(self,T):
        # COMPUTE c_lVec (TODO : Change to non-standard sp. ht. capacity)
        # UNITS : (J/Kg-K)
        theta = (T-298.0)/700.0
        if (T > 700):
            print 'Drop temperature exceeded 700 K' 
        r = (np.dot(self.funcMat,self.propMat[6,:]) - 19.7779) + (np.dot(self.funcMat,self.propMat[7,:]) + 22.5981)*theta + \
            (np.dot(self.funcMat,self.propMat[8,:]) - 10.7983)*(theta**2.0)
        r = r/self.MW
        return r
      
    def D(self,p,Tin):
        # READ DIFFUSION PROPERTIES (FUNCTIONAL GROUPS)
        # Currently based on Wilke-Lee
        # TODO : Change to functional group
        eps_air = 78.6
        Sigma_air = 3.711e-10
        
        evaVec = self.epsVec*eps_air
        Tstar = Tin/evaVec
        omegaD = lambda T: 1.06036/(T**0.15610) + 0.193/np.exp(0.47635*T) + 1.03587/np.exp(1.52996*T) + 1.76474/np.exp(3.89411*T)
        MW_air = 28.97e-3
        MWa = 2e3*self.MW*MW_air/(self.MW + MW_air)
        r = 101325.0*(3.03 - 0.98/(MWa**0.5))*(1e-27)*(Tin**1.5)/(p*(MWa**0.5)*((self.SigmaVec + Sigma_air)**2.0)*omegaD(Tstar))
        return r
      
    def specVol(self,T):
        # RACKETT EQUATION
        # COMPUTE LIQUID SPECIFIC VOLUME (m3/mol) AND LIQUID DENSITY (Kg/m3)
        phiVec = np.zeros(len(self.TcVec))
        for i in xrange(len(self.TcVec)):
            if (T > self.TcVec[i]):
                phiVec[i] = - ((1 - (298/self.TcVec[i]) )**(2/7))
            else:
                phiVec[i] = ((1 - (T/self.TcVec[i]) )**(2/7)) - ((1 - (298/self.TcVec[i]) )**(2/7))
        ZcVec = (0.29056 - 0.08775*self.omegaVec)
        r = (1e-3)*(-0.00435 + np.dot(self.funcMat,self.propMat[5,:]))
        r = r*(ZcVec**phiVec)
        return r
      
    def Psat(self,T):
        # COMPUTE PSatVec
        # LEE-KESLER SATURATION VAPOR PRESSURE
        f0 = lambda Tr : 5.92714 - 6.09648/Tr - 1.28862*np.log(Tr) + 0.169347*(Tr**6)
        f1 = lambda Tr : 15.2518 - 15.6875/Tr - 13.4721*np.log(Tr) + 0.43577*(Tr**6)
        r = self.PcVec * np.exp( f0(T/self.TcVec) + self.omegaVec * f1(T/self.TcVec) )
        return r
      
    def Tb(self,p):
        # SWITCH TO BOILING VECTOR FOR ATMOSPHERIC PRESSURE
        if (p >= 1e5 and p <= 1.1e5):
            print 'Using boiling point correlation'
            r = self.TbVec
        else:
            f0 = lambda Tr : 5.92714 - 6.09648/Tr - 1.28862*np.log(Tr) + 0.169347*(Tr**6.0)
            f1 = lambda Tr : 15.2518 - 15.6875/Tr - 13.4721*np.log(Tr) + 0.43577*(Tr**6.0)
            r = np.zeros(len(self.TcVec))
            for i in xrange(len(self.TcVec)):
                sol = sp.minimize_scalar(lambda T: abs(self.PcVec[i]*np.exp(f0(T/self.TcVec[i]) + self.omegaVec[i]*
                      f1(T/self.TcVec[i]))- p),bounds=(250,800),method='bounded')
                r[i] = sol.x
        return r

    #def activity(self,xVec,T):
    #                  
    #      # R TABLE
    #      R = self.propMat(9,:)
    #      assert(length(R) == self.numGroups)
    #      
    #      # Q TABLE
    #      Q = self.propMat(10,:)
    #      assert(length(Q) == self.numGroups)
    #      
    #      # ENERGY INTERACTION PARAMETER
    #      Psi = np.exp(-self.a/T)
    #      
    #      # COORDINATION NUMBER
    #      z = 10;
    #      
    #      # r VECTOR
    #      r = (self.funcMat*R')'
    #      
    #      # q VECTOR
    #      q = (self.funcMat*Q')'
    #      
    #      # L VECTOR
    #      LVec = (z/2)*(r - q) - (r - 1)
    #      
    #      # THETA VECTOR
    #      thetaVec = xVec.*q/dot(xVec,q)
    #      
    #      # PHI VECTOR
    #      phiVec = xVec.*r/dot(xVec,r)
    #      
    #      # COMBINATORIAL ACTIVITY
    #      gammaCVec = np.exp(np.log(phiVec./xVec) + (z/2)*q.*np.log(thetaVec./phiVec) + LVec ...
    #          - (phiVec./xVec)*dot(xVec,LVec))
    #      
    #      # SET ACTIVITY OF COMPLETELY VAPORIZED SPECIES TO 1
    #      gammaCVec(isnan(gammaCVec)) = 1
    #      
    #      #------------------------------------------
    #      # RESIDUAL ACTIVITY ROUTINE
    #      #------------------------------------------
    #      
    #      # CALCULATE GROUP MOLE FRACTION
    #      XVec = xVec*self.funcMat
    #      XVec = XVec/sum(XVec)
    #      
    #      # CALCULATE Theta VECTOR
    #      ThetaVec = Q.*XVec/dot(Q,XVec)
    #      
    #      # CALCULATE FUNCTIONAL GROUP AND ISOLATED GAMMAS
    #      
    #      # AVOID DIVISION-BY-ZERO IN LAST TERM
    #      divRes = ThetaVec./(ThetaVec*Psi)
    #      divRes(isnan(divRes)) = 0
    #      
    #      GammaVec = Q.*(1 - np.log(ThetaVec*Psi) - divRes*Psi' )
    #      GammaIVec = zeros(self.num_comp,self.numGroups)
    #      
    #      # REFER TO IJHMT PAPER FOR DETAILS
    #      for i = 1:self.num_comp
    #          XIVec = self.funcMat(i,:)/sum(self.funcMat(i,:))
    #          ThetaIVec = Q.*XIVec/dot(Q,XIVec)
    #          
    #          divRes = ThetaIVec./(ThetaIVec*Psi)
    #          divRes(isnan(divRes)) = 0
    #          
    #          GammaIVec(i,:) = Q.*(1 - np.log(ThetaIVec*Psi) - divRes*Psi' )
    #          
    #      
    #      # RESIDUAL ACTIVITY
    #      gammaRVec = zeros(1,self.num_comp)
    #      for i = 1:self.num_comp
    #          gammaRVec(i) = np.exp( dot( self.funcMat(i,:) , GammaVec - GammaIVec(i,:)  ) )
    #      
    #      # TOTAL ACTIVITY
    #      ret = gammaCVec.*gammaRVec;
