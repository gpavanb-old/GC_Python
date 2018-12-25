# PROGRAM TO CALCULATE DISTILLATION CURVES (FLASH, FRACTIONAL AND STEP-VOLUME)

# PROTOTYPES ARE
# distMoleFrac(funcName, initName, dist_type)
# distMoleFrac(funcName, init_comp, expData, dist_type)

import numpy as np
import scipy.interpolate as spi
import scipy.optimize as spo
import pandas as pd
import GC

def lhsFunc(x_l,bufVec,bet):
    # SOLVE NON-LINEAR EQUATION
    # REFER TO IJHMT PAPER FOR DETAILS
    lhs = 0.0
    for i in xrange(len(x_l)):
        lhs = lhs + ( x_l[i]/(bufVec[i] + bet) )

    return lhs
  
def RecFrac(x_l,p,T,MW,pSatVec,gammaVec):

    # PERFORM FLASH CALCULATION TO OBTAIN
    # RECOVERED MASS FRACTION
    
    # COMPUTE VAPOR MOLE FRACTION
    KVec = gammaVec*pSatVec/p
    bufVec = 1.0/(KVec-1)

    rec = spo.minimize_scalar(lambda bet: abs(lhsFunc(x_l,bufVec,bet)),bounds=(0,1),method='bounded')
    rec = rec.x

    #if (isempty(rec))
    #    rec = 0
    
    xVec = x_l/(1 + rec*(KVec-1))
    yVec = KVec*x_l
    
    # CONVERT TO MASS FRACTION AND CALCULATE RECOVERED MASS FRACTION
    rec = np.dot(MW,rec*yVec)/np.dot(MW,(1-rec)*xVec + rec*yVec)
    return rec

def distMoleFrac(funcName,init_comp,expData,dist_type):

  # TODO : Variable arguments
  relDir = './src/GC_Python'
  # LOAD EXPERIMENTAL DATA
  df = pd.read_csv(relDir + '/Output/' + expData + '.csv', header=None)
  dumpMatS = df.values

  # READ UNIFAC PARAMETERS
  fuel = GC.GroupContribution(funcName)

  # INITIAL LIQUID MASS FRACTIONS
  x_l_init = init_comp

  # SYSTEM CONDITIONS
  # SYSTEM PRESSURE (Pa)
  p = 0.1e6

  # FINAL TEMPERATURE (K)
  T_fin = 600

  # INITIAL TEMPERATURE (K)
  T_init = 270

  # NON-IDEAL FLAG (1 FOR UNIFAC)
  niFlag = 0

  # TEMPERATURE VECTOR
  dT = 1
  tempVec = np.arange(T_init,T_fin,dT)
  ## MULTICOMPONENT CRITICAL PROPERTIES

  # CREATE BOILING POINT VECTOR
  # LEE-KESLER SATURATION VAPOR PRESSURE
  MW = fuel.MW
  TbVec = fuel.Tb(p)

  # INITIALIZE THE FINAL RESULT ARRAYS
  recF = 0.0
  recT = 0.0

  # RECOVERED FLASH FRACTION
  if (dist_type == 'flash'):
    recF = np.zeros(len(tempVec))
    
    for i in xrange(len(tempVec)):
      T = tempVec[i]
      pSatVec = fuel.Psat(T)
      # COMPUTE ACTIVITIES
      if (niFlag == 1):
        gammaVec = fuel.activity(x_l_init,T)
      else:
        gammaVec = np.ones(fuel.num_comp)
        recF[i] = RecFrac(x_l_init,p,T,MW,pSatVec,gammaVec)
    recT = tempVec
 
  if (dist_type == 'fractional'):
    # RECOVERED FRACTIONAL FRACTION
    tVec = np.arange(T_init,T_fin,dT)
    recFr = np.zeros(len(tempVec))
    moleMatFr = np.zeros((len(tVec),len(x_l_init)))
    massTot = np.dot(fuel.MW,x_l_init)
    x_l = x_l_init
    for i in xrange(len(tVec)):
      # REMOVE BOILING SPECIES
      mask = ((tVec[i]/TbVec >= 1) & (x_l != 0))
      x_l[mask] = 0
          
      massFrac = np.dot(x_l,fuel.MW)/massTot
      recFr[i] = 1-massFrac
      moleMatFr[i,:] = x_l/sum(x_l)
    recT = tVec
    recF = recFr

  if (dist_type == 'step_volume'):
    # STEP-VOLUME DISTILLATION
    vStep = 0.01
    recTS = np.empty((0,1), dtype=np.float64)
    recFS = np.empty((0,1), dtype=np.float64)
    moleVec = x_l_init
    massTot = np.dot(fuel.MW,x_l_init)
    x_l = x_l_init
    while(np.any(moleVec) != 0):
      Tsol = spo.minimize_scalar(lambda T : abs(np.dot(x_l,fuel.Psat(T)) - p),bounds=(T_init,T_fin),method='bounded')
      Tsol = Tsol.x
      #if (isempty(Tsol)):
      #  disp('Solution out of specified range. Change and try again')
      recTS = np.append(recTS,Tsol)
            
      # GET VAPOR COMPOSITION
      xv = fuel.Psat(Tsol)*x_l/p
            
      # UPDATE LIQUID COMPOSITION
      moleVec = np.maximum(np.zeros(len(moleVec)),moleVec - vStep*xv)
      x_l = moleVec/sum(moleVec)
      #moleMatFS = [moleMatFS;x_l]
            
      # CALCULATE RECOVERED MASS FRACTION
      massFrac = np.dot(moleVec,fuel.MW)/massTot
      recFS = np.append(recFS,1-massFrac)
    recT = recTS
    recF = recFS

  # CREATE UNIQUE OUTPUT
  # TRUNCATE
  resMat = np.vstack((recT,recF))
  u, indices = np.unique(resMat[1,:],return_index=True)
  resMat = resMat[:,indices]

  # CALCULATE ERROR NORM
  num_pts = 50
  exp_lim = [np.amin(dumpMatS[:,0]),np.amax(dumpMatS[:,0])]
  xq = np.linspace(exp_lim[0],exp_lim[1],num_pts)
 
  res = spi.interp1d(resMat[1,:],resMat[0,:],kind='previous')
  expe = spi.interp1d(dumpMatS[:,0],dumpMatS[:,1],kind='previous')
  res = res(xq)
  expe = expe(xq)
  diff = np.trapz(xq,abs(expe-res))
  return diff
