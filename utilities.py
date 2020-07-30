# To use CPP code in python
#from ctypes import POINTER, c_double, c_long, CDLL
import numpy as np
# the norm pdf
from scipy.stats import norm
from scipy.stats import multivariate_normal as mnorm
# for general inverse of matrix
from numpy.linalg import pinv, svd, inv
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as tmnorm
#from tqdm import  tqdm
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
# to use functions in R languge
import rpy2.robjects as robj
import pickle
from time import time as Time
from easydict import EasyDict as edict

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

## 1Darray to pointer for cpp
#def arr2pter(arr):
#    return arr.ctypes.data_as(POINTER(c_double))
#
#def arrint2pter(arr):
#    return arr.ctypes.data_as(POINTER(c_long))
#
#cppMainLoop = CDLL("example.o").MainLoop
#cppMainLoop.restype = POINTER(c_double)
#cppMainLoop.argtypes = [
#                          POINTER(c_double), 
#                          POINTER(c_double), 
#                          POINTER(c_long), 
#                          POINTER(c_long), 
#                          c_long, c_long, c_long, c_long]
#
#def wrapMainLoop(Xmat, dXmat, canpts1, canpts2, M0, Lmin, r, n):
#    Xmat = Xmat.flatten()
#    dXmat = dXmat.flatten()
#    Xmat = np.ascontiguousarray(Xmat, dtype=np.float64)
#    dXmat = np.ascontiguousarray(dXmat, dtype=np.float64)
#    return cppMainLoop(arr2pter(Xmat), arr2pter(dXmat), arrint2pter(canpts1), arrint2pter(canpts2), M0, Lmin, r, n)

def bw_nrd0_R(time):
    bw_nrd0 = robj.r["bw.nrd0"]
    time_r = robj.FloatVector(time)
    return np.array(bw_nrd0(time_r))[0]

def smooth_spline_R(x, y, lamb):
    smooth_spline_f = robj.r["smooth.spline"]
    x_r = robj.FloatVector(x)
    y_r = robj.FloatVector(y)
    args = {"x": x_r, "y": y_r, "lambda": lamb}
    spline = smooth_spline_f(**args)
    ysp = np.array(robj.r['predict'](spline, deriv=0).rx2('y'))
    ysp_dev1 = np.array(robj.r['predict'](spline, deriv=1).rx2('y'))
    return {"yhat": ysp, "ydevhat": ysp_dev1}

# effcient dynamic programming to optimize the MBIC, all the change points ase based on a python index system, main loop is based on cpp
#def cppEGenDy(tXmat, tdXmat, alpha, Lmin=None, canpts=None, MaxM=None, Taget="min", diag=False, Ms=None, savepath=None):
#    """
#    tXmat: array, r x n. n is length of sequence. 
#    canpts: candidate point set. list or array, not including 0 and n
#    MaxM: int, maximal number of change point 
#    Lmin: The minimal length between 2 change points
#    Ms: the list containing prespecified number of change points.
#       When Ms=None, it means using MBIC to determine the number of change points
#    savepath: if None, dont save. Otherwise save the intermediate results
#    """
#
#    r, n = tXmat.shape
#    if Lmin is None:
#        Lmin = r
#        
#    if Taget == "min":
#        tagf = np.min
#        tagargf = np.argmin
#        decon = np.inf
#    else:
#        tagf = np.max
#        tagargf = np.argmax
#        decon = -np.inf
#
#    if Ms is not None:
#        Ms = sorted(Ms)
#    if canpts is None:
#        canpts = np.arange(n-1)
#    else:
#        canpts = np.array(canpts)-1
#    M0 = len(canpts) # number of change point in candidate point set
#    if MaxM is None:
#        MaxM = M0 + 1
#    else:
#        MaxM = MaxM + 1
#    mbicMaxM = MaxM
#    if not (Ms is None or len(Ms)==0):
#        MaxM = Ms[-1]+1 if Ms[-1]>=(MaxM-1) else MaxM
#    canpts_full1 = np.concatenate(([-1], canpts, [n-1]))
#    canpts_full2 = canpts_full1[1:]
#
#    t = Time()
#    Hmat = np.zeros((M0+1, M0+1)) + decon
#
#    # create a matrix 
#    cppHmat = wrapMainLoop(tXmat, tdXmat, canpts_full1, canpts_full2, M0, Lmin, r, n)
#    # Here I have to convert cppHmat to array, but the speed is not faster than python code, so I didn't do it.
#    print(f"Time is {Time()-t}s")
#    # vector contains results for each number of change point
#    U = np.zeros(MaxM) 
#    U[0] = Hmat[0, -1]
#    D = Hmat[:, -1]
#    # contain the location of candidate points  (in python idx)
#    Pos = np.zeros((M0+1, MaxM-1)) + decon
#    Pos[M0, :] = np.ones(MaxM-1) * M0
#    tau_mat = np.zeros((MaxM-1, MaxM-1)) + decon
#    for k in range(MaxM-1):
#        for j in range(M0): # n = M0 + 1
#            dist = Hmat[j, j:-1] + D[(j+1):]
#            #print(dist)
#            D[j] = np.min(dist)
#            Pos[j, 0] = np.argmin(dist) + j + 1
#            if k > 0:
#                Pos[j, 1:(k+1)] = Pos[int(Pos[j, 0]), 0:k]
#        U[k+1] = D[0]
#        tau_mat[k, 0:(k+1)] = Pos[0, 0:(k+1)] - 1
#    U = U + np.log(n)**alpha * (np.arange(1, MaxM+1))*(r**2 + (r**2-r)/2) + np.log(np.arange(1,MaxM+1))
#
#    if savepath is not None:
#        with open(savepath, "wb") as f:
#            U0 = U - (np.log(n)**alpha * (np.arange(1, MaxM+1))*(r**2 + (r**2-r)/2) + np.log(np.arange(1,MaxM+1)))
#            res = {}
#            res["U0"] = U0
#            res["U"] = U
#            res["Hmat"] = Hmat
#            res["tau_mat"] = tau_mat
#            pickle.dump(res, f)
#    
#    mbic_numchg = np.argmin(U[:mbicMaxM])
#    if mbic_numchg == 0:
#        mbic_ecpts = np.array([])
#    else:
#        idx = tau_mat[int(mbic_numchg-1),: ]
#        idx = np.array(idx[idx<np.inf], dtype=np.int)
#        mbic_ecpts = np.array(canpts)[idx] + 1
#        
#    if Ms is None or len(Ms)==0:
#        if not diag:
#            return {"U":U, "mbic_ecpts": mbic_ecpts}
#        else:
#            return {"U":U, "mbic_ecpts": mbic_ecpts, "idxMat": tau_mat}
#    else:
#        ecptss = []
#        for numchg in Ms:
#            if numchg == 0:
#                ecpts = np.array([])
#            else:
#                idx = tau_mat[int(numchg-1),: ]
#                idx = np.array(idx[idx<np.inf], dtype=np.int)
#                ecpts = np.array(canpts)[idx] + 1
#            ecptss.append(ecpts)
#        if not diag:
#            return {"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts}
#        else:
#            return {"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts, "idxMat": tau_mat}
            

# effcient dynamic programming to optimize the MBIC, all the change points are based on a python index system
def EGenDy(tXmat, tdXmat, Gclass, alpha, Lmin=None, canpts=None, MaxM=None, Taget="min", diag=False, Ms=None, savepath=None):
    """
    tXmat: array, r x n. n is length of sequence. 
    Gclass: class  arguments (low, up], Ys
    canpts: candidate point set. list or array, not including 0 and n, index not from 0
    MaxM: int, maximal number of change point 
    Lmin: The minimal length between 2 change points
    Ms: the list containing prespecified number of change points.
       When Ms=None, it means using MBIC to determine the number of change points
    savepath: if None, dont save. Otherwise save the intermediate results

    Return:
        change point set not from 0
    """
    def _nloglk(i, j):
        length = j - i
        if length >= Lmin:
            return mbic.NLoglikOne([i, j])  
        else:
            return decon 

    r, n = tXmat.shape
    if Lmin is None:
        Lmin = r
        
    if Taget == "min":
        tagf = np.min
        tagargf = np.argmin
        decon = np.inf
    else:
        tagf = np.max
        tagargf = np.argmax
        decon = -np.inf

    if Ms is not None:
        Ms = sorted(Ms)
    if canpts is None:
        canpts = np.arange(n-1)
    else:
        canpts = np.array(canpts)-1
    M0 = len(canpts) # number of change point in candidate point set
    if MaxM is None:
        MaxM = M0 + 1
    else:
        MaxM = MaxM + 1
    mbicMaxM = MaxM
    if not (Ms is None or len(Ms)==0):
        MaxM = Ms[-1]+1 if Ms[-1]>=(MaxM-1) else MaxM
    canpts_full1 = np.concatenate(([-1], canpts, [n-1]))
    canpts_full2 = canpts_full1[1:]

    Hmat = np.zeros((M0+1, M0+1)) + decon
    mbic = Gclass(tXmat, tdXmat, canpts, alpha)

    # create a matrix 
    for ii in tqdm(range(M0+1), desc="Main Loop"):
        for jj in range(ii, M0+1):
            iidx, jjdx = canpts_full1[ii],  canpts_full2[jj]
            Hmat[ii, jj]  = _nloglk(iidx, jjdx)

    # vector contains results for each number of change point
    U = np.zeros(MaxM) 
    U[0] = Hmat[0, -1]
    D = Hmat[:, -1]
    # contain the location of candidate points  (in python idx)
    Pos = np.zeros((M0+1, MaxM-1)) + decon
    Pos[M0, :] = np.ones(MaxM-1) * M0
    tau_mat = np.zeros((MaxM-1, MaxM-1)) + decon
    for k in range(MaxM-1):
        for j in range(M0): # n = M0 + 1
            dist = Hmat[j, j:-1] + D[(j+1):]
            #print(dist)
            D[j] = np.min(dist)
            Pos[j, 0] = np.argmin(dist) + j + 1
            if k > 0:
                Pos[j, 1:(k+1)] = Pos[int(Pos[j, 0]), 0:k]
        U[k+1] = D[0]
        tau_mat[k, 0:(k+1)] = Pos[0, 0:(k+1)] - 1
    U = U + np.log(n)**alpha * (np.arange(1, MaxM+1))*(r**2 + (r**2-r)/2) + np.log(np.arange(1,MaxM+1))
    if savepath is not None:
        with open(savepath, "wb") as f:
            U0 = U - (np.log(n)**alpha * (np.arange(1, MaxM+1))*(r**2 + (r**2-r)/2) + np.log(np.arange(1,MaxM+1)))
            res = {}
            res["U0"] = U0
            res["U"] = U
            res["Hmat"] = Hmat
            res["tau_mat"] = tau_mat
            pickle.dump(edict(res), f)
    
    mbic_numchg = np.argmin(U[:mbicMaxM])
    if mbic_numchg == 0:
        mbic_ecpts = np.array([])
    else:
        idx = tau_mat[int(mbic_numchg-1),: ]
        idx = np.array(idx[idx<np.inf], dtype=np.int)
        mbic_ecpts = np.array(canpts)[idx] + 1
        
    if Ms is None or len(Ms)==0:
        if not diag:
            return {"U":U, "mbic_ecpts": mbic_ecpts}
        else:
            return {"U":U, "mbic_ecpts": mbic_ecpts, "idxMat": tau_mat}
    else:
        ecptss = []
        for numchg in Ms:
            if numchg == 0:
                ecpts = np.array([])
            else:
                idx = tau_mat[int(numchg-1),: ]
                idx = np.array(idx[idx<np.inf], dtype=np.int)
                ecpts = np.array(canpts)[idx] + 1
            ecptss.append(ecpts)
        if not diag:
            return {"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts}
        else:
            return {"U":U, "ecptss": ecptss, "mbic_ecpts": mbic_ecpts, "idxMat": tau_mat}
            


# dynamic programming to optimize the MBIC, all the change points ase based on a python index system
def GenDy(tXmat, tdXmat, Gclass, alpha, Lmin=None, canpts=None, MaxM=None, Taget="min"):
    """
    tXmat: array, r x n. n is length of sequence. 
    Gclass: class  arguments (low, up], Ys
    canpts: candidate point set. list or array, not including 0 and n
    MaxM: int, maximal number of  change point 
    Lmin: The minimal length between 2 change points
    """
    def _nloglk(i, j):
        length = canpts_full2[j] - canpts_full1[i]
        if length >= Lmin:
            return mbic.NLoglikOne([canpts_full1[i], canpts_full2[j]])  
        else:
            return decon 

    r, n= tXmat.shape
    if Lmin is None:
        Lmin = r
        
    if Taget == "min":
        tagf = np.min
        tagargf = np.argmin
        decon = np.inf
    else:
        tagf = np.max
        tagargf = np.argmax
        decon = -np.inf
    r, n = tXmat.shape
    if canpts is None:
        canpts = np.arange(n-1)
    else:
        canpts = np.array(canpts)-1
    M0 = len(canpts) # number of change point in candidate point set
    if MaxM is None:
        MaxM = M0
    canpts_full1 = np.concatenate(([-1], canpts, [n-1]))
    canpts_full2 = canpts_full1[1:]

    Hmat = np.zeros((MaxM+1, M0+1)) + decon
    Hindmat = np.zeros((MaxM+1, M0+1))  - 1 # contain the location of candidate points  (in python idx)

    for jj in tqdm(range(MaxM+1), desc="Main Loop"):
        mbic = Gclass(tXmat, tdXmat, list(range(jj)), alpha)
        # jj: number of change point
        for ii in tqdm(range(jj, M0+1), desc="Inner Loop", leave=False):
            # ii: idx of candidate point
            if jj == 0:
                Hmat[jj, ii] = _nloglk(jj, ii)
            else:
                vs = [Hmat[jj-1, i-1] + _nloglk(i, ii) for i in range(jj, ii+1)] 
                Hmat[jj, ii] = tagf(vs)
                Hindmat[jj, ii] = tagargf(vs) + jj - 1 
            if ii == M0:
                Hmat[jj, ii] += mbic.getC()

    return {"Mat":Hmat, "idxMat": Hindmat, "n":n}



# function to parse the dynamic programming results
def DyresParse(DyRes, canpts=None):
    res = {}
    if canpts is None:
        canpts = np.arange(1, DyRes["n"])
    else:
        canpts = np.array(canpts)

    Hmat = DyRes["Mat"]
    Hindmat = DyRes["idxMat"]
    MaxM, M0 = Hmat.shape[0] - 1, Hmat.shape[1] - 1
    
    if Hmat[MaxM, 0] > 0 :
        tagargf = np.argmin
    else:
        tagargf = np.argmax

    oneCol = Hmat[:, M0]
    numchg = tagargf(oneCol) 

    res["numchg"] = numchg

    ecpts_can = []
    colidx  = -1
    for k in range(numchg, 0, -1):
        ecpt_can = int(Hindmat[k, colidx])
        ecpts_can.append(ecpt_can)
        colidx = ecpt_can

    ecpts_can = sorted(ecpts_can)
    res["ecpts"] = canpts[ecpts_can]

    return res
        


## MBIC loss functions
class MBIC:
    def __init__(self, Xmat, dXmat, taus, alpha):
        self.Xmat = Xmat
        self.dXmat = dXmat
        self.taus = taus # taus is set of change points based on python index system
        self.alpha = alpha
        self.MBICv = None

    def Afun(self, intv):
        low , up = np.array(intv) + 1
        pdXmat = self.dXmat[:, low:up]
        pXmat = self.Xmat[:, low:up]
        estAT = torch.pinverse(pXmat.matmul(pXmat.T)).matmul(pXmat).matmul(pdXmat.T)
        estA = estAT.T
        return estA

    def samCov(self, intv):
        low, up = np.array(intv) + 1
        estAmat = self.Afun(intv)
        den = up - low
        Tmat = self.dXmat[:, low:up] - estAmat.matmul(self.Xmat[:, low:up])
        return Tmat.matmul(Tmat.T)/den

    def NLoglikOne(self, intv):
        r, _ = self.Xmat.shape
        low , up = np.array(intv) + 1
        SigHat = self.samCov(intv)
        estAmat = self.Afun(intv)
        meanV = torch.zeros(r, dtype=torch.float64)
        # Add 1e-10 to avoid numerical problem
        mnrv = tmnorm(loc=meanV, covariance_matrix=SigHat+1e-10*torch.eye(r))
        Tmat = self.dXmat[:, low:up] - estAmat.matmul(self.Xmat[:, low:up])
        NloglikOneV = - mnrv.log_prob(Tmat.T).sum()
        #NloglikOneV = 0
        #for i in range(Tmat.shape[1]):
        #    NloglikOneV += -mnrv.log_prob(Tmat[:, i])
        return NloglikOneV


    def getC(self):
        r, n = self.Xmat.shape
        M = len(self.taus)
        C = np.log(n)**self.alpha * (M+1) * (r**2 + (r**2-r)/2) + np.log(M+1)
        return C

    def __call__(self):
        r, n = self.Xmat.shape
        M = len(self.taus)
        C = self.getC()
        tausfull = np.concatenate(([0], self.taus, [n]))
        Nloglikv = 0
        for i in range(M+1):
            Nloglikv += self.NLoglikOne([tausfull[i], tausfull[i+1]])
        self.MBICv = Nloglikv + C
        return self.MBICv




# generate simulation data,Ymat
def genYmat(cpts, U, svdA, VT, h, n):
    k = 0
    d, _ = svdA.shape
    Xct = norm.rvs(scale=h/4, size=(d, 1))
    Yct = Xct + norm.rvs(scale=h/4, size=(d, 1))
    dXct = h * U.dot(np.diag(svdA[:, k])).dot(VT).dot(Xct)
    
    Xmat = [Xct]
    Ymat = [Yct]
    dXmat = [dXct]
    
    for j in range(1, n):
        if (j+1) in cpts and (j+1) != n:
            k += 1
        Xct = Xmat[-1] + dXmat[-1]
        Yct = Xct + norm.rvs(scale=h/6, size=(d, 1))
        dXct = h * U.dot(np.diag(svdA[:, k])).dot(VT).dot(Xct)
    
        Xmat.append(Xct)
        Ymat.append(Yct)
        dXmat.append(dXct)
    
    Ymat = np.array(Ymat).squeeze()
    return Ymat.T


# torch verse of rho function
def rhof(M, b0):
    U, S, V = torch.svd(M)
    S[S<=b0] = 0
    return U.matmul(torch.diag(S)).matmul(V.T)



def getsvd(dXmat, Xmat, time, downrate=1):
    h = bw_nrd0_R(time)
    n, T = Xmat.shape
    dXmat, Xmat, time = torch.tensor(dXmat, dtype=torch.float64), torch.tensor(Xmat, dtype=torch.float64), torch.tensor(time, dtype=torch.float64)
    # n x T, n x T, T 
    Kmat = torch.zeros(n, n, dtype=torch.float64)
    for idx, s in enumerate(time[::downrate]):
        #print(idx)
        t_diff = time - s
        kernels = 1/np.sqrt(2*np.pi) * torch.exp(-t_diff**2/2/h**2) # normal_pdf(x/h)
        kernelroot = kernels ** (1/2)
        kerdXmat = kernelroot.unsqueeze(-1) * (dXmat.T) # T x n
        kerXmat = kernelroot.unsqueeze(-1) * (Xmat.T) # T x n
        M = kerXmat.T.matmul(kerXmat)/T
        XY = kerdXmat.T.matmul(kerXmat)/T
        invM = torch.pinverse(M, rcond=h**2*1e-2)
        Kmat = Kmat + XY.matmul(invM)
    return Kmat

# numpy version
def getsvdnp(dXmat, Xmat, time, downrate=1):
    h = bw_nrd0_R(time)
    n, T = Xmat.shape
    Kmat = np.zeros((n, n))
    for idx, s in enumerate(time[::downrate]):
        t_diff = time - s
        kernels = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/h**2) # normal_pdf(x/h)
        kernelroot = kernels ** (1/2)
        kerdXmat = kernelroot[:, np.newaxis] * (dXmat.T) # T x n
        kerXmat = kernelroot[:, np.newaxis] * (Xmat.T) # T x n
        M = kerXmat.T.dot(kerXmat)/T
        XY = kerdXmat.T.dot(kerXmat)/T
        invM = np.linalg.pinv(M, rcond=h**2*1e-2)
        Kmat = Kmat + XY.dot(invM)
    return Kmat
       

def dynProgA(y, x, Kmax, Lmin=1):
    Nr = Kmax - 1
    n = y.shape[1]
    V = np.zeros((n, n)) + np.inf
    for j1 in range(n-Lmin+1):
        for j2 in range(j1+Lmin-1, n):
            nj = j2 - j1 + 1
            Yj = y[:, j1:(j2+1)]
            Xj = x[:, j1:(j2+1)]
            A = Yj.matmul(Xj.T).matmul(pinv(Xj.matmul(Xj.T)))
            resd = (Yj - A.dot(Xj)).T
            sig = resd.T.dot(resd)/nj
            #print(sig, j1, j2)
            hatV = inv(sig)
            V[j1, j2] = - np.log(mnorm.pdf(resd, np.zeros(sig.shape[0]), sig)).sum()

    U = np.zeros(Kmax)
    U[0] = V[0, n-1]
    D = V[:, n-1]
    Pos = np.zeros((n, Nr))
    Pos[n-1, :] = np.ones(Nr) * n
    tau_mat = np.zeros((Nr, Nr))
    for k in range(Nr):
        for j in range(n-1):
            dist = V[j, j:(n-1)] + D[(j+1):n]
            D[j] = dist.min()
            Pos[j, 0] = dist.argmin() + j + 1# may be wrong
            if k > 1:
                Pos[j, 1:k] = Pos[int(Pos[j, 0]), 0:(k-1)]
        U[k+1] = D[0]
        tau_mat[k, 0:(k+1)] = Pos[0, 0:(k+1)] - 1

    out = {}
    out["Test"] = tau_mat 
    out["obj"] = (np.arange(Kmax), U)
    return out


class LRCPT:
    def __init__(self, Ymat, time, canpts=None, downrate=8):
        """
        Ymat: d x n, n length of Time-dimension
        
        """
        self.Ymat = Ymat
        self.time = time
        self.Xmat = None
        self.dXmat = None
        self.tXmat = None
        self.tdXmat = None
        self.canpts = canpts
        self.res = None
        self.estAfull = None
        self.DyRes = None
        self.downrate = downrate
        self.U = None 
        self.V = None 
        self.svs = None
        
    def _GetXs(self):
        d, n = self.Ymat.shape
        Xmatlist = []
        dXmatlist = []
        for i in range(d):
            spres = smooth_spline_R(x=self.time, y=self.Ymat[i, :], lamb=1e-4)
            Xmatlist.append(spres["yhat"])
            dXmatlist.append(spres["ydevhat"])
        self.Xmat = np.array(Xmatlist)
        self.dXmat = np.array(dXmatlist)
        
    def Pre(self):
        if self.Xmat is None:
            self._GetXs()
        estAfull = getsvd(self.dXmat, self.Xmat, self.time, downrate=self.downrate)
        self.estAfull = estAfull
        self.U, self.svs, self.V = torch.svd(estAfull)
    
    def __call__(self, r, Lmin, alpha, MaxM=None, Ms=None, savepath=None):
        if self.U is None:
            self.Pre()
        Xmat, dXmat = torch.tensor(self.Xmat), torch.tensor(self.dXmat)
        tXmat, tdXmat = self.V[:, :r].T.matmul(Xmat), self.U[:, :r].T.matmul(dXmat)
        self.tXmat, self.tdXmat = tXmat, tdXmat
        self.res = EGenDy(tXmat, tdXmat, MBIC, alpha, canpts=self.canpts, Lmin=Lmin, MaxM=MaxM, Ms=Ms, savepath=savepath)
        return self.res
    
    def SamPlot(self, i):
        if self.Xmat is None:
            self._GetXs()
        plt.title(f"Sample {i}th Curve")
        plt.plot(self.Ymat[i, :], "g-", label="noisy curve")
        plt.plot(self.Xmat[i, :], "r--", label="estimate curve")
        plt.legend()
        
    def _GetCanpts(self):
        """
        Obtain the candidate points set
        """
        pass
        

class MBICarr(MBIC):
    def Afun(self, intv):
        low , up = np.array(intv) + 1
        pdXmat = self.dXmat[:, low:up]
        pXmat = self.Xmat[:, low:up]
        estAT = np.linalg.pinv(pXmat.dot(pXmat.T)).dot(pXmat).dot(pdXmat.T)
        estA = estAT.T
        return estA

    def samCov(self, intv):
        low, up = np.array(intv) + 1
        estAmat = self.Afun(intv)
        den = up - low
        Tmat = self.dXmat[:, low:up] - estAmat.dot(self.Xmat[:, low:up])
        return Tmat.dot(Tmat.T)/den


    def NLoglikOne(self, intv):
        r, _ = self.Xmat.shape
        low , up = np.array(intv) + 1
        SigHat = self.samCov(intv)
        estAmat = self.Afun(intv)
        meanV = np.zeros(r)
        # Add 1e-10 to avoid numerical problem
        Tmat = self.dXmat[:, low:up] - estAmat.dot(self.Xmat[:, low:up])
        NloglikOneV = - mnorm.logpdf(Tmat.T, mean=meanV, cov=SigHat+1e-10*np.eye(r)).sum()
        return NloglikOneV


class LRCPTarr(LRCPT):
    def Pre(self):
        if self.Xmat is None:
            self._GetXs()
        estAfull = getsvdnp(self.dXmat, self.Xmat, self.time, downrate=self.downrate)
        self.estAfull = estAfull
        self.U, self.svs, VT = np.linalg.svd(estAfull)
        self.V = VT.T
        
    def __call__(self, r, Lmin, alpha, MaxM=None, Ms=None, savepath=None):
        if self.U is None:
            self.Pre()
        Xmat, dXmat = self.Xmat, self.dXmat
        tXmat, tdXmat = self.V[:, :r].T.dot(Xmat), self.U[:, :r].T.dot(dXmat)
        self.tXmat, self.tdXmat = tXmat, tdXmat
        self.res = EGenDy(tXmat, tdXmat, MBICarr, alpha, canpts=self.canpts, Lmin=Lmin, MaxM=MaxM, Ms=Ms, savepath=savepath)
        return self.res



def Res2Newalpha(res, newalpha, MaxM, r, n, fixed=5):
    U = res.U0 + (np.log(n)**newalpha * (np.arange(1, MaxM+2))*(r**2 + (r**2-r)/2) + np.log(np.arange(1,MaxM+2)))
    numchg = np.argmin(U)
     
    mbicecpts = res.tau_mat[numchg-1, ]
    mbicectps = mbicecpts[mbicecpts<np.inf]
    ecpts = res.tau_mat[fixed-1, ]
    ecpts = ecpts[ecpts<np.inf]
    return {"mbic_ecpts":mbicectps, 
             "ecpts": ecpts}
