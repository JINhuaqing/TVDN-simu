import numpy as np
import rpy2.robjects as robj
from easydict import EasyDict as edict
from numpy.linalg import inv, svd
from scipy.signal import detrend

timeLims = edict()
timeLims.st02 = [35, 95]
timeLims.st03 = [20, 80]

# load the gt for MEG--Eye data
def txt2Time(txtF):   
    with open(txtF, "r") as f:
        data = f.readlines() 
    data = data[1:]
    data = [i.strip().split("(") for i in data]
    data = [float(i[0]) for i in data if len(i)>1]
    return data

# Time to change points
def time2pts(ts, lims, Up=7200):
    ts = np.array(ts)
    timeC = 60
    ts = ts[ts>=lims[0]]
    ts = ts[ts<=lims[1]]
    ts = ts - lims[0]
    cpts = ts*Up/timeC
    cpts = cpts.astype(np.int)
    
    res = edict()
    res.ts = ts
    res.cpts = cpts
    return res


def py2Rmat(Mat):
    Mat = np.array(Mat)
    nrow, _ = Mat.shape
    rVec = robj.FloatVector(Mat.ravel())
    rMat = robj.r.matrix(rVec, nrow=nrow, byrow=True)
    return rMat

def py2Rvec(vec):
    vec = np.array(vec)
    vec = vec.astype(np.float)
    return robj.FloatVector(vec)


# Obtain the weighted U from the detection obj
def obtainAbswU(DetObj):
    eigVecs = DetObj.midRes.eigVecs[:, :DetObj.paras.r]
    kpidxs = np.concatenate([[0], DetObj.ecpts]).astype(np.int)
    eigVals = DetObj.RecResCur.LamMs[:, kpidxs]
    wU = eigVecs.dot(eigVals)
    return np.abs(wU)


def minmax(x):
    num = x - np.min(x)
    den = np.max(x) - np.min(x)
    return num/den



# Reconstruct the curve for DMD method for each segment
def SegPredDMD(Ymat, low, up, rank=0, initX=None):
    X = Ymat[:, low:(up-1)]
    Xprime = Ymat[:, (low+1):up]
    d, n1 = X.shape
    Xu, Xs, Xvt = np.linalg.svd(X, False)
    if rank <= 0:
        rank = np.sum(np.cumsum(Xs)/np.sum(Xs) <= 0.8) + 1
    Xur = Xu[:, :rank]
    Xvtr = Xvt[:rank, :]
    Xsr = Xs[:rank]
    Ahat = Xprime.dot(Xvtr.T).dot(np.diag(1/Xsr)).dot(Xur.T)
    Xpred = np.zeros((d, n1+1))
    if initX is None:
        initX = X[:, 0]
    Xpred[:, 0] = initX
    for i in range(1, n1+1):
        Xpred[:, i] = Ahat.dot(Xpred[:, i-1])
    return Xpred


# Reconstruct the curve for DMD method for each segment 
# when using half of the data
def SegPredDMDHalf(Ymat, low, up, rank=0, initX=None):
    upNew = low + int((up-low)/2)
    X = Ymat[:, low:(upNew-1)]
    Xprime = Ymat[:, (low+1):upNew]
    n1 = int(up - low)
    d, _ = X.shape
    Xu, Xs, Xvt = np.linalg.svd(X, False)
    if rank <= 0:
        rank = np.sum(np.cumsum(Xs)/np.sum(Xs) <= 0.8) + 1
    Xur = Xu[:, :rank]
    Xvtr = Xvt[:rank, :]
    Xsr = Xs[:rank]
    Ahat = Xprime.dot(Xvtr.T).dot(np.diag(1/Xsr)).dot(Xur.T)
    Xpred = np.zeros((d, n1))
    if initX is None:
        initX = X[:, 0]
    Xpred[:, 0] = initX
    for i in range(1, n1):
        if i < int((up-low)/2):
            Xpred[:, i] = Xprime[:, i-1]
        else:
            Xpred[:, i] = Ahat.dot(Xpred[:, i-1])
    return Xpred

# Reconstruct the curve for DMD method given the estimated change points
def PredDMD(Ymat, ecpts, rank=0, SegPredDMD=SegPredDMDHalf):
    d, n = Ymat.shape
    ecpts1 = np.concatenate([[0], ecpts])
    ecpts2 = np.concatenate([ecpts, [n]])
    Xpred = np.zeros((d, n))
    for low, up in zip(ecpts1, ecpts2):
        if low == 0:
            initX = None
        Xpred[:, low:up] = SegPredDMD(Ymat, low, up, rank=rank, initX=initX)
        initX = None
    return Xpred


# Reconstruct Xmat from results segment-wisely when using half data
def ReconXmatSWHalf(ecpts, ndXmat, nXmat, kpidxs, eigVecs, Ymat, tStep, r, is_full=False):
    """
    Input: 
        ecpts: Estimated change points, 
        ndXmat: a rAct x n matrix
        nXmat: a rAct x n matrix
        kpidxs: The intermedian output when calculating ndXmat, nXmat
        eigVecs: The matrix of eigen vectors of A matrix, d x d
        Ymat: The matrix to construct, d x n 
        tStep: The time step
        r: The rank setted beforehand, in most cases, r=rAct. If we have non-complex singular values, r < rAct
        if_full: Where outputing full info or not

    Return:
        Estimated Xmat, d x n
    """
    #print(f"The class calls the new reconstruction function, ReconXmatNew")
    rAct, n = ndXmat.shape
    d, _ = Ymat.shape
    ecptsfull = np.concatenate(([0], ecpts, [n])) - 1
    ecptsfull = ecptsfull.astype(np.int)
    numchgfull = len(ecptsfull)

    ResegS = np.zeros((numchgfull-1, r), dtype=np.complex)
    for  itr in range(numchgfull-1):
        lower = ecptsfull[itr] + 1
        upper = ecptsfull[itr+1] + 1
        Ycur = ndXmat[:, lower:upper]
        Xcur = nXmat[:, lower:upper]
        _, ncol = Xcur.shape
        hncol = int(ncol/2)
        lams = np.zeros(r, dtype=np.complex) + np.inf
        for j in range(int(rAct/2)):
            tY = Ycur[(2*j):(2*j+2), :hncol]
            tX = Xcur[(2*j):(2*j+2), :hncol]
            corY = tY.dot(tX.T)
            corX = np.trace(tX.dot(tX.T))
            a = np.trace(corY)/corX
            b = (corY[1, 0] - corY[0, 1])/corX
            lams[kpidxs[j]] = a + b*1j
        tmpIdx = np.where(lams==np.inf)[0]
        lams[tmpIdx] = np.conjugate(lams[tmpIdx-1])
        ResegS[itr, :] = lams
    
    LamMs = np.zeros((r, n), dtype=np.complex)
    LamMs[:, 0] = ResegS[0, :]
    for itr in range(1, numchgfull):
        lower = ecptsfull[itr-1] + 1
        upper = ecptsfull[itr] + 1
        LamMs[:, lower:upper] = ResegS[itr-1, ].reshape(-1, 1)
    
    ecptsf = np.concatenate([[0], ecpts, [n]])
    fixIdxs = []
    for i, j in zip(ecptsf[:-1], ecptsf[1:]):
        num = int((j-i)/2)
        fixIdxs += list(range(int(i), int(i)+num))
    EstXmat = np.zeros((d, n), dtype=np.complex)
    EstXmat[:, 0] = Ymat[:, 0]
    invEigVecsr = inv(eigVecs)[:r, :]
    eigVecsr = eigVecs[:, :r]
    for i in range(1, n):
        if i in fixIdxs:
            EstXmat[:, i] = Ymat[:, i]
        else:
            mTerm = np.diag(LamMs[:, i])
            rTerm = invEigVecsr.dot(EstXmat[:, i-1])
            EstXmat[:, i] = eigVecsr.dot(mTerm).dot(rTerm) * tStep + EstXmat[:,i-1]
        
    if is_full:
        ReDict = edict()
        ReDict.EstXmatReal = detrend(EstXmat.real)
        ReDict.EstXmatRealOrg = EstXmat.real
        ReDict.EstXmatImag = EstXmat.imag
        ReDict.LamMs = LamMs
        return ReDict
    else:
        return detrend(EstXmat.real)
    
 # Reconstruct Xmat from results segment-wisely
def ReconXmatSW(ecpts, ndXmat, nXmat, kpidxs, eigVecs, Ymat, tStep, r, is_full=False):
    """
    Input: 
        ecpts: Estimated change points, 
        ndXmat: a rAct x n matrix
        nXmat: a rAct x n matrix
        kpidxs: The intermedian output when calculating ndXmat, nXmat
        eigVecs: The matrix of eigen vectors of A matrix, d x d
        Ymat: The matrix to construct, d x n 
        tStep: The time step
        r: The rank setted beforehand, in most cases, r=rAct. If we have non-complex singular values, r < rAct
        if_full: Where outputing full info or not

    Return:
        Estimated Xmat, d x n
    """
    #print(f"The class calls the new reconstruction function, ReconXmatNew")
    rAct, n = ndXmat.shape
    d, _ = Ymat.shape
    ecptsfull = np.concatenate(([0], ecpts, [n])) - 1
    ecptsfull = ecptsfull.astype(np.int)
    numchgfull = len(ecptsfull)

    ResegS = np.zeros((numchgfull-1, r), dtype=np.complex)
    for  itr in range(numchgfull-1):
        lower = ecptsfull[itr] + 1
        upper = ecptsfull[itr+1] + 1
        Ycur = ndXmat[:, lower:upper]
        Xcur = nXmat[:, lower:upper]
        lams = np.zeros(r, dtype=np.complex) + np.inf
        for j in range(int(rAct/2)):
            tY = Ycur[(2*j):(2*j+2), :]
            tX = Xcur[(2*j):(2*j+2), :]
            corY = tY.dot(tX.T)
            corX = np.trace(tX.dot(tX.T))
            a = np.trace(corY)/corX
            b = (corY[1, 0] - corY[0, 1])/corX
            lams[kpidxs[j]] = a + b*1j
        tmpIdx = np.where(lams==np.inf)[0]
        lams[tmpIdx] = np.conjugate(lams[tmpIdx-1])
        ResegS[itr, :] = lams
    
    LamMs = np.zeros((r, n), dtype=np.complex)
    LamMs[:, 0] = ResegS[0, :]
    for itr in range(1, numchgfull):
        lower = ecptsfull[itr-1] + 1
        upper = ecptsfull[itr] + 1
        LamMs[:, lower:upper] = ResegS[itr-1, ].reshape(-1, 1)
    
    EstXmat = np.zeros((d, n), dtype=np.complex)
    EstXmat[:, 0] = Ymat[:, 0]
    invEigVecsr = inv(eigVecs)[:r, :]
    eigVecsr = eigVecs[:, :r]
    for i in range(1, n):
        if i in ecpts:
            EstXmat[:, i] = Ymat[:, i]
        else:
            mTerm = np.diag(LamMs[:, i])
            rTerm = invEigVecsr.dot(EstXmat[:, i-1])
            EstXmat[:, i] = eigVecsr.dot(mTerm).dot(rTerm) * tStep + EstXmat[:,i-1]
        
    if is_full:
        ReDict = edict()
        ReDict.EstXmatReal = detrend(EstXmat.real)
        ReDict.EstXmatRealOrg = EstXmat.real
        ReDict.EstXmatImag = EstXmat.imag
        ReDict.LamMs = LamMs
        return ReDict
    else:
        return detrend(EstXmat.real)