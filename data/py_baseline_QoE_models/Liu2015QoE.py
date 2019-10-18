import math
import numpy as np
def Liu2015QoE(VQM, f, T, AMVM, ls):
    alpha = 3.2;
    a = 3.35;
    b = 3.98;
    c = 2.5;
    d = 1800;
    k = 0.02;
    B1 = 73.6;
    B2 = 1608;
    MV_TH = 0.012;
    mu = 0.05;
    C1 = 0.15;
    C2 = 0.82;

    I_ID = min(alpha * ls[0], 100)
    D_ST = sum(ls[1:])
    N_ST = len(ls[1:])

    # I_ID = 0
    # D_ST = sum(ls[0:])
    # N_ST = len(ls[0:])

    if (AMVM < MV_TH):
        # print("AMV")
        I_ST = a * D_ST + b * N_ST - c * math.sqrt(D_ST * N_ST) + d * AMVM
    else:
        I_ST = a * D_ST + b * N_ST - c * math.sqrt(D_ST * N_ST) + d * MV_TH

    VQM = np.asarray(VQM)
    numSeg = int(len(VQM) / (f * T));
    VQM = np.reshape(VQM, (int(f * T), int(numSeg)));
    VQM = np.mean(VQM, axis=0);
    P2 = 0;
    D = np.zeros((numSeg, 1));
    for iii in range(1,numSeg):
        if (abs(VQM[iii] - VQM[iii-1]) < mu):
            D[iii] = D[iii-1] + 1;
        else:
            D[iii] = 0;
    #
        if ((VQM[iii] - VQM[iii-1]) > 0):
            P2 = P2 + (VQM[iii] - VQM[iii-1])**2;
    P2 = P2 / numSeg;
    #
    P1 = 0;
    for iii in range(numSeg):
        P1 = P1 + VQM[iii] * math.exp(k * T * D[iii]);
    # end
    P1 = P1 / numSeg;
    #
    I_LV = B1 * P1 + B2 * P2;
    #
    R = 100 - I_ID - I_ST - I_LV + C1 * I_ID * math.sqrt(I_ST + I_LV) + C2 * math.sqrt(I_ST * I_LV);
    Q = 1 + 0.035 * R + (7e-6) * R * (R - 60) * (100 - R);
    return Q
