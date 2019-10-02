function Q = Liu2015QoE( VQM, f, T, AMVM, ms, ls )
%   This is an implemenatation of the algorithm described in 
%	User Experience Modeling for DASH Video
%   VQM: frame VQM score
%   AMVM: average motion vector magnitude
%   f: frame rate
%   T: segment duration
%   ms: stalling frames
%   ls: duration of each stalling event
%   Author: Zhengfang Duanmu
    
    % model parameters
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
    
    % impairments of initial buffering and stalling
    if ~isempty(ms)
        if (ms(1) == 0)
            I_ID = min(alpha * ls(1), 100);
            D_ST = sum(ls(2:end));
            N_ST = length(ls(2:end));
        else
            I_ID = 0;
            D_ST = sum(ls);
            N_ST = length(ls);
        end
        
        if (AMVM < MV_TH)
            I_ST = a * D_ST + b * N_ST - c * sqrt(D_ST * N_ST) + d * AMVM;
        else
            I_ST = a * D_ST + b * N_ST - c * sqrt (D_ST * N_ST) + d * MV_TH;
        end
    else
        I_ID = 0;
        I_ST = 0;
    end
    
    % video quality
    numSeg = length(VQM) / (f * T);
    VQM = reshape(VQM, [f * T, numSeg]);
    VQM = mean(VQM, 1);
    
    P2 = 0;
    D = zeros(numSeg, 1);
    for iii = 2:numSeg
        if (abs(VQM(iii) - VQM(iii-1)) < mu)
            D(iii) = D(iii-1) + 1;
        else
            D(iii) = 0;
        end
        
        if ((VQM(iii) - VQM(iii-1)) > 0)
            P2 = P2 + (VQM(iii) - VQM(iii-1))^2;
        end
    end
    P2 = P2 / numSeg;
    
    P1 = 0;
    for iii = 1:numSeg
        P1 = P1 + VQM(iii) * exp(k * T * D(iii));
    end
    P1 = P1 / numSeg;
    
    I_LV = B1 * P1 + B2 * P2;
    
    R = 100 - I_ID - I_ST - I_LV + C1 * I_ID * sqrt(I_ST + I_LV) + C2 * sqrt(I_ST * I_LV);
    Q = 1 + 0.035 * R + (7e-6) * R * (R - 60) * (100 - R);
end