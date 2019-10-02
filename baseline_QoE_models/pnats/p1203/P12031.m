function O22 = P12031(bitrate, disRes, codRes, fps, handheld)
    % This is an implementation of ITU P.1203.1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % model parameteres
    % 8.1.1 parameters
    q1 = 4.66;
    q2 = -0.07;
    q3 = 4.06;
    
    % 8.1.2 parameters
    u1 = 72.61;
    u2 = 0.32;
    
    % 8.1.3 parameters
    t1 = 30.98;
    t2 = 1.29;
    t3 = 64.65;
    
    % device parameters
    htv1 = -0.60293;
    htv2 = 2.12382;
    htv3 = -0.36936;
    htv4 = 0.03409;
    
    %% model implementation
    O22 = zeros(length(bitrate), 1);
    % 8.1.1 Quantization degradation
    quant = mode0(bitrate, codRes, fps);
    MOSq = q1 + q2 .* exp(q3 .* quant);
    MOSq = max(min(MOSq, 5), 1);
    
    Dq = 100 - RfromMOS(MOSq);
    Dq = max(min(Dq, 100), 0);
    
    % 8.1.2 Upscaling degradation
    scaleFactor = max(disRes./codRes, 1);
    Du = u1 .* log10(u2 .* (scaleFactor - 1) + 1);
    Du = max(min(Du, 100), 0);
    
    % 8.1.3 Temporal degradation
    Dt1 = 100 .* (t1 - t2 .* fps) ./ (t3 + fps);
    Dt2 = Dq .* (t1 - t2 .* fps) ./ (t3 + fps);
    Dt3 = Du .* (t1 - t2 .* fps) ./ (t3 + fps);
    
    Dt = zeros(length(Dq), 1);
    Dt(fps < 24) = Dt1(fps < 24) - Dt2(fps < 24) - Dt3(fps < 24);
    Dt = max(min(Dt, 100), 0);
    
    % 8.1.4 Integration
    D = max(min(Dq + Du + Dt, 100), 0);
    
    Q = 100 - D;
    for iii = 1:length(Dq)
        if Du(iii) == 0 && Dt(iii) == 0
            O22(iii) = MOSq(iii);
        else
            O22(iii) = MOSfromR(Q(iii));
        end
    end
    
    if handheld
        MOSqh = htv1 + htv2 .* O22 + htv3 .* O22^2 + htv4 .* O22^3;
        O22 = max(min(MOSqh, 5), 1);
    end
end