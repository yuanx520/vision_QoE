function Q = P12033( O21, O22, ms, ls, forest )
%   This is an implemenatation of the algorithm described in 
%	P.1203.3
%   Inputs:
% ====================================================
%   I.14
%   numStalls: number of stalling events
%   totalBuffLen: total length of stalling events
%   avgBuffLen: average interval between stalling events
%
% ====================================================
%   O21 and O22: one score per second
%   ms: stalling frames
%   ls: duration of each stalling event
%   forest: random forest as a cell array
%   Author: Zhengfang Duanmu
    
    %% model parameters
    % Coefficient sets for w_buffi
    C_ref7 = 0.48412879;
    C_ref8 = 10;
    
    % Coefficient sets for negBias
    C1 = 1.87403625;
    C2 = 7.85416481;
    C23 = 0.01853820;
    
    % Coefficient sets for O.34
    av1 = -0.00069084;
    av2 = 0.15374283;
    av3 = 0.97153861;
    av4 = 0.02461776;
    
    % Coefficient sets for O.35
    t1 = 0.00666620027943848;
    t2 = 0.0000404018840273729;
    t3 = 0.156497800436237;
    t4 = 0.143179744942738;
    t5 = 0.0238641564518876;
    c1 = 0.67756080;
    c2 = -8.05533303;
    c3 = 0.17332553;
    c4 = -0.01035647;
    
    % Coefficient sets for O.46
    S1 = 9.35158684;
    S2 = 0.91890815;
    S3 = 11.0567558;
    
%     load('randomForest.mat');
    
    %% total duration of the session
    T = length(O22);
    t = 1:T;
    
    %% 8.1.1 Parameters related to stalling
    numStalls = 0;
    totalBuffLen = 0;
    avgBuffLen = 0;
    if ~isempty(ms)
        if (length(ms) > 1)
        	avgBuffLen = mean(ms(2:end) - ms(1:end-1));
        end
        w_buff = C_ref7 + (1 - C_ref7) .* exp(ms .* (log(0.5) / (-C_ref8)));
        totalBuffLen = sum(ls .* w_buff);
        numStalls = length(ms);
    end

    % O34
    O34 = max(min(av1 + av2 .* O21 + av3 .* O22 + av4 .* O21 .* O22, 5), 1);
    
    %% 8.1.2 Parameters related to audiovisual quality
    % negativeBias
    wdiff = C1 + (1 - C1) .* exp(-(T-t) .* (log(0.5) / (-C2)));
    O34_diff = O34 .* wdiff(:);
    negPerc = prctile(O34_diff, 10);
    negBias = max(0, -negPerc) * C23;
    % vidQualSpread
    vidQualSpread = max(O22) - min(O22);
    % vidQualChangeRate
    vidQualChangeRate = sum(abs(O22(2:end) - O22(1:end-1)) > 0.2) / T;
    % qDirChangesTot
    maFilter = (1/5)*ones(5, 1);
    O22pad = padarray(O22(:), [2 0], 'replicate', 'both');
    O22MA = conv(O22pad, maFilter, 'valid');
    QC = [];
    for ppp = 1:3:(length(O22MA) - 3)
        qqq = ppp + 3;
        diffMA = O22MA(ppp) - O22MA(qqq);
        if (diffMA > 0.2)
            QC = [QC, 1]; %#ok
        elseif (diffMA > -0.2 && diffMA <0.2)
            QC = [QC, 0]; %#ok
        else
            QC = [QC, -1]; %#ok
        end
    end
    QCnoZero = QC(QC ~= 0);
    qDirChangesTot = 0;
    if ~isempty(QCnoZero)
        qDirChangesTot = qDirChangesTot + 1;
    end
    for ppp = 2:length(QCnoZero)
        if (QCnoZero(ppp) ~= QCnoZero(ppp-1))
            qDirChangesTot = qDirChangesTot + 1;
        end
    end
    % qDirChangesLongest
    qc_len = [];
    distances = [];
    for index = 1:length(QC)
        if (QC(index) ~= 0)
            if ~isempty(qc_len)
                if qc_len(size(qc_len, 1), 2) ~= QC(index)
                    qc_len = [qc_len; index, QC(index)]; %#ok
                end
            else
                qc_len = [index, QC(index)];
            end
        end
    end
    if ~isempty(qc_len)
        qc_len = [1, 0; qc_len];
        qc_len = [qc_len; length(QC)+1, 0]; %#ok
        
        for ppp = 2:length(QC)
            distances = [distances, QC(ppp) - QC(ppp-1)]; %#ok
        end
        qDirChangesLongest = max(distances) * 3;
    else
        qDirChangesLongest = T;
    end
    
    %% 8.1.3 Parameters related to machine learning module
    mediaLength = min(length(O21), length(O22));  % 13
    reBuffCount = 0;
    initBuffDur = 0;
    stallDurWoIB = 0;
    if ~isempty(ms)
        msWoIB = ms(ms ~= 0);
        reBuffCount = length(msWoIB);
        stallDurWoIB = sum(ls);
        
        if ms(1) == 0
            initBuffDur = ls(1);
            stallDurWoIB = stallDurWoIB - initBuffDur;
        end
    end
    stallDur = 1/3 * initBuffDur + stallDurWoIB; % 1
    reBuffFreq = reBuffCount / mediaLength; % 2
    stallRatio = stallDur / mediaLength; % 3 
    if ~isempty(ms)
        timeLastRebuffToEnd = mediaLength - ms(end); % 4
    else
        timeLastRebuffToEnd = 0;
    end
    
    averagePvScoreOne = sum(O22(1:(round(length(O22)/3)))) ...
        / length( 1:(round(length(O22)/3)) ); % 5
    averagePvScoreTwo = sum(O22((round(length(O22)/3)+1):(round(2*length(O22)/3)))) ...
        / length( (round(length(O22)/3)+1):(round(2*length(O22)/3)) ); % 6
    averagePvScoreThree = sum(O22((round(2*length(O22)/3)+1):end)) ...
        / length( (round(2*length(O22)/3)+1):length(O22) ); % 7
    onePercentilePvScore = prctile(O22, 1); % 8
    fivePercentilePvScore = prctile(O22, 5); % 9
    tenPercentilePvScore = prctile(O22, 10); % 10
    averagePaScoreOne = sum(O21(1:(round(length(O21)/2)))) ...
        / length( 1:(round(length(O21)/2)) ); % 11
    averagePaScoreTwo = sum(O21((round(length(O21)/2)+1):end)) ...
        / length( (round(length(O21)/2)+1):length(O21) ); % 12
    
    %%  O35 and its dependencies 8-2 -- 8-11
    % O35baseline
    w1 = t1 + t2 .* exp(t ./ (T/t3));
    w2 = t4 - t5 .* O34;
    O35baseline = sum(w1(:) .* w2 .* O34) / sum(w1(:) .* w2);
    % negBias has been computed

    % oscComp
    qDiff = max(0.1 + log10(vidQualSpread + 0.01), 0);
    oscTest = (qDirChangesTot / T < 0.25) & (qDirChangesLongest < 30);
    if oscTest
        % should we remove min in Eq. 8-8? cauz it should be a penalty term
        oscComp = qDiff * exp(min(c1 * qDirChangesTot + c2, 1.5));
    else
        oscComp = 0;
    end
    % adaptComp
    adaptTest = (qDirChangesTot / T < 0.25);
    if adaptTest
        % should we remove min in Eq. 8-9? cauz it should be a penalty term
        adaptComp = c3 * vidQualSpread * vidQualChangeRate + c4;
    else
        adaptComp = 0;
    end
    % O35
    O35 = O35baseline - negBias - oscComp - adaptComp;
    % Eq. 8-13
    SI = exp(-numStalls / S1) * exp(-(totalBuffLen / T) / S2) * exp(-(avgBuffLen / T) / S3);
    
    input_feature = [reBuffCount; stallDur; reBuffFreq; stallRatio; ...
        timeLastRebuffToEnd; averagePvScoreOne; averagePvScoreTwo; ...
        averagePvScoreThree; onePercentilePvScore; fivePercentilePvScore; ...
        tenPercentilePvScore; averagePaScoreOne; averagePaScoreTwo; mediaLength];
    RF_prediction = getRfPred(input_feature, forest);
    % Eq. 8-12
    O46 = 0.75 * (1 + (O35 - 1) * SI) + 0.25 * RF_prediction;
    % Eq. 8-14
    Q = 0.02833052 + 0.98117059 * O46;
end