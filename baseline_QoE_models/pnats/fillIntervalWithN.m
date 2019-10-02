function [Q_filled] = fillIntervalWithN( Qr, r, ms, ls, N )
% This function predicts QoE score from known per-frame raw video quality
% and position of stalling
% author: Zhengfang Duanmu
% Input: 
%   1. Qr: raw predicted quality scores of each frame
%   2. r: frame rate
%   3. ms: array of starting frames of the stalling events
%   4. ls: array of duration of the stalling events in seconds
%
% Q_m = mean(Q)
% Q = Qt + dQs
% dQs: quality variation due to stalling events
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% input validation
    if (length(ms) ~= length(ls))
        error('The size of ms and ls should be identical!');
    end

    %% buffer initialization
    Ns = length(ms);
    Q_filled = zeros(length(Qr) + round(r * sum(ls)), 1);
    
    %% computing Qr_tail
    fs = round(ls*r);
    cls = cumsum(fs); % cummulative ls
    
    if Ns == 0
        Q_filled = Qr;
    else
        if ms(1) == 0
            Q_filled(1:(ms(1)+cls(1))) = padarray(Qr(1:(ms(1))),[fs(1) 0], N, 'pre');
        else
            Q_filled(1:(ms(1)+cls(1))) = padarray(Qr(1:(ms(1))),[fs(1) 0], N, 'post');
        end
        
        for i = 2:Ns
            Q_filled((ms(i-1)+cls(i-1)+1):(ms(i)+cls(i))) = padarray(Qr((ms(i-1)+1:ms(i))),[fs(i) 0], N, 'post');
        end
        Q_filled((ms(end)+cls(end)+1):end) = Qr(ms(end)+1:end);
    end
end