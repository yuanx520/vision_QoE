function Q = Bentaleb2016QoE( ssimplus, nStall, tIB )
%   This is an simplified implemenatation of the algorithm in ICME2014
%   bitrates: array of bitrate, with each entry represents the bitrate of
%   one segment
%   Tstall: total rebuffer time in second
%   Ts: startup delay in second

    % model parameters
    alpha = 0.25;
    beta = 0.25;
    gamma = 0.25;
    delta = 0.25;
    
    % model parameters
    % TQ: overall video quality
    TQ = mean(ssimplus);
    % SQ: overall quality variations
    SQ = sum(abs(ssimplus(2:end) - ssimplus(1:end-1))) / (length(ssimplus) - 1);
    
    Q = alpha * TQ - beta * SQ - gamma * nStall - delta * tIB;
end