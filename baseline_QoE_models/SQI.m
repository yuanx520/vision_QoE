function [Q_m, Q, dQs_m] = SQI( Qr, r, ms, ls, varargin )
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

    p = inputParser;
    % dQs parameters
    % For SSIM and MSSSIM:  'To_init' = 40;  'T1_init' = 1, 'To' = 55; 'T1' = 75;
    % For PSNR:  'To_init', 15; 'T1_init' = 2; 'To' = 10, 'T1' = 15;
    default_To_init = 2;
    default_To = 1;
    default_T1_init = 0.5;
    default_T1 = 1.2;
    default_P0 = 80;
    
    addRequired(p,'Qr');
    addRequired(p,'r');
    addRequired(p,'ms');
    addRequired(p,'ls');
    addParameter(p, 'To_init', default_To_init, @isnumeric);
    addParameter(p, 'To', default_To, @isnumeric);
    addParameter(p, 'T1_init', default_T1_init, @isnumeric);
    addParameter(p, 'T1', default_T1, @isnumeric);
    addParameter(p, 'P0', default_P0, @isnumeric);
    
    parse(p, Qr, r, ms, ls, varargin{:});
    
    % dQs parameters
    To_init = p.Results.To_init;
    To = p.Results.To;
    T1_init = p.Results.T1_init;
    T1 = p.Results.T1;

    P0 = p.Results.P0;

    %% input validation
    if (length(ms) ~= length(ls))
        error('The size of ms and ls should be identical!');
    end

    %% buffer initialization
    Ns = length(ms);
    Qh = zeros(length(Qr) + round(r * sum(ls)), 1);
    dQs_m = zeros(length(Qr) + round(r * sum(ls)), Ns);
    
    %% computing Qr_tail
    fs = round(ls*r);
    cls = cumsum(fs); % cummulative ls
    ns = ms + [0, cls(1:end-1)];
    
    if Ns == 0
        Qh = Qr;
    else
        if ms(1) == 0
            Qh(1:(ms(1)+cls(1))) = padarray(Qr(1:(ms(1))),[fs(1) 0], P0, 'pre');
        else
            Qh(1:(ms(1)+cls(1))) = padarray(Qr(1:(ms(1))),[fs(1) 0],'replicate','post');
        end
        
        for i = 2:Ns
            Qh((ms(i-1)+cls(i-1)+1):(ms(i)+cls(i))) = padarray(Qr((ms(i-1)+1:ms(i))),[fs(i) 0],'replicate','post');
        end
        Qh((ms(end)+cls(end)+1):end) = Qr(ms(end)+1:end);
    end
    
    Qt = Qh;
    
    %% computing dQs
    for i = 1:Ns
        % initial buffering without the first frame
        if ns(i) == 0
            dQs_m(1:(1+fs(i)),1) = P0 * ...
                ( (-1) + exp( - ( (0:fs(i))/(r*To_init) )) );
            dQs_m((ns(i)+fs(i)+1):end,1) = P0 * ...
                ( (-1) + exp( - ( fs(i)/(r*To_init) )) ) * ...
                exp( - ( (1:(length(Qh)-(ns(i)+fs(i))) ) / (r*T1_init) ) );
        % initial buffering with the first frame showing
        elseif ns(i) == 1
            dQs_m(ns(i):(ns(i)+fs(i)),1) = Qh(ns(i)) * ...
                ( (-1) + exp( - ( (0:fs(i))/(r*To_init) )) );
            dQs_m((ns(i)+fs(i)+1):end,1) = Qh(ns(i)) * ...
                ( (-1) + exp( - ( fs(i)/(r*To_init) )) ) * ...
                exp( - ( (1:(length(Qh)-(ns(i)+fs(i))) ) / (r*T1_init) ) );
        else
            dQs_m(ns(i):(ns(i)+fs(i)),i) = Qh(ns(i)) * ...
                ( (-1) + exp( - ( (0:fs(i))/(r*To) )) );
            dQs_m((ns(i)+fs(i)+1):end,i) = Qh(ns(i)) * ...
                ( (-1) + exp( - ( (fs(i))/(r*To) )) ) * ...
                exp( - ( (1:(length(Qh)-(ns(i)+fs(i))) ) / (r*T1) ) );
        end
    end
    dQs = sum(dQs_m,2);
    
    %% computing Q
    Q = Qt + dQs;
    Q_m = mean(Q);
end

