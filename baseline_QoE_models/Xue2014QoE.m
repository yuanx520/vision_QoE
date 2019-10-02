function [Q, y] = Xue2014QoE( QP, bitcount, f, ms, ls )
%   This is an simplified implemenatation of the algorithm in ICME2014
%   QP: compression quantization parameter of each frame
%   bitcount: bit count of each frame
%   f: frame rate
%   ms: stalling frames
%   ls: duration of each stalling event
%   Author: Zhengfang Duanmu

    % optional input parameters
    c = 0.05;
    % epsilon = 0.05;
    
    % model parameters
    a = (-1)/51;
    b = 1;
    gamma = 0.71;
    W_init = 0.5;
    QP_init = 27;
    
    % initialization
    if size(QP,1)<size(QP,2)
        QP = QP';
    end
    
    if size(bitcount,1)<size(bitcount,2)
        bitcount = bitcount';
    end
    
    q = a.*QP+b;
    x = zeros(length(QP)+sum(ls*f), 1);
    y = zeros(length(QP)+sum(ls*f), 1);
    x_0 = -W_init*(a*QP_init+b);
    % offline process, so maximum bitcount estimation is not necessary
    r_max = max(bitcount);
    
    fs = round(ls*f);
    cls = cumsum(fs); % cummulative ls
    
    if (isempty(ms))
        x = q;
    else
        if ms(1) == 0
            x(1:(ms(1)+cls(1))) = padarray(q(1:(ms(1))),[fs(1) 0],x_0,'pre');
        end
        
        for iii = 2:length(ms)
            W_stall = (log(bitcount(ms(iii)-1))+c)/(log(r_max)+c);
            x_stall = -W_stall*q(ms(iii)-1);
            x((ms(iii-1)+cls(iii-1)+1):(ms(iii)+cls(iii))) = padarray(q((ms(iii-1)+1:ms(iii))),[fs(iii) 0],x_stall,'post');
        end
        
        x((ms(end)+cls(end)+1):end) = q(ms(end)+1:end);
    end
    
    y(1) = x(1);
    for i = 2:length(y)
        y(i) = gamma*y(i-1)+(1-gamma)*x(i);
    end
    Q = mean(y);
end

