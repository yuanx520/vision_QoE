function Q = Mok2011QoE( T_init, T_rebuf, f_rebuf )
    % T_init: duration of initial buffering
    % T_rebuf: mean duration of stalling
    % f_rebuf: frequency of global stalling in Second-1
    % Author: Zhengfang Duanmu
    if (T_init == 0)
        l_init = 0;
    elseif (T_init > 0 && T_init <= 1)
        l_init = 1;
    elseif (T_init >= 1 && T_init <= 5)
        l_init = 2;
    elseif (T_init > 5)
        l_init = 3;
    else
        error('T_init cannot be negative');
    end
    
    if (f_rebuf == 0)
        l_fr = 0;
    elseif (f_rebuf > 0 && f_rebuf <= 0.02)
        l_fr = 1;
    elseif (f_rebuf > 0.02 && f_rebuf <= 0.15)
        l_fr = 2;
    elseif (f_rebuf > 0.15)
        l_fr = 3;
    else
        error('f_rebuf cannot be negative');
    end
    
    if (T_rebuf == 0)
        l_rebuf = 0;
    elseif (T_rebuf > 0 && T_rebuf <= 5)
        l_rebuf = 1;
    elseif (T_rebuf > 5 && T_rebuf <= 10)
        l_rebuf = 2;
    elseif (T_rebuf > 10)
        l_rebuf = 3;
    else
        error('T_rebuf cannot be negative');
    end
    Q = 4.23 - 0.0672*l_init - 0.742*l_fr - 0.106*l_rebuf;
end

