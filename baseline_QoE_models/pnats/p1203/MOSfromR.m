function MOS = MOSfromR(Q)
    MOS_MAX = 4.9;
    MOS_MIN = 1.05;
    MOS = zeros(length(Q), 1);
    for iii = 1:length(Q)
        if (Q(iii) > 0 && Q(iii) < 100)
            MOS(iii) = (MOS_MIN+(MOS_MAX-MOS_MIN)/100*Q(iii)+Q(iii)*(Q(iii)-60)*(100-Q(iii))*7.0e-6);
        elseif (Q(iii) >= 100)
            MOS(iii) = MOS_MAX;
        else
            MOS(iii) = MOS_MIN;
        end
    end
end