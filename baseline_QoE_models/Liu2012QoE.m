function Q = Liu2012QoE( bitrates, bufratio )
    Q = -3.7 * bufratio + bitrates / 20;
end

