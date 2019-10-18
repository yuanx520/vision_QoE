 def Yin2015QoE(bitrates, Tstall, Ts):
# %   This is an simplified implemenatation of the algorithm in ICME2014
# %   bitrates: array of bitrate, with each entry represents the bitrate of
# %   one segment
# %   Tstall: total rebuffer time in second
# %   Ts: startup delay in second

    # % model parameters
    lambda = 1;
    mu = 3000;
    mu_s = 3000;

    # % model parameters
    # % OVQ: overall video quality
    OVQ = sum(bitrates);
    # % OQV: overall quality variations
    OQV = sum(abs(bitrates[1:] - bitrates[1:end-1]));

    Q = OVQ - lambda * OQV - mu * Tstall - mu_s * Ts;
# end
