function quant = mode0(br, codRes, fps)
    % model parameters
    a1 = 11.99835;
    a2 = -2.99992;
    a3 = 41.24751;
    a4 = 0.13183;
    
    % core model
    bpp = br ./ (codRes .* fps);
    quant = a1 + a2 .* log (a3 + log(br) + log(br .* bpp + a4));
end