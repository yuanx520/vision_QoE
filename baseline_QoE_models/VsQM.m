function Q = VsQM( Qr, r, ms, ls )
    %   1. Qr: raw predicted quality scores of each frame
    %   2. r: frame rate
    %   3. ms: array of starting frames of the stalling events
    %   4. ls: array of duration of the stalling events in seconds
    % author: Zhengfang Duanmu
    
    % scaling factor C = 1 (does not matter since it does not have any
    % impact on correlation, so in this implementation, C is omitted
    % C = 1;
    
    % weight factor which represents the degree of degradation that each
    % segment adds to the total video degradation
    w = [1.3822; 1.2622; 1.0568; 0.9875];
    % total number of frames
    n_frames = length(Qr);
    % divide up the video into four time intervals
    f_bound = n_frames.*[1/4, 1/2, 3/4];
    % time of each segment in seconds
    t_segment = n_frames/r;
    % determine stalling events in each time interval
    s_g1 = ms < f_bound(1);
    s_g2 = ms >= f_bound(1) & ms < f_bound(2);
    s_g3 = ms >= f_bound(2) & ms < f_bound(3);
    s_g4 = ms >= f_bound(3);
    % total stalling length in seconds within each of the segment
    L = zeros(1,4);
    L(1) = sum(ls(s_g1));
    L(2) = sum(ls(s_g2));
    L(3) = sum(ls(s_g3));
    L(4) = sum(ls(s_g4));
    % VsQM
    Q = exp(-(L*w)/t_segment);
end

