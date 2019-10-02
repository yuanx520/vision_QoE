function Q = P1203(audio_bitrate, video_bitrate, disRes, codRes, fps, handheld, ms, ls, forest)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This is an implementation of ITU P.1203
    % Input:
    % 1. audio_bitrate: one entry per second in kbps/s
    % 2. video_bitrate: one entry per second in kbps/s
    % 3. disRes: display resolution
    % 4. codRes: coding resolution
    % 5. fps: frame rate
    % 6. handheld: boolean indicator of mobile devices
    % 7. ms: position of stalling position in sec
    % 8. ls: duration of stalling events in sec
    % 9. forest: random forest in cell array
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    O21 = P12032(audio_bitrate, 1);
    O22 = P12031(video_bitrate, disRes, codRes, fps, handheld);
    Q = P12033( O21, O22, ms, ls, forest );
end