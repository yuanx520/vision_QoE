clear; clc;

load('sourceVideo.mat');
load('actualBitrate.mat');
sourceNames = sourceVideo.name;
tVideo = 10; % 10 seconds test video (without stalling)
segmentDuration = 2;

count = 1;
for iii = 1:length(sourceNames)
    load(['representations/' sourceNames{iii} '.mat']);
    load(['streamInfo/' sourceNames{iii} '.mat']);
    bitrateLadder = eval(['actualBitrate.' sourceNames{iii}]);
    for jjj = 1:length(streamInfo)
        videoInfo = streamInfo(jjj, :);
        fps = double(videoInfo{1});
        selectedRep = double(videoInfo{2});
        bitrates = [];
        seqPSNR = [];

        % stalling duration
        stallTime = (sum(streamInfo{jjj,5})+streamInfo{jjj,3}) / fps;
        % overall duration of the streaming session
        duration = (sum(streamInfo{jjj,5})+streamInfo{jjj,3}) / fps + tVideo;

        switching = (videoInfo{2}(2:end) ~= videoInfo{2}(1:end-1));
        mw = (1+2*fps*(1:4)).*switching;
        % magnitude of switching in kbps
        mw(mw == 0) = [];

        for kkk = 1:length(selectedRep)
            b = bitrateLadder(selectedRep(kkk)+1);
            bitrates = [bitrates, b]; %#ok

            load(['VQAResults/PSNR/' sourceNames{iii} '/' representation{selectedRep(kkk)+2, 5}]);
            segmentPSNR = psnr((kkk-1)*segmentDuration*fps+1:kkk*segmentDuration*fps);
            seqPSNR = [seqPSNR; segmentPSNR]; %#ok
        end

        % duration of initial buffering
        tInit = double(videoInfo{3}) / fps;
        % duration of stalling events in second
        lStall = double(videoInfo{5}) ./ fps;
        % number of stalling events
        nStall = length(lStall);
        % average duration of stalling event
        tStall = mean(lStall);
        if (isnan(tStall))
            tStall = 0;
        end

        Q_PSNR(1, count) = mean(seqPSNR); %#ok
        count = count + 1;
    end

%     save(['VQAResult/PSNR/' sourceNames{iii} '.mat'], 'Q_PSNR');
%     clear Q_PSNR;
    count = 1;
end

% To obtain 450 results, we need to concatenate results of each video
% content. See sample code below:
QO_PSNR = [];
for iii = 1:length(sourceNames)
    load(['VQAResults/PSNR/' sourceNames{iii}]);
    QO_PSNR = [QO_PSNR; Q_PSNR']; %#ok
end
