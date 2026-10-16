
% COMBINED LIVE TRAFFIC LIGHT AND TRAFFIC SIGN DETECTION
% Prototype only. Not for real driving decisions.

clc;

% ---------------------------------------------------------
% 1. PROJECT FOLDER AND MODEL FILES
% ---------------------------------------------------------

projectFolder = ...
    'C:\Users\Soham\OneDrive\Documents\TrafficLightProject';

addpath(projectFolder);

detectorFile = fullfile(projectFolder, 'detectorNet.mat');
recognitionFile = fullfile(projectFolder, 'recognitionNet.mat');
signFunctionFile = fullfile(projectFolder, 'detectTrafficSigns.m');

fprintf('Detector file exists: %d\n', isfile(detectorFile));
fprintf('Recognition file exists: %d\n', isfile(recognitionFile));
fprintf('Sign function exists: %d\n', isfile(signFunctionFile));

if ~isfile(detectorFile) || ~isfile(recognitionFile)
    error('Cannot find the traffic-sign model files.');
end

if ~isfile(signFunctionFile)
    error('Cannot find detectTrafficSigns.m.');
end

% ---------------------------------------------------------
% 2. LOAD MODELS ONCE
% ---------------------------------------------------------

detector = yolov4ObjectDetector('tiny-yolov4-coco');

signDetectorData = load(detectorFile, 'dlyolo');
signRecognitionData = load(recognitionFile, 'dlconvnet');

dlyolo = signDetectorData.dlyolo;
dlconvnet = signRecognitionData.dlconvnet;

fprintf('Models loaded. Opening webcam...\n');

% ---------------------------------------------------------
% 3. OPEN ONE WEBCAM
% ---------------------------------------------------------

cam = webcam;

fig = figure( ...
    'Name', 'Live Traffic Light and Traffic Sign Detection', ...
    'NumberTitle', 'off');

ax = axes('Parent', fig);

previousState = "SEARCHING";
stateHistory = strings(0,1);
signErrorShown = false;

try
    while ishandle(fig)

        % Capture one live frame
        rawFrame = snapshot(cam);
        displayFrame = rawFrame;

        % -------------------------------------------------
        % 4. TRAFFIC-SIGN DETECTION
        % -------------------------------------------------

        try
            [signBoxes, signNames] = detectTrafficSigns( ...
                rawFrame, dlyolo, dlconvnet);

            for k = 1:size(signBoxes,1)
                displayFrame = insertObjectAnnotation( ...
                    displayFrame, 'rectangle', signBoxes(k,:), ...
                    ['SIGN: ' char(string(signNames{k}))], ...
                    'Color', 'yellow', 'LineWidth', 3);
            end

        catch signError
            if ~signErrorShown
                warning('Traffic-sign detection failed: %s', ...
                    signError.message);
                signErrorShown = true;
            end
        end

        % -------------------------------------------------
        % 5. TRAFFIC-LIGHT DETECTION
        % -------------------------------------------------

        [lightBoxes, lightScores, lightLabels] = ...
            detect(detector, rawFrame, 'Threshold', 0.05);

        keep = contains(lower(string(lightLabels)), ...
            'traffic light') & lightScores >= 0.05;

        lightBoxes = lightBoxes(keep,:);
        lightScores = lightScores(keep);

        detectedState = "NO TRAFFIC LIGHT";

        if ~isempty(lightBoxes)

            [~, best] = max(lightScores);
            box = round(lightBoxes(best,:));

            x1 = max(1, box(1));
            y1 = max(1, box(2));
            x2 = min(size(rawFrame,2), box(1)+box(3)-1);
            y2 = min(size(rawFrame,1), box(2)+box(4)-1);

            if x2 > x1 && y2 > y1
                crop = rawFrame(y1:y2, x1:x2, :);
                detectedState = estimateLampColor(crop);

                displayFrame = insertObjectAnnotation( ...
                    displayFrame, 'rectangle', ...
                    [x1 y1 x2-x1+1 y2-y1+1], ...
                    ['TRAFFIC LIGHT: ' char(detectedState)], ...
                    'Color', 'green', 'LineWidth', 3);
            end
        end

        % -------------------------------------------------
        % 6. SMOOTH STATUS OVER FIVE FRAMES
        % -------------------------------------------------

        stateHistory(end+1,1) = detectedState;

        if numel(stateHistory) > 5
            stateHistory(1) = [];
        end

        stableState = "SEARCHING";

        if numel(stateHistory) == 5 && ...
                all(stateHistory == stateHistory(1))
            stableState = stateHistory(1);
        end

        % -------------------------------------------------
        % 7. DISPLAY BOTH DETECTORS
        % -------------------------------------------------

        displayFrame = insertText(displayFrame, [10 10], ...
            "TRAFFIC LIGHT STATUS: " + stableState, ...
            'FontSize', 18, 'BoxOpacity', 0.75);

        imshow(displayFrame, 'Parent', ax);
        title(ax, 'Live Traffic Light and Traffic Sign Detection');
        drawnow;

        % -------------------------------------------------
        % 8. ALERT WHEN THE STABLE STATE CHANGES
        % -------------------------------------------------

        if stableState ~= "SEARCHING" && ...
                stableState ~= previousState

            fprintf('\nTRAFFIC LIGHT ALERT: %s\n', ...
                char(stableState));

            switch stableState
                case "RED"
                    fprintf('Reminder: stop safely where required.\n');
                    beep;

                case "YELLOW"
                    fprintf('Reminder: use caution.\n');
                    beep;

                case "GREEN"
                    fprintf('Proceed only when the way is clear.\n');
                    beep;

                case "UNKNOWN"
                    fprintf('Lamp color is uncertain.\n');
            end

            previousState = stableState;
        end
    end

catch ME
    clear cam;

    if ishandle(fig)
        close(fig);
    end

    rethrow(ME);
end

clear cam;

% ---------------------------------------------------------
% 9. TRAFFIC-LIGHT COLOR ESTIMATION
% ---------------------------------------------------------

function state = estimateLampColor(crop)

    hsv = rgb2hsv(im2double(crop));

    H = hsv(:,:,1);
    S = hsv(:,:,2);
    V = hsv(:,:,3);

    active = S > 0.45 & V > 0.50;

    redMask = active & (H < 0.04 | H > 0.94);
    yellowMask = active & H >= 0.10 & H <= 0.18;
    greenMask = active & H >= 0.22 & H <= 0.45;

    counts = [nnz(redMask), nnz(yellowMask), nnz(greenMask)];
    [largest, idx] = max(counts);

    if largest < 5
        state = "UNKNOWN";
        return;
    end

    names = ["RED", "YELLOW", "GREEN"];
    state = names(idx);
end
