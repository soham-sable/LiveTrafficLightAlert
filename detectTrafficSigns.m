
function [selectedBbox, signNames] = detectTrafficSigns(img, dlyolo, dlconvnet)
% Detect and recognize traffic signs.
% Detector input: 448x448x3
% Recognition input: 48x48x3 with zero-center normalization.

classNames = { ...
    'addedLane','slow','dip','speedLimit25','speedLimit35', ...
    'speedLimit40','speedLimit45','speedLimit50','speedLimit55', ...
    'speedLimit65','speedLimitUrdbl','doNotPass','intersection', ...
    'keepRight','laneEnds','merge','noLeftTurn','noRightTurn','stop', ...
    'pedestrianCrossing','stopAhead','rampSpeedAdvisory20', ...
    'rampSpeedAdvisory45','truckSpeedLimit55','rampSpeedAdvisory50', ...
    'turnLeft','rampSpeedAdvisoryUrdbl','turnRight','rightLaneMustTurn', ...
    'yield','yieldAhead','school','schoolSpeedLimit25','zoneAhead45', ...
    'signalAhead'};

selectedBbox = zeros(0,4);
signNames = strings(0,1);

if isempty(img) || ndims(img) ~= 3 || size(img,3) ~= 3
    return;
end

originalImg = img;
[h,w,~] = size(originalImg);

%% 1. Prepare detector input
detectorImg = imresize(originalImg,[448 448]);
detectorImg = im2single(detectorImg(:,:,3:-1:1));
dlImg = dlarray(detectorImg,'SSC');

%% 2. Get detector predictions
raw = predict(dlyolo,dlImg,Outputs='fc26');
predictions = gather(extractdata(raw));
predictions = predictions(:);

if numel(predictions) ~= 539
    warning('Expected 539 detector outputs; received %d.', ...
        numel(predictions));
    return;
end

%% 3. Decode YOLO predictions
classes = 1;
num = 2;
side = 7;
thresh = 0.2;

boxes = zeros(0,4);
probs = zeros(0,1);

for i = 0:(side*side-1)
    for n = 0:num-1

        p_index = side*side*classes + i*num + n + 1;
        scale = predictions(p_index);

        for j = 0:classes
            class_index = i*classes + 1;
            tempProb = scale * predictions(class_index+j);

            if tempProb <= thresh
                continue;
            end

            row = floor(i/side);
            col = mod(i,side);

            box_index = side*side*(classes+num) ...
                + (i*num+n)*4 + 1;

            bxX = (predictions(box_index)+col)/side;
            bxY = (predictions(box_index+1)+row)/side;
            bxW = predictions(box_index+2)^2;
            bxH = predictions(box_index+3)^2;

            boxX = (bxX-bxW/2)*w+1;
            boxY = (bxY-bxH/2)*h+1;
            boxW = bxW*w;
            boxH = bxH*h;

            x1 = max(1,floor(boxX));
            y1 = max(1,floor(boxY));
            x2 = min(w,ceil(boxX+boxW-1));
            y2 = min(h,ceil(boxY+boxH-1));

            if x2 <= x1 || y2 <= y1
                continue;
            end

            boxes(end+1,:) = [x1 y1 x2-x1+1 y2-y1+1]; %#ok<AGROW>
            probs(end+1,1) = tempProb; %#ok<AGROW>
        end
    end
end

if isempty(boxes)
    return;
end

%% 4. Remove overlapping boxes
[selectedBbox,~] = selectStrongestBbox( ...
    boxes,probs,'OverlapThreshold',0.5);

if isempty(selectedBbox)
    return;
end

%% 5. Recognize each sign with zero-center normalization
meanImage = dlconvnet.Layers(1).Mean;
signNames = strings(size(selectedBbox,1),1);

for k = 1:size(selectedBbox,1)

    x1 = selectedBbox(k,1);
    y1 = selectedBbox(k,2);
    x2 = min(w,x1+selectedBbox(k,3)-1);
    y2 = min(h,y1+selectedBbox(k,4)-1);

    crop = originalImg(y1:y2,x1:x2,:);

    if isempty(crop)
        signNames(k) = "unknown";
        continue;
    end

    crop = imresize(crop,[48 48]);
    crop = single(crop);

    % Important: apply the trained network's stored mean.
    crop = crop - meanImage;

    dlCrop = dlarray(crop,'SSC');
    output = predict(dlconvnet,dlCrop);

    probabilities = gather(extractdata(output));
    [~,idx] = max(probabilities(:));

    if idx >= 1 && idx <= numel(classNames)
        signNames(k) = string(classNames{idx});
    else
        signNames(k) = "unknown";
    end
end

end
