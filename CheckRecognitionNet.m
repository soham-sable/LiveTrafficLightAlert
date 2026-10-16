
S = load('detectorNet.mat','dlyolo');
net = S.dlyolo;

img = imread('stop.jpg');
[h,w,~] = size(img);

img448 = imresize(img,[448 448]);
img448 = im2single(img448(:,:,3:-1:1));
dlImg = dlarray(img448,'SSC');

raw = predict(net,dlImg,Outputs='fc26');
p = gather(extractdata(raw));
p = p(:);

% Find the highest raw confidence.
conf = p(50:147);
[bestConf,idx] = max(conf);

fprintf('Highest raw confidence: %.6f\n',bestConf);
fprintf('Best confidence index: %d\n',idx);

% Calculate scores for every box.
scores = zeros(98,1);
k = 1;

for i = 0:48
    classIndex = i + 1;

    for n = 0:1
        confIndex = 49 + i*2 + n + 1;
        scores(k) = p(classIndex) * p(confIndex);
        k = k + 1;
    end
end

fprintf('Highest calculated score: %.6f\n',max(scores));
fprintf('Boxes above 0.01: %d\n',sum(scores > 0.01));
fprintf('Boxes above 0.2: %d\n',sum(scores > 0.2));
