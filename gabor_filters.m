% Read and display input image
A = imread('../kobi.png');
Agray = rgb2gray(A);
figure
imshow(A)

% Design array of Gabor filters
imageSize = size(A);
numRows = imageSize(1);
numCols = imageSize(2);

wavelengthMin = 4/sqrt(2);
wavelengthMax = hypot(numRows,numCols);
n = 10;%floor(log2(wavelengthMax/wavelengthMin));
wavelength = 2.^(0:(n-2)) * wavelengthMin;

deltaTheta = 45;
orientation = 0:deltaTheta:(180-deltaTheta);

g = gabor(wavelength,orientation);
gabormag = imgaborfilt(Agray,g);

% Post-process the Gabor Magnitude Images into Gabor Features
% Gaussian filtering (get rid of local variation within texture)
for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    K = 3;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),K*sigma);
end

% Map of spatial location
X = 1:numCols;
Y = 1:numRows;
[X,Y] = meshgrid(X,Y);
featureSet = cat(3,gabormag,X);
featureSet = cat(3,featureSet,Y);

% 2D array to 1D vector of observations
numPoints = numRows*numCols;
X = reshape(featureSet,numRows*numCols,[]);

% normalize features using mean & std_dev
X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide,X,std(X));

% Cluster Gabor Texture Features
% kmeans
% L = kmeans(X,2,'Replicates',5);

% gaussian mixture model
% options = statset('MaxIter',1000);
% gmm = fitgmdist(X, 2, 'Options', options);
% L = cluster(gmm, X);

% fuzzy c-means
[centers, U] = fcm(X, 2);
[values indexes] = max(U);

L = reshape(indexes, [numRows numCols]);
figure
imshow(label2rgb(L))