% load image
I = rgb2gray(imread('../myanmar.png'));
imageSize = size(I);
numRows = imageSize(1);
numCols = imageSize(2);
figure('name', 'Original')
imshow(I)

% image matrix to observations vector
X = double(reshape(I, numRows*numCols, []));

% kmeans
L1 = kmeans(X, 2, 'Replicates', 5);
kmeansClusters = reshape(L1, [numRows numCols]);
figure('name', 'Kmeans clustering')
imshow(label2rgb(kmeansClusters))

% gaussian mixture model
gmm = fitgmdist(X, 2);
L2 = cluster(gmm, X);
gmmClusters = reshape(L2, [numRows numCols]);
figure('name', 'GMM clustering')
imshow(label2rgb(gmmClusters))

% fuzzy c-means
[centers, U] = fcm(X, 2);
[values indexes] = max(U);
fcmClusters = reshape(indexes, [numRows numCols]);
figure('name', 'Fuzzy C-means')
imshow(label2rgb(fcmClusters))