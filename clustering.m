% load image
I = rgb2gray(imread('../myanmar.png'));

% display image
imageSize = size(I);
numRows = imageSize(1);
numCols = imageSize(2);
figure('name', 'Original')
imshow(I)

% image matrix to observations vector
X = double(reshape(I, numRows*numCols, []));

% kmeans
[L1, kmeansCenters] = kmeans(X, 2);
kmeansClusters = reshape(L1, [numRows numCols]);
kmeansCenters = sort(kmeansCenters);
figure('name', 'Kmeans clustering')
imshow(label2rgb(kmeansClusters))

% gaussian mixture model
gmm = fitgmdist(X, 2);
L2 = cluster(gmm, X);
mus = gmm.mu;
sigmas = squeeze(gmm.Sigma);
gmmClusters = reshape(L2, [numRows numCols]);
figure('name', 'GMM clustering')
imshow(label2rgb(gmmClusters))

% plot, image histogram, kmeans centroids, and gaussian mixtures
figure('name', 'Image histogram')
[counts, x] = imhist(I, 100);
bar(x, counts, 'b');
vline(kmeansCenters(1), 'r', strcat('Water centroid:', num2str(kmeansCenters(1))));
vline(kmeansCenters(2), 'r', strcat('Land centroid:', num2str(kmeansCenters(2))));
vline(55, 'k', '');
vline(80, 'k', '');
hold on;
text(55, 2000, '\leftarrow x = 55');
text(80, 2000, '\leftarrow x = 80');
plot(numRows*numCols * normpdf([0:255], mus(1), sqrt(sigmas(1))), 'g', 'LineWidth', 2);
plot(numRows*numCols * normpdf([0:255], mus(2), sqrt(sigmas(2))), 'g', 'LineWidth', 2);
xlim([0 255]);
xlabel('Intensity value');
ylabel('Frequency of occurrence');
plot(normpdf([0:255], mus(2), sqrt(sigmas(2))))

% fuzzy c-means
[centers, U] = fcm(X, 2);
[values indexes] = max(U);
fcmClusters = reshape(indexes, [numRows numCols]);
figure('name', 'Fuzzy C-means')
imshow(label2rgb(fcmClusters)); return;

% test equality of fuzzy c-means & kmeans results
figure('name', 'equal')
imshow(kmeansClusters == fcmClusters)