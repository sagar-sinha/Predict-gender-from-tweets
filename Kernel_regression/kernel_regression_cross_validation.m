X_images_train_pca = X_images_train_pca_scores(:,1:40);
X_pca = [X_words_train X_images_train_pca X_image_features_train];
X_s = X_pca;
n = size(X_s,1);
n_folds = 10;
part = mod(randperm(n), n_folds) + 1;
sigma = 1100;
totalNFoldError = 0;
for i = 1:10
    trainingIndices = find(part ~= i);
    trainingPoints = X_s(trainingIndices, :);
    trainingLabels = Y(trainingIndices, :);

    testingIndices = find(part == i);
    testingPoints = X_s(testingIndices, :);
    actualTestingLabels = Y(testingIndices, :);
    predictions = zeros(size(actualTestingLabels,1),1);
    for j = 1:size(testingPoints,1)
        sum_my = 0;
        denominator = 0;
        for k = 1:size(trainingPoints,1)
            l = testingPoints(j,:) - trainingPoints(k,:);
            distance = exp( -((norm(l)^2)/(2*sigma*sigma)));
            sum_my = sum_my + distance*trainingLabels(k);
            denominator = denominator + distance;
        end
        y_pred = sum_my/denominator;
        predictions(j) = y_pred;
    end
    predictions(predictions> 0.5) = 1;
    predictions(predictions <= 0.5) = 0;
    ridgeNFoldError = sum(predictions ~= actualTestingLabels)/size(predictions,1);
    totalNFoldError = totalNFoldError + ridgeNFoldError;
end

error = totalNFoldError/max(part);