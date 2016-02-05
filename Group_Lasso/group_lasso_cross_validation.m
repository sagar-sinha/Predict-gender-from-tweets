X_images_pca = X_images_train_pca_scores(:,1:40);
n = size(X_words_train,1);
n_folds = 10;
part = mod(randperm(n), n_folds) + 1;
totalNFoldError = 0;

lambda1 = 3500;
lambda2 = 200;
lambda3 = 0;

for i = 1:max(part)
    trainingIndices = find(part ~= i);
    trainingPoints_words = X_words_train(trainingIndices, :);
    trainingPoints_image_features = X_image_features_train(trainingIndices, :);
    trainingPoints_images = X_images_pca(trainingIndices, :);
    trainingLabels = Y(trainingIndices, :);

    testingIndices = find(part == i);
    testingPoints_words = X_words_train( testingIndices, :);
    testingPoints_image_features = X_image_features_train(testingIndices, :);
    testingPoints_images = X_images_pca(testingIndices, :);
    actualTestingLabels = Y(testingIndices, :);
    
    [w1,w2,w3] = get_group_lasso_model(trainingLabels, trainingPoints_words,trainingPoints_image_features,trainingPoints_images, lambda1, lambda2, lambda3);
    
    testingPoints_words2 = [ones(size(testingPoints_words,1),1) testingPoints_words];
    testingPoints_image_features2 = [ones(size(testingPoints_image_features,1),1) testingPoints_image_features];
    testingPoints_images2 = [ones(size(testingPoints_images,1),1) testingPoints_images];
    predictions = testingPoints_words2*w1 + testingPoints_image_features2*w2 + testingPoints_images2*w3;
    predictions(predictions> 0.5) = 1;
    predictions(predictions <= 0.5) = 0;

    ridgeTestingLabels = predictions;
    ridgeNFoldError = sum(ridgeTestingLabels ~= actualTestingLabels)/size(ridgeTestingLabels,1);
    totalNFoldError = totalNFoldError + ridgeNFoldError;
end

error = totalNFoldError/max(part);