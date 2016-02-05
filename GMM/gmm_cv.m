X_NB_SW = [X_words_train_pca_scores(:,1:5) X_images_train_pca_scores(:,1:5) X_image_features_train]; %36 percent error best

n = size(X_NB_SW,1);
n_folds = 10;
part = mod(randperm(n), n_folds) + 1;

totalNFoldError = 0;

number_of_cvs = 3;

for i = 1:number_of_cvs
    trainingIndices = find(part ~= i);
    trainingPoints = X_NB_SW(trainingIndices, :);
    trainingLabels = Y(trainingIndices, :);

    testingIndices = find(part == i);
    testingPoints = X_NB_SW(testingIndices, :);
    actualTestingLabels = Y(testingIndices, :);
    
    number_of_mixture_models = 100;
    predictedTestingLabels = gmm_predict(trainingPoints, trainingLabels, testingPoints, number_of_mixture_models);

    MLNFoldError = sum(predictedTestingLabels ~= actualTestingLabels)/size(actualTestingLabels,1);
    totalNFoldError = totalNFoldError + MLNFoldError;
end

error = totalNFoldError/number_of_cvs;