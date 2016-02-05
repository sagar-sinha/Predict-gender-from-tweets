X_NB = [ X_words_train_pca_scores(:,1:8) X_images_train_pca_scores(:,1:20) X_image_features_train]; %27 percent

% Figuring out good values

%X_NB = [ X_words_train_pca_scores(:,1:5) ]; %34 percent
%X_NB = [ X_images_train_pca_scores(:,1:20) ]; %37 percent
%X_NB = [ X_image_features_train ];  %33 percent
%X_NB = [ X_image_features_train ]; %X_images_train_pca_scores(:,1:20) X_image_features_train];
%Y = Y

% X_NB =  X_words_train_pca_scores(:,1:4500);
% nb_error_function = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= nb_predict(xtrain, ytrain, xtest));
% inmodel_nb = sequentialfs(nb_error_function,X_NB,Y);
% X_NB_SW = X_NB(:,inmodel_nb);

%X_NB_SW_words_image_features = [X_NB_SW X_image_features_train];
    
n = size(X_NB,1);
n_folds = 10;
part = mod(randperm(n), n_folds) + 1;

totalNFoldError = 0;

number_of_cvs = 10;

for i = 1:number_of_cvs
    trainingIndices = find(part ~= i);
    trainingPoints = X_NB(trainingIndices, :);
    trainingLabels = Y(trainingIndices, :);

    testingIndices = find(part == i);
    testingPoints = X_NB(testingIndices, :);
    actualTestingLabels = Y(testingIndices, :);
    
    predictedTestingLabels = nb_predict(trainingPoints, trainingLabels, testingPoints);

    MLNFoldError = sum(predictedTestingLabels ~= actualTestingLabels)/size(actualTestingLabels,1);
    totalNFoldError = totalNFoldError + MLNFoldError;
end

error = totalNFoldError/number_of_cvs;
