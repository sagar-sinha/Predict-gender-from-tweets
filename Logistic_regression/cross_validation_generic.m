function [error] = cross_validation_generic(X_s, Y, error_ml_technique, number_of_cvs)
n = size(X_s,1);
n_folds = 10;
part = mod(randperm(n), n_folds) + 1;

totalNFoldError = 0;

%for i = 1:max(part)
for i = 1:number_of_cvs
    trainingIndices = find(part ~= i);
    trainingPoints = X_s(trainingIndices, :);
    trainingLabels = Y(trainingIndices, :);

    testingIndices = find(part == i);
    testingPoints = X_s(testingIndices, :);
    actualTestingLabels = Y(testingIndices, :);

    MLNFoldError = error_ml_technique(trainingPoints,trainingLabels,testingPoints, actualTestingLabels);
    totalNFoldError = totalNFoldError + MLNFoldError;
end

error = totalNFoldError/number_of_cvs;