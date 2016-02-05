X_images_train_pca = X_images_train_pca_scores(:,1:40);
X_images_test_pca = X_images_test_pca_scores(:,1:40);
X_train_pca = [X_words_train X_images_train_pca X_image_features_train];
X_test_pca = [X_words_test X_images_test_pca X_image_features_test];
trainingPoints = X_train_pca; 
trainingLabels = Y;
testingPoints = X_test_pca;
sigma = 1100;
predictions = zeros(size(testingPoints,1),1);
for j = 1:size(testingPoints,1)
        numerator = 0;
        denominator = 0;
        for k = 1:size(trainingPoints,1)
            l = testingPoints(j,:) - trainingPoints(k,:);
            weight = exp( -((norm(l)^2)/(2*sigma*sigma)));
            numerator = numerator + weight*trainingLabels(k);
            denominator = denominator + weight;
        end
        y_pred = numerator/denominator;
        predictions(j) = y_pred;
end
predictions(predictions > 0.5) = 1;
predictions(predictions <= 0.5) = 0;
dlmwrite('submit.txt', predictions); 