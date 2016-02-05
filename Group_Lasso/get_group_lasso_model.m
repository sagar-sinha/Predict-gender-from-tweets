function [w1,w2,w3] = get_group_lasso_model(trainingLabels, trainingPoints_words,trainingPoints_image_features,trainingPoints_images, lambda1, lambda2, lambda3)

[residual1,w1] = get_residual_and_weight(trainingLabels,  trainingPoints_words, lambda1);
[residual2,w2] = get_residual_and_weight(residual1,  trainingPoints_image_features, lambda2);
[~,w3] = get_residual_and_weight(residual2,  trainingPoints_images, lambda3);