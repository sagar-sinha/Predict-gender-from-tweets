function predictions = make_final_prediction(model,X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model
X_words_test = X_test(:,1:5000);
X_images_test = X_test(:,5001:35000);
X_image_features_test =  X_test(:,35001:35007);
[X_images_test_pca_loadings, X_images_test_pca_scores, X_images_test_pca_latent] = pca(X_images_test);
X_images_test_pca = X_images_test_pca_scores(:,1:40);
X_test_pca = [X_words_test X_images_test_pca X_image_features_test];

[predictions score_ada] = predict(model.ClassTreeEns,X_test_pca);


