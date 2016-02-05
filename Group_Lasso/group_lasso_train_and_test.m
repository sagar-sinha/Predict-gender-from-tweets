% If you want to test multiple times comment out the section under training below and test as many times as you may like! Have fun :)
% Start of training the model 

lambda1 = 3500;
lambda2 = 200;
lambda3 = 0;
X_images_train_pca = X_images_train_pca_scores(:,1:40);
[w1,w2,w3] = get_group_lasso_model(Y, X_words_train, X_image_features_train, X_images_train_pca, lambda1, lambda2, lambda3);

% End of training the model
 
 %Test 
X_images_test_pca = X_images_test_pca_scores(:,1:40);
X_words_test2 = [ones(size(X_words_test,1),1) X_words_test];
X_image_features_test2 = [ones(size(X_image_features_test,1),1) X_image_features_test];
X_images_test2 = [ones(size(X_images_test_pca,1),1) X_images_test_pca];
predictions = X_words_test2*w1 + X_image_features_test2*w2 + X_images_test2*w3;
predictions(predictions > 0.5) = 1;
predictions(predictions <= 0.5) = 0;
dlmwrite('submit.txt', predictions);  