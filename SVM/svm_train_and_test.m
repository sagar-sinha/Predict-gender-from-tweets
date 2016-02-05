X_images_train_pca = X_images_train_pca_scores(:,1:40);
X_images_test_pca = X_images_test_pca_scores(:,1:40);
X_train_pca = [X_words_train X_images_train_pca X_image_features_train];
X_test_pca = [X_words_test X_images_test_pca X_image_features_test];


Xtrain = sparse(X_train_pca);
Xtest = sparse(X_test_pca);
Y_test_dummy = zeros(size(Xtest,1),1);
kernel = @(a,b)kernel_intersection(a,b);
predictions = kernel_libsvm(Xtrain, Y, Xtest, Y_test_dummy, kernel);
dlmwrite('submit.txt', predictions); 