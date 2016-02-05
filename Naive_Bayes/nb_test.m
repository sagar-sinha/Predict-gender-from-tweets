train_x = [ X_words_train_pca_scores(:,1:8) X_images_train_pca_scores(:,1:20) X_image_features_train];
train_y = Y;
test_x = [ X_words_test_pca_scores(:,1:8) X_images_test_pca_scores(:,1:20) X_image_features_test];
predictions = nb_predict( train_x, train_y, test_x );
dlmwrite('submit_nb.txt', predictions);