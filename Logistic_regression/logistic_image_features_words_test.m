addpath('./liblinear');
train_x = [X_image_features_train X_words_train];
train_y = Y;
test_x = [X_image_features_test X_words_test];
predictions = logistic_predict( train_x, train_y, test_x );
dlmwrite('submit_logistic_image_features_words.txt', predictions);