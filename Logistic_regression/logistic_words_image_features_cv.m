X_log = [X_image_features_train X_words_train];

error_ml_technique = @(train_x,train_y,test_x,test_y) ( 1 - logistic( train_x, train_y, test_x, test_y ) ) ;
cross_validation_generic( X_log, Y, error_ml_technique, 10)