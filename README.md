Description - In the quest to find the best model for the task of predicting the gender of a user from his or her tweets and profile picture we try and implement all the well known methods. Discriminative methods like Logistic regression and Support Vector Machine, generative methods like Naive Bayes and Gaussian Mixture model, instance based method like - Kernel regression, a regularization method like Group Lasso and a set of decision trees using LogitBoost. The best result is achieved by the ensemble of decision trees with an accuracy of 89.45 % on a test set of 5000 samples.

NOTE : Before running any of the following models, load all the training and test data. The script "Load_X_and_Y.m" does this easily for you. You may want to change the paths though. The programs assume that the data is present 
in the workspace. Also, we have not used any test data for training purposes. No PCA on pool set of training and test data is required for training purpose.

All the models are under their respective folders. 

I) Model - Group Lasso  

        1) To train and test the model run the file - "group_lasso_train_and_test.m". Running this scipt results in a file called "submit.txt" with the predictions on the test set.
        2) The Lambda values used for this method were figured out by cross validation. Look at the file "group_lasso_cross_validation.m" 



II) Model - Kernel Regression 

        1) To test the model run the file - "kernel_regression_test.m". Running this scipt results in a file called "submit.txt" with the predictions on the test set.
        2) The sigma value used for this method was figured out by cross validation. Look at the file "kernel_regression_cross_validation.m" 



III) Model - SVM 

      	1) To train and test the model run the file - "svm_train_and_test.m". Running this scipt results in a file called "submit.txt" with the predictions on the test set.




IV) Model - Logistic regression

       	1) Train and get predictions for test data: logistic_image_features_words_test.m
               - This file trains the Logistic regression on the training data and generates predictions for the test data.
               - The predictions are written to : submit_logistic_image_features_words.txt
		
		
	2) Helper functions:
	a) cross_validation_generic.m ---> function [error] = cross_validation_generic(X_s, Y, error_ml_technique, number_of_cvs)
            - cross_validation_generic performs crosvalidation on feature matrix X_s and label vector Y.
            - It uses the function handle error_ml_technique(trainingPoints,trainingLabels,testingPoints,actualTestingLabels)
              that returns the error betwen predictions for testingPoints and actualTestingLabels
            - number_of_cvs defines the number of validations to run
        b) logistic.m ---> function [ precision, predicted_label ] = logistic( train_x, train_y, test_x, test_y )
            The function trains a logistic regression on train_x and train_y and then returns the predicted labels for test_x
            as well as the precision of predictions w.r.t.test_y
        c) logistic_predict.m ---> function [ predicted_label ] = logistic_predict( train_x, train_y, test_x )
            The function trains a logistic regression on train_x and train_y and then returns ONLY the predicted labels for test_x
			
			
    	3) Cross Validation file: logistic_words_image_features_cv.m
    
        - This file was used to run cross validation on logistic regression to figure out which features give us the least error.
          For logistic regression we settled on [X_image_features_train X_words_train] as the best features by trial and error.
		  
		  
   

		
		
V) Model - Naive Bayes

	1) Train and get predictions for test data: nb_test.m
        - This file trains the naive bayes classifier on the training data and generates predictions for the test data.
        - The predictions are written to : submit_nb.txt
		
		
      	2) Helper functions:
        a) nb_predict.m ---> function [ predictedLabels ] = nb_predict( train_x, train_y, test_x )
        The function trains a naive bayes classifier on train_x and train_y and then returns ONLY the predicted labels for test_x
			
			
    	3) Cross Validation file: nb_cv.m
        - This file was used to run cross validation on naive bayes classifier to figure out which features give us the least error.
          We settled on [ X_words_train_pca_scores(:,1:8) X_images_train_pca_scores(:,1:20) X_image_features_train] as the best features by trial and error.
   

		
		

VI) Model - Gaussian Mixture Model
	1) Train and get predictions for test data: gmm_test.m
        - This file trains the gaussian mixture model on the training data and generates predictions for the test data.
        - The predictions are written to : submit_gmm.txt
		
		
	2) Helper functions:
        a) gmm_predict.m ---> function [ predictions ] = gmm_predict( train_x, train_y, test_x, number_of_mixture_models)  
            - The function trains a gaussian mixture model on only train_x and then assigns each cluster the most likely label (either male or female)
            - We then use mvnpdf to find the probability of each point in test_x belonging to different clusters
            - We the assign each point in test_x the label of the cluster with highest probability
			
			
    	3) Cross Validation file: gmm_cv.m
        - This file was used to run cross validation on naive bayes classifier to figure out which features AND Number of Mixture Models give us the least error.
          We settled on [X_words_train_pca_scores(:,1:5) X_images_train_pca_scores(:,1:5) X_image_features_train] as the best features and 100 as the Number of Mixture Models by trial and error.
		  
VII) Model - Ensemble Decision Tree

	1) Test and get predictions for test data by running the script - "prep_leaderboard_submission.m" .
	2) To train the model see the file "generate_model.m"
   
          
