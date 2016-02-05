% Load data
X_test = [X_words_test X_images_test X_image_features_test];
model = init_model;
predictions = make_final_prediction(model, X_test);

% Use turnin on the output file
% turnin -c cis520 -p leaderboard submit.txt
dlmwrite('submit.txt', predictions);
