X_images_train_pca = X_images_train_pca_scores(:,1:40);
X_train_pca = [X_words_train X_images_train_pca X_image_features_train];
ClassTreeEns = fitensemble(X_train_pca,Y,'LogitBoost',1000,'Tree');
save('ClassTreeEns.mat', 'ClassTreeEns');

