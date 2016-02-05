Y = load('../kit/kit/train/genders_train.txt');
X_words_train = load('../kit/kit/train/words_train.txt');
X_words_test = load('../kit/kit/test/words_test.txt');
X_images_train = load('../kit/kit/train/images_train.txt');
X_images_test = load('../kit/kit/test/images_test.txt');
X_image_features_train = load('../kit/kit/train/image_features_train.txt');
X_image_features_test = load('../kit/kit/test/image_features_test.txt');

[X_images_train_pca_loadings, X_images_train_pca_scores, X_images_train_pca_latent] = pca(X_images_train);
X_images_train_pca_varience_explained = cumsum(X_images_train_pca_latent)/sum(X_images_train_pca_latent);

[X_images_test_pca_loadings, X_images_test_pca_scores, X_images_test_pca_latent] = pca(X_images_test);
X_images_test_pca_varience_explained = cumsum(X_images_test_pca_latent)/sum(X_images_test_pca_latent);

[X_words_train_pca_loadings, X_words_train_pca_scores, X_words_train_pca_latent] = pca(X_words_train);
X_words_train_pca_varience_explained = cumsum(X_words_train_pca_latent)/sum(X_words_train_pca_latent);

[X_words_test_pca_loadings, X_words_test_pca_scores, X_words_test_pca_latent] = pca(X_words_test);
X_words_test_pca_varience_explained = cumsum(X_words_test_pca_latent)/sum(X_words_test_pca_latent);
