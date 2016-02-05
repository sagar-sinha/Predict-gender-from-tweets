function [ predicted_label ] = logistic_predict( train_x, train_y, test_x )
    addpath('./liblinear');
    model = train(train_y, sparse(train_x), ['-s 0', 'col']);
    [predicted_label] = predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']);
end