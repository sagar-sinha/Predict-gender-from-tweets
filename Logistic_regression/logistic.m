function [ precision, predicted_label ] = logistic( train_x, train_y, test_x, test_y )
    addpath('./liblinear');
    model = train(train_y, sparse(train_x), ['-s 0', 'col']);
    [predicted_label] = predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']);

    precision = 1 - sum(predicted_label~=test_y) / length(test_y);
end

