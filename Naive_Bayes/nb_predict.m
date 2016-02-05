function [ predictedLabels ] = nb_predict( train_x, train_y, test_x )
    NBModel = fitNaiveBayes(train_x,train_y);
    predictedLabels = predict(NBModel,test_x);
end

