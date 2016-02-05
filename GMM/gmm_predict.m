function [ predictions ] = gmm_predict( train_x, train_y, test_x, number_of_mixture_models)    
    GMModel = fitgmdist(train_x, number_of_mixture_models, 'RegularizationValue', 0.1);
    p_combined = zeros(size(train_x,1), number_of_mixture_models);
    for j = 1:number_of_mixture_models
        p_c = mvnpdf(train_x, GMModel.mu(j,:), GMModel.Sigma(:,:,j));
        p_combined(:,j) = p_c;
    end
    [~,cluster_indices_train] = max(p_combined,[],2);


    % separate into k clusters and assign labels to each cluster
    label = zeros(number_of_mixture_models,1);
    label_confidence = zeros(number_of_mixture_models,1);
    label_count = zeros(number_of_mixture_models,1);
    for j = 1:number_of_mixture_models
        c = find(cluster_indices_train == j);
        if c
            table = tabulate(train_y(c));
            [max_value,index] = max(table(:,2));
            label_confidence(j) = sum(train_y(c) == table(index,1))/size(train_y(c),1);
            label(j) = table(index,1);
            label_count(j) = size(train_y(c),1);
        end
    end
    
    p_combined_test = zeros(size(test_x,1), number_of_mixture_models);
    for j = 1:number_of_mixture_models
        p_c_test = mvnpdf(test_x, GMModel.mu(j,:), GMModel.Sigma(:,:,j));
        p_combined_test(:,j) = p_c_test;
    end
    [~,cluster_indices_test] = max(p_combined_test,[],2);
    
    predictions = label(cluster_indices_test);
end