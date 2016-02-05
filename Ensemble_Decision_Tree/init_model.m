function model = init_model()

load('ClassTreeEns.mat');
model.ClassTreeEns = ClassTreeEns;

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
