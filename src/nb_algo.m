% Make a synthetic dataset
STE = rand(84,1)*20;
ZCR = randi(900,84,1);
%I=imread('5.jpg');
%J=rgb2gray(I);
Category = [repmat({'Detected'},42,1); repmat({'Not Detected'},42,1)];
STE(1:42) = 15 + STE(1:42);     % Make class 1 different so there's something to learn.
D = dataset(STE, ZCR, Category);
% Note: You should convert your dataset to 'Table' because dataset is deprecated:
T = dataset2table(D);
% You could have created a table directly like this: T = table(STE, ZCR, Category);
% Fit naive bayes model
model = fitcnb(T, 'Category')
% Compute cross-validated error. Chance misclassification rate would be 0.5
MisclassificationRate = kfoldLoss(crossval(model, 'KFold', 10))