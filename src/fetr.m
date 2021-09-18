function [ProjectedTestImage, ProjectedImages] = fetr(Qfeature,Dfeature)
ProjectedImages = [];
Train_Number = size(Dfeature,2);
for i = 1 : Train_Number
    temp = Dfeature(:,i); % Projection of centered images into facespace
    ProjectedImages = [ProjectedImages temp]; 
end

%%%%%%%%%%%%%%%%%%%%%%%% 
ProjectedTestImage = Qfeature; % Test image feature vector

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Euclidean distances 
% Euclidean distances between the projected test image and the projection
% of all centered training images are calculated. 

Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages(:,i);
    temp = sqrt(sum(( ProjectedTestImage - q ).^2));
    Euc_dist = [Euc_dist temp];
end
[Euc_dist_min , Recognized_index] = min(Euc_dist);
ProjectedTestImage = ProjectedImages(:,Recognized_index);
ER = sort(Euc_dist,'descend');
figure;
plot(ER);
xlabel('Iterations');
ylabel('Error values');
title('Performance Graph');
disp('Euclidean Distance:');
disp(ER);


