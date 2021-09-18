load Trainset.mat
xdata = meas;
group = label;
training=[1,2,3,4];
training_labels=[1,5,3,4];
testing=[2,5,6,8];
testing_labels=[3,5,3,8];
k=[0,1];
KNN_(1,training,training_labels,testing,testing_labels)
%KNN_(1,1,2,1,0)