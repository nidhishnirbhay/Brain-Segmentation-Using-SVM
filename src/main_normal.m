clc
clear all
close all
cd TestImages 
[filename, pathname] = uigetfile('*.jpg;*.bmp;*.gif', 'Pick an Image File');
    if isequal(filename,0) || isequal(pathname,0)
       warndlg('User pressed cancel');
    else
       disp(['User selected ', fullfile(pathname, filename)]);
       im = imread(filename);
      
Input=imresize(im,[512 512]);
Image=Input;
cd ..
[r c p] = size(Input);
if p==3
Image=rgb2gray(Image);
end
         imshow(Input);
         title('Test Image');
    end    
    helpdlg('Test Image Selected');
    
     [ll lh hl hh] = dwt2(Image,'db3');
    s1 = [ll lh;hl hh];

Min_val = min(min(lh));
Max_val = max(max(lh));
level = round(Max_val - Min_val);
GLCM = graycomatrix(lh,'GrayLimits',[Min_val Max_val],'NumLevels',level);
stat_feature = graycoprops(GLCM);
fea11 = stat_feature.Energy;
fea21 = stat_feature.Contrast;
fea31 = stat_feature.Correlation;
fea41=entropy(lh);
fea51 =mean(mean(lh));

F1=[fea11 fea21 fea31 fea41 fea51 ];

Min_val = min(min(hl));
Max_val = max(max(hl));
level = round(Max_val - Min_val);
GLCM = graycomatrix(hl,'GrayLimits',[Min_val Max_val],'NumLevels',level);
stat_feature = graycoprops(GLCM);
fea12 = stat_feature.Energy;
fea22 = stat_feature.Contrast;
fea32 = stat_feature.Correlation;
fea42=entropy(hl);
fea52 =mean(mean(hl));
F2=[fea12 fea22 fea32 fea42 fea52 ];
    Q = [F1 F2]';
    imshow(s1,[]);
    title('Daubachies Wavelet');
    helpdlg('Wavelet based Energy Features Extracted');
    
 TT = [];
cd Database
for i = 1 : 30
    
    % I have chosen the name of each image in databases as a corresponding
    % number. However, it is not mandatory!
    str = int2str(i);
    str = strcat(str,'.jpg');
       
    img = imread(str);
    Input=imresize(img,[512 512]);
     [r c p] = size(Input);
     if p==3
     img=Input(:,:,2);
     end
     
    [irow, icol] = size(img);
    % % % % % 1 level decomp
    
    [ll lh hl hh] = dwt2(img,'db3');
    s1 = [ll lh;hl hh];


   Min_val = min(min(lh));
Max_val = max(max(lh));
level = round(Max_val - Min_val);
GLCM = graycomatrix(lh,'GrayLimits',[Min_val Max_val],'NumLevels',level);
stat_feature = graycoprops(GLCM);
fea11 = stat_feature.Energy;
fea21 = stat_feature.Contrast;
fea31 = stat_feature.Correlation;
fea41=entropy(lh);
fea51 =mean(mean(lh));

F1=[fea11 fea21 fea31 fea41 fea51 ];

Min_val = min(min(hl));
Max_val = max(max(hl));
level = round(Max_val - Min_val);
GLCM = graycomatrix(hl,'GrayLimits',[Min_val Max_val],'NumLevels',level);
stat_feature = graycoprops(GLCM);
fea12 = stat_feature.Energy;
fea22 = stat_feature.Contrast;
fea32 = stat_feature.Correlation;
fea42=entropy(hl);
fea52 =mean(mean(hl));

F2=[fea12 fea22 fea32 fea42 fea52 ];

F=[F1 F2]';


%     F = [F1 F2 F3 F4 F5]';
    TT = [TT F];
end
cd ..
Database_feature=TT;
features = Database_feature;                                                                                          
helpdlg('Database loaded sucessfully');
    % % % % NN Training 
%%%%%% Importing Database features from workspace
[Qfeature features] = fetr(Q,features);
[r c] = size(features);
Q = Qfeature;

[r1, c1] = size(features);
str1 = 'image';
str3 = '.mat';
for i = 1:c1
    name = strcat(str1,num2str(i));
    P = features(:,i);
    save(name,'P');
end
features=Database_feature;
% % % % Training in PNN
M = 3;
N =1;
[r1 c1] = size(features);
str1 = 'image';str3 = '.mat';
for i = 1:c1
    name = strcat(str1,num2str(i));
    valu = load(name);
    P(:,i) = valu.P;

    if M==0
        N =N+1;
        M = 2;
    else
       M = M-1;
    end
    T1 (1,i) = N;
end
disp(P);
disp(T1);
T1 = ind2vec(T1);

net = svm(P,T1);
%  Q = queryfeature;
    out = sim(net,Q);
    out = vec2ind(out);
    result = round(out);
    
if result==1
       helpdlg('NORMAL');
else
     helpdlg('ABNORMAL');
end

%%%%%%%performance analysics%%%%%%%%

 Tp = 4; Fn = 2;  %%%%%%%after classification
   Fp = 1; Tn = 3;  %%%%%Tp --> Abnormality correctly classified as abnormal
                    %%%%%Fn --> Abnormality incorrectly classified as normal
                    %%%%%Fp --> Normal incorrectly classified as abnormal
                    %%%%%Tn --> Normal correctly classified as normal
                      
Sensitivity = (Tp./(Tp+Fn)).*100;
Specificity = (Tn./(Tn+Fp)).*100;

Accuracy = ((Tp+Tn)./(Tp+Tn+Fp+Fn)).*100;

% figure('Name','Performance Metrics','MenuBar','none'); 
bar3(1,Sensitivity,0.3,'m');
hold on;
bar3(2,Specificity,0.3,'r');
hold on;
bar3(3,Accuracy,0.3,'g');
hold off;

title('Performance Metrics');
xlabel('Parametrics--->');
zlabel('Value--->');
legend('Sensitivity','Specificity','Accuracy');

disp('Sensitivity: '); disp(Sensitivity);
disp('Specificity: '); disp(Specificity);
disp('Accuracy:'); disp(Accuracy);
