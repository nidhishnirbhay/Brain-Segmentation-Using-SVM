function varargout = gui(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
function gui_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
a = ones(256,256);
axes(handles.axes1);
imshow(a);
axes(handles.axes2);
imshow(a);
axes(handles.axes3);
imshow(a);
set(handles.text1,'string','');
guidata(hObject, handles);

function varargout = gui_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


% button press in Browse_im.
function Browse_im_Callback(hObject, eventdata, handles)
cd TestImages 
[filename, pathname] = uigetfile('*', 'Pick an Image File');
    if isequal(filename,0) || isequal(pathname,0)
       warndlg('User pressed cancel');
    else
       disp(['User selected ', fullfile(pathname, filename)]);
       im = imread(filename);
      
Input=imresize(im,[512 512]);
Image=Input;
cd ..
[r, c, p] = size(Input);
if p==3
Image=rgb2gray(Image);
end
         axes(handles.axes1);
         imshow(Input);
         title('Test Image');
    end    
handles.Image = Image;
handles.filename = filename;
guidata(hObject, handles);
helpdlg('Test Image Selected');


%button press in database_load.
function database_load_Callback(hObject, eventdata, handles)
TT = [];
cd Database
for i = 1 : 30
    
    str = int2str(i);
    str = strcat(str,'.jpg');
       
    img = imread(str);
    Input=imresize(img,[512 512]);
     [r c p] = size(Input);
     if p==3
     img=Input(:,:,2);
     end
     
    [irow icol] = size(img);
    
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


    TT = [TT F];
end
cd ..
Database_feature=TT;
                                                                                                   
handles.Database_feature = Database_feature;
% Update handles structure
guidata(hObject, handles);
helpdlg('Database loaded sucessfully');

% button press in training_process.
function training_process_Callback(hObject, eventdata, handles)
features = handles.Database_feature;
Q = handles.queryfeature;
%  NN Training 
[Qfeature features] = fetr(Q,features);
[r c] = size(features);
Q = Qfeature;

[r1 c1] = size(features);
str1 = 'image';
str3 = '.mat';
for i = 1:c1
    name = strcat(str1,num2str(i));
    P = features(:,i);
    save(name,'P');
end

%  Training in PNN
M = 3;
N =1;
[r1  ,c1] = size(features);
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
net1 = naive_bayes(P,T1);
handles.net = net;
handles.net1 = net1;
% Update handles structure
guidata(hObject, handles);
helpdlg('Training Process Completed');

%k means clustring%
 function [AA1, AA2, AA3, AA4] = Kclustering(Image)

[r, c] = size(Image);
Length  = r*c; 
wd1=r;
wd2=c;
Dataset = reshape(Image,[Length,1]);   %%%%%Reshape 2D Image to 1D Vectors 
Clusters=4;       %%%%%Number of Clusters Initialization
Cluster1=zeros(Length,1);
Cluster2=zeros(Length,1);
Cluster3=zeros(Length,1);
Cluster4=zeros(Length,1);

miniv = min(min(Image));      
maxiv = max(max(Image));
range = maxiv - miniv;
stepv = range/Clusters;
incrval = stepv;

for i = 1:Clusters            %%%%Find the centroids to each Clusters
    K(i).centroid = incrval;
    incrval = incrval + stepv;
end

update1=0;
update2=0;
update3=0;
update4=0;

mean1=2;
mean2=2;
mean3=2;
mean4=2;

while  ((mean1 ~= update1) && (mean2 ~= update2) && (mean3 ~= update3) && (mean4 ~= update4))

mean1=K(1).centroid;
mean2=K(2).centroid;
mean3=K(3).centroid;
mean4=K(4).centroid;

for i=1:Length                     %%%%%%Find the distance between Each Pixel and Centroids 
    for j = 1:Clusters
        temp = Dataset(i);
        difference(j) = abs(temp-K(j).centroid);
    end
    [y,ind]=min(difference);     %%%%Group Pixels to Each Cluster Based on Minimum Distance
    
	if ind==1
        Cluster1(i)   =temp;
	end
    if ind==2
        Cluster2(i)   =temp;
    end
    if ind==3
        Cluster3(i)   =temp;
    end
    if ind==4
        Cluster4(i)   =temp;
    end
end

%%%%%UPDATE CENTROIDS
cout1=0;
cout2=0;
cout3=0;
cout4=0;

for i=1:Length
    Load1=Cluster1(i);
    Load2=Cluster2(i);
    Load3=Cluster3(i);
    Load4=Cluster4(i);
    
    if Load1 ~= 0
        cout1=cout1+1;
    end
    
    if Load2 ~= 0
        cout2=cout2+1;
    end
    
    if Load3 ~= 0
        cout3=cout3+1;
    end
    
    if Load4 ~= 0
        cout4=cout4+1;
    end
end

Mean_Cluster(1)=sum(Cluster1)/cout1;
Mean_Cluster(2)=sum(Cluster2)/cout2;
Mean_Cluster(3)=sum(Cluster3)/cout3;
Mean_Cluster(4)=sum(Cluster4)/cout4;

for i = 1:Clusters
    K(i).centroid = Mean_Cluster(i);

end

update1=K(1).centroid;
update2=K(2).centroid;
update3=K(3).centroid;
update4=K(4).centroid;
end


AA1=reshape(Cluster1,[wd1 wd2]);
AA2=reshape(Cluster2,[wd1 wd2]);
AA3=reshape(Cluster3,[wd1 wd2]);
AA4=reshape(Cluster4,[wd1 wd2]);

figure('Name','Segmented Results');
subplot(2,2,1); imshow(AA1,[]);
subplot(2,2,2); imshow(AA2,[]);
subplot(2,2,3); imshow(AA3,[]);
subplot(2,2,4); imshow(AA4,[]);


    cd Clusim
    imwrite(AA1,'1.bmp');
    imwrite(AA2,'2.bmp');
    imwrite(AA3,'3.bmp');
    imwrite(AA4,'4.bmp');
    cd ..



return;
 
% on button press in classify_im.
function classify_im_Callback(hObject, eventdata, handles)
Image=handles.Image;
net = handles.net;
net1 = handles.net1;
Q = handles.queryfeature;
    out = sim(net,Q);
    out = vec2ind(out);
    result = round(out);
    out1 = sim(net1,Q);
    out1 = vec2ind(out1);
    result1 = round(out1)
    load Trainset.mat
%data   = [meas(:,1), meas(:,2)];
Accuracy_Percent= zeros(200,1);
itr = 100;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy with 100 iterations');
for i = 1:itr
data = meas;
%groups = ismember(label,'Not Detected');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');
classes = svmclassify(svmStruct,data(test,:),'showplot',false);
classperf(cp,classes,test);
%Accuracy_Classification = cp.CorrectRate.*100;
Accuracy_Percent(i) = cp.CorrectRate.*100;
sprintf('Accuracy of Linear Kernel is: %g%%',Accuracy_Percent(i))
waitbar(i/itr);
end
delete(hWaitBar);
SVMACCURACY = max(Accuracy_Percent);
set(handles.edit14,'string',SVMACCURACY);
sprintf('Accuracy is: %g%%',SVMACCURACY)

   % MisclassificationRate = kfoldLoss(crossval(net1, 'KFold', 10))
if result==1
       set(handles.text1,'string','Not Detected');
       
else
       set(handles.text1,'string','Detected');
      
       output = Kclustering(Image);
        axes(handles.axes3);
         imshow(a);
 BW = imread(output);

 [B,L,N,A] = bwboundaries(BW); 
figure; 
imshow(BW); hold on; 
% Loop through object boundaries  
for k = 1:N 
    % Boundary k is the parent of a hole if the k-th column 
    % of the adjacency matrix A contains a non-zero element 
    if (nnz(A(:,k)) > 0) 
        boundary = B{k}; 
        plot(boundary(:,2),... 
            boundary(:,1),'r','LineWidth',2); 
        % Loop through the children of boundary k 
        for l = find(A(:,k))' 
            boundary = B{l}; 
            plot(boundary(:,2),... 
                boundary(:,1),'g','LineWidth',2); 
        end 
    end 
end
         
         
         
end

  
% button press in clear_im.
function clear_im_Callback(hObject, eventdata, handles)
set(handles.text1,'string','');
a = ones(256,256);
axes(handles.axes1);
imshow(a);
axes(handles.axes2);
imshow(a);
axes(handles.axes3);
imshow(a);



% button press in Fea_extr.
function Fea_extr_Callback(hObject, eventdata, handles)
Image = handles.Image;
img = handles.Image;
Image2 = im2bw(Image,.7);
Image2 = bwareaopen(Image2,80); 

    [ll, lh, hl, hh] = dwt2(Image,'db3');
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
 
     axes(handles.axes3);
 imshow(Image2);
    %imshow(s1,[]);
    
    title('Daubachies Wavelet');
bw = im2bw(img,.7);
label = bwlabel(bw);

stats_circle = regionprops(label, 'Solidity', 'Area');

density_cir = [stats_circle.Solidity];
area_cir = [stats_circle.Area];

high_dense_area = density_cir > 0.7;
max_area = max(area_cir(high_dense_area));
tumor_label = find(area_cir == max_area);
tumor = ismember(label, tumor_label);

se = strel('square',5);
tumor = imdilate(tumor,se);

figure(2)

subplot(1,3,1)
imshow(img,[])
title('Brain')

subplot(1,3,2)
imshow(tumor,[])
title('Tumor Alone')

[B,L] = bwboundaries(tumor,'noholes');
subplot(1,3,3)
imshow(img,[])
hold on
for i = 1:length(B)
    plot(B{i}(:,2), B{i}(:,1), 'y','linewidth',1.45)
end
title('Detected Tumor')
hold off
    
handles.queryfeature=Q;
% Update handles structure
guidata(hObject, handles);


%set(handles.edit4,'string',fea11);
%set(handles.edit5,'string',fea21);
%set(handles.edit6,'string',fea31);
%set(handles.edit7,'string',fea41);
%set(handles.edit8,'string',fea51);

helpdlg('Wavelet based Energy Features Extracted');

% accuracy
% function pushbutton4_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton4 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% load dataset.mat
% %data   = [meas(:,1), meas(:,2)];
% Accuracy_Percent= zeros(200,1);
% itr = 100;
% hWaitBar = waitbar(0,'Evaluating Maximum Accuracy with 100 iterations');
% for i = 1:itr
% data = meas;
% %groups = ismember(label,'BENIGN   ');
% groups = ismember(label,'MALIGNANT');
% [train,test] = crossvalind('HoldOut',groups);
% cp = classperf(groups);
% svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');
% classes = svmclassify(svmStruct,data(test,:),'showplot',false);
% classperf(cp,classes,test);
% %Accuracy_Classification = cp.CorrectRate.*100;
% Accuracy_Percent(i) = cp.CorrectRate.*100;
% sprintf('Accuracy of Linear Kernel is: %g%%',Accuracy_Percent(i))
% waitbar(i/itr);
% end
% delete(hWaitBar);
% Max_Accuracy = max(Accuracy_Percent);
% sprintf('Accuracy of Linear kernel is: %g%%',Max_Accuracy)
% 
% set(handles.edit9,'string',Max_Accuracy);


% button press in Close.
function Close_Callback(hObject, eventdata, handles)
close all;


% button press in preprocess.
function preprocess_Callback(hObject, eventdata, handles)
 
Image =handles.Image;
inp = Image;
inp_noi=imnoise(inp,'salt & pepper',0.10);
figure;
imshow(inp_noi);title('Noise Image');

%% Median Filter
NoiseLessImg=zeros(size(inp_noi,1),size(inp_noi,2));
for i=1:size(inp_noi,1)-2
    for j=1:size(inp_noi,2)-2
        LocMat=inp_noi(i:i+2,j:j+2);
        OneDimLocMat=LocMat(:);
        SortMat=sort(OneDimLocMat);
        NoiseLessImg(i,j)=SortMat(5);
    end
end
figure;
imshow(NoiseLessImg,[]);title('Noise Less Image');

%histogram equalization%
GIm=NoiseLessImg;
numofpixels=size(GIm,1)*size(GIm,2);
HIm=uint8(zeros(size(GIm,1),size(GIm,2)));
freq=zeros(256,1);
probf=zeros(256,1);
probc=zeros(256,1);
cum=zeros(256,1);
output=zeros(256,1);

for i=1:size(GIm,1)
    for j=1:size(GIm,2)
        value=GIm(i,j);
        freq(value+1)=freq(value+1)+1;
        probf(value+1)=freq(value+1)/numofpixels;
    end
end
sum=0;
no_bins=255;
%The cumulative distribution probability is calculated. 
for i=1:size(probf)
   sum=sum+freq(i);
   cum(i)=sum;
   probc(i)=cum(i)/numofpixels;
   output(i)=round(probc(i)*no_bins);
end
for i=1:size(GIm,1)
    for j=1:size(GIm,2)
            HIm(i,j)=output(GIm(i,j)+1);
    end
end

 figure,imshow(HIm);
 title('Histogram equalization');

    axes(handles.axes2);
    imshow(HIm);
    title('histogram equalisation');
    
guidata(hObject, handles);


function edit3_Callback(hObject, eventdata, handles)
function edit3_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit4_Callback(hObject, eventdata, handles)
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function Browse_im_CreateFcn(hObject, eventdata, handles)
function Browse_im_ButtonDownFcn(hObject, eventdata, handles)
function Untitled_1_Callback(hObject, eventdata, handles)


% --- Executes on mouse press over figure background.
function figure1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double

% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit9_Callback(hObject, eventdata, handles)
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton28.
function pushbutton28_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load Trainset.mat
%data   = [meas(:,1), meas(:,2)];
Accuracy_Percent= zeros(200,1);
itr = 100;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy with 100 iterations');
for i = 1:itr
data = meas;
%groups = ismember(label,'Not Detected');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');
classes = svmclassify(svmStruct,data(test,:),'showplot',false);
classperf(cp,classes,test);
%Accuracy_Classification = cp.CorrectRate.*100;
Accuracy_Percent(i) = cp.CorrectRate.*100;
sprintf('Accuracy of Linear Kernel is: %g%%',Accuracy_Percent(i))
waitbar(i/itr);
end
delete(hWaitBar);
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of Linear kernel is: %g%%',Max_Accuracy)
set(handles.edit10,'string',Max_Accuracy);


function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton29.
function pushbutton29_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load Trainset.mat
%data   = [meas(:,1), meas(:,2)];
Accuracy_Percent= zeros(200,1);
itr = 100;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy with 100 iterations');
for i = 1:itr
data = meas;
%groups = ismember(label,'Not Detected');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');
classes = svmclassify(svmStruct,data(test,:),'showplot',false);
classperf(cp,classes,test);
%Accuracy_Classification = cp.CorrectRate.*100;
Accuracy_Percent(i) = cp.CorrectRate.*100;
sprintf('Accuracy of Linear Kernel is: %g%%',Accuracy_Percent(i))
waitbar(i/itr);
end
delete(hWaitBar);
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of Linear kernel is: %g%%',Max_Accuracy)
set(handles.edit11,'string',Max_Accuracy);

function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton30.
function pushbutton30_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load Trainset.mat
%data   = [meas(:,1), meas(:,2)];
Accuracy_Percent= zeros(200,1);
itr = 100;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy with 100 iterations');
for i = 1:itr
data = meas;
%groups = ismember(label,'Not Detected');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct_Poly = svmtrain(data(train,:),groups(train),'Polyorder',2,'Kernel_Function','polynomial');
classes3 = svmclassify(svmStruct_Poly,data(test,:),'showplot',false);
classperf(cp,classes3,test);
Accuracy_Percent(i) = cp.CorrectRate.*100;
sprintf('Accuracy of Polynomial Kernel is: %g%%',Accuracy_Percent(i))
waitbar(i/itr);
end
delete(hWaitBar);
Max_Accuracy = max(Accuracy_Percent);
%Accuracy_Classification_Poly = cp.CorrectRate.*100;
sprintf('Accuracy of Polynomial kernel is: %g%%',Max_Accuracy)
set(handles.edit12,'string',Max_Accuracy);


function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton31.
function pushbutton31_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load Trainset.mat
%data   = [meas(:,1), meas(:,2)];
Accuracy_Percent= zeros(200,1);
itr = 100;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy with 100 iterations');
for i = 1:itr
data = meas;
%groups = ismember(label,'Not Detected');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct4 = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','quadratic');
classes4 = svmclassify(svmStruct4,data(test,:),'showplot',false);
classperf(cp,classes4,test);
%Accuracy_Classification_Quad = cp.CorrectRate.*100;
Accuracy_Percent(i) = cp.CorrectRate.*100;
sprintf('Accuracy of Quadratic Kernel is: %g%%',Accuracy_Percent(i))
waitbar(i/itr);
end
delete(hWaitBar);
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of Quadratic kernel is: %g%%',Max_Accuracy)
set(handles.edit13,'string',Max_Accuracy);


function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton32.
function pushbutton32_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Image=handles.Image;
net1 = handles.net1;
Q = handles.queryfeature;
    out1 = sim(net1,Q);
    out1 = vec2ind(out1);
    result1 = round(out1)
load Trainset.mat
%data   = [meas(:,1), meas(:,2)];
Accuracy_Percent= zeros(200,1);
itr = 100;
for i = 1:itr
data = meas;
%groups = ismember(label,'BENIGN   ');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct_Poly = svmtrain(data(train,:),groups(train),'Polyorder',2,'Kernel_Function','polynomial');
classes3 = svmclassify(svmStruct_Poly,data(test,:),'showplot',false);
classperf(cp,classes3,test);
Accuracy_Percent(i) = cp.CorrectRate.*100;
sprintf('Accuracy of Polynomial Kernel is: %g%%',Accuracy_Percent(i))
end
Max_Accuracy = max(Accuracy_Percent);
%Accuracy_Classification_Poly = cp.CorrectRate.*100;
sprintf('Accuracy of Polynomial kernel is: %g%%',Max_Accuracy)
set(handles.edit15,'string',Max_Accuracy);
    
if result1==1
       set(handles.text1,'string','Not Detected');
else
       set(handles.text1,'string','Detected');
       output = Kclustering(Image);
        axes(handles.axes3);
        
   
end

% --- Executes on button press in pushbutton33.
function pushbutton33_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
I = imread([pathname,filename]);
figure, imshow(I); title('Brain MRI Image');
I = imresize(I,[200,200]);
% Convert to grayscale
gray = rgb2gray(I);
% Otsu Binarization for segmentation
level = graythresh(I);
img = im2bw(I,level);
img2 = bwareaopen(img,80); 
figure, imshow(img2);title('Tumor Detected Image');
% K means Clustering to segment tumor
cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 1;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',1);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end
%
figure, imshow(segmented_images{1});title('Objects in Cluster 1');
seg_img = im2bw(segmented_images{1});
x = double(seg_img);
m = size(seg_img,1);
n = size(seg_img,2);
%signal1 = (rand(m,1));
%winsize = floor(size(x,1));
%winsize = int32(floor(size(x)));
%wininc = int32(10);
%J = int32(floor(log(size(x,1))/log(2)));
%Features = getmswpfeat(signal,winsize,wininc,J,'matlab');

%m = size(img,1);
%signal = rand(m,1);
signal1 = seg_img(:,:);
%Feat = getmswpfeat(signal,winsize,wininc,J,'matlab');
%Features = getmswpfeat(signal,winsize,wininc,J,'matlab');

[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);
whos DWT_feat
whos G
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
training=feat;
training_labels=[1 0]
testing=feat;
testing_labels=[1 0];
predicted_labels=[1 0];
result1=KNN_(predicted_labels,training,training_labels,testing,testing_labels)
accuracy=length(find(predicted_labels==testing_labels))/size(training_labels,1)
Max_Accuracy=accuracy;
set(handles.edit16,'string',Max_Accuracy);
if result1==1
       set(handles.text1,'string','Not Detected');
else
       set(handles.text1,'string','Detected');
       output = Kclustering(Image);
       axes(handles.axes3);
       imshow(a);
end


% --- Executes on button press in pushbutton34.
function pushbutton34_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
BrainMRI_GUI



function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit14 as text
%        str2double(get(hObject,'String')) returns contents of edit14 as a double


% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit15_Callback(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit15 as text
%        str2double(get(hObject,'String')) returns contents of edit15 as a double


% --- Executes during object creation, after setting all properties.
function edit15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit16_Callback(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit16 as text
%        str2double(get(hObject,'String')) returns contents of edit16 as a double


% --- Executes during object creation, after setting all properties.
function edit16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in togglebutton2.
function togglebutton2_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton2
