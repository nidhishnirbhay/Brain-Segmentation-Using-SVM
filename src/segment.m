function  output = segment(Image)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[AA1, AA2, AA3, AA4] = Kclustering(Image); 
   
   cd Clusim
   file = uigetfile('*','pick file');
   segout= imread(file);
   cd ..
   
   boundary = bwboundaries(im2bw(segout));
   figure;
   imshow(Image); title('Tumor Localization');
   hold on;
   for ii=1:1:length(boundary)
       btemp = boundary{ii};
       plot(btemp(:,2),btemp(:,1),'r','LineWidth',2);
   end
   hold off;
   figure;
   imshow(segout);
   title('Segmented Image');
    
   I = segout;
   I = im2bw(I);

   Se = strel('disk',6);
   I = imerode(I,Se);
   I = imfill(I,'holes');
   out = bwlabel(I,8);
   output = bwareaopen(out,400);

return
end

