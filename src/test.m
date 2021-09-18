close all
clear
clc

img = imread('5.jpg');

bw = im2bw(img, 0.7);
label = bwlabel(bw);

stats_circle = regionprops(label, 'Solidity', 'Area');

density_cir = [stats_circle.Solidity];
area_cir = [stats_circle.Area];

high_dense_area = density_cir > 0.3;
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
