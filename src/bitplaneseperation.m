%B=A+(p5+p6+p7+p8);
 close all
J=imread('6.jpg');
J = rgb2gray(J);
imshow(J);
figure;
subplot(3,3,1);

imshow(J);
%d = J;
d=double(J);
p1=mod(d,2);
subplot(3,3,2);
imshow(p1);
p2=mod(floor(d/2),2);
subplot(3,3,3);
imshow(p2);
p3=mod(floor(d/4),2);
subplot(3,3,4);
imshow(p3);
p4=mod(floor(d/8),2);
subplot(3,3,5);
imshow(p4);
p5=mod(floor(d/16),2);
subplot(3,3,6);
imshow(p5);
p6=mod(floor(d/32),2);
subplot(3,3,7);
imshow(p6);
p7=mod(floor(d/64),2);
subplot(3,3,8);
imshow(p7);
p8=mod(floor(d/128),2);
subplot(3,3,9);
imshow(p8);
A = size(J);
B = size(J);
P=[p1 p2 p3 p4];
for j=1:4
        
         medianFilteredImage = medfilt2(P(j),[3 3]);
    noisePixels = ((P(j) == 255) | (P(j) == 0));
    fixedImage = size(P(j));
    fixedImage(noisePixels) = medianFilteredImage(noisePixels);
    F(j) = medianFilteredImage;
    A= A+F(j);
end
B=(p1+p2+p3+p4+p5+p6+p7+p8);
figure(45)
imshow(B,[]);
