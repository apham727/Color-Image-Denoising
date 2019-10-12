x = input('Enter image file name: ', 's');
rgbImage = imread("images/input/" + x);
noisyRGB = imnoise(rgbImage,'salt & pepper', 0.02);
imshow(noisyRGB)
imwrite(noisyRGB, strcat('images/noisy/noisy_', x), 'jpg');