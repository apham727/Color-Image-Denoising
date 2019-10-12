x = input('Enter image file name: ', 's');
rgbImage = imread("images/input/" + x);
compressedImage = imresize(rgbImage, 1/8);
noisyRGB = imnoise(compressedImage,'salt & pepper', 0.02);
imshow(noisyRGB)
imwrite(noisyRGB, strcat('images/noisy/noisy_', x), 'jpg');