i=imread(('training/glioma/(20).jpg'));
i=rgb2gray(i);
subplot(2,4,1);
imshow(i);
title('Original image','FontSize',12);
%-------------------------------------
%2.Preprocessing
I=adapthisteq(i,'NumTiles',[2 2],'ClipLimit',0.0001);
subplot(2,4,2);
imshow(I);
title('Enhanced Image','FontSize',12);
%----------------------------------------------
%3.Skull Stripping
bw=I>95;
bw=bwareaopen(bw,800);
bw1=imcomplement(bw);
I1=uint8(bw1).*I;
subplot(243);
imshow(bw);
title('Skull-stripped image','FontSize',12);
%---------------------------------------------------
%4.Wavelet decompose image:
X=bw;
[c,s]=wavedec2(X,2,'haar');
[H1,V1,D1] = detcoef2('all',c,s,1);
A1 = appcoef2(c,s,'haar',1); 
V1img = wcodemat(V1,25,'mat',1);
H1img = wcodemat(H1,255,'mat',1);
D1img = wcodemat(D1,255,'mat',1);
A1img = wcodemat(A1,255,'mat',1);
[H2,V2,D2] = detcoef2('all',c,s,2);
A2 = appcoef2(c,s,'haar',2); 
V2img = wcodemat(V2,256,'mat',1);
H2img = wcodemat(H2,255,'mat',1);
D2img = wcodemat(D2,255,'mat',1);
A2img = wcodemat(A2,255,'mat',1);
subplot(244);
imagesc(A2img);
title('Wavelet decomposed image','FontSize',12)
bw3=A2img>1;
bw4=imcomplement(bw3);
subplot(245);
imshow(bw3);
title('intense image','FontSize',12)
subplot(246);
imshow(bw4);
title('Inverse intense image','FontSize',12)
%-----------------------------------------------------
%5.Tumor region Detection:
[BW,maskedImage] = SEG1(I1);
BW1=bwareaopen(BW,2500);
BW2=uint8(BW1).*I;
[BW3,maskedImage] = SEG2(BW2);
BW3=bwareaopen(BW3,500);
[BWF,properties] = FR(BW3);
BWF=bwareaopen(BWF,801);
BWFF=uint8(BWF).*I1;
subplot(247);
imshow(BWF);
title('Tumor region Detection','FontSize',12);