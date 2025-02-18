for l=1:20 
i=imread(strcat('training/meningioma/(',num2str(l),').jpg'));
i=rgb2gray(i);
%-------------------------------------
%2.Preprocessing
I=adapthisteq(i,'NumTiles',[2 2],'ClipLimit',0.0001);
%----------------------------------------------
%3.Skull Stripping
bw=I>95;
bw=bwareaopen(bw,800);
bw1=imcomplement(bw);
I1=uint8(bw1).*I;
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
bw3=A2img>1;
bw4=imcomplement(bw3);
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
%6-Features Extraction 
features=[];
glcm=graycomatrix(BWFF);
stats=graycoprops(glcm,'Contrast Correlation Energy Homogeneity');
Contrast=stats.Contrast;
Correlation=stats.Correlation;
Energy=stats.Energy;
Homogeneity=stats.Homogeneity;
state=regionprops(BWF,'Area');
area=state.Area;
Mean=mean2(BWFF);
Standard_Deviation=std2(BWFF);
Entropy=entropy(BWFF);
RMS=mean2(rms(BWFF));
Kurtosis=kurtosis(double(BWFF(:)));
Skewness= skewness(double(BWFF(:)));
Feature_Matrix1(l,:)=[Contrast,Correlation,Energy,Homogeneity,area,Entropy,Mean,Skewness,Kurtosis];
Out_Matrix1(l,:)='M';
end

for l=1:20 
i=imread(strcat('training/glioma/(',num2str(l),').jpg'));
i=rgb2gray(i);
%-------------------------------------
%2.Preprocessing
I=adapthisteq(i,'NumTiles',[2 2],'ClipLimit',0.0001);
%----------------------------------------------
%3.Skull Stripping
bw=I>95;
bw=bwareaopen(bw,800);
bw1=imcomplement(bw);
I1=uint8(bw1).*I;
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
bw3=A2img>1;
bw4=imcomplement(bw3);
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
%6-Features Extraction 
features=[];
glcm=graycomatrix(BWFF);
stats=graycoprops(glcm,'Contrast Correlation Energy Homogeneity');
Contrast=stats.Contrast;
Correlation=stats.Correlation;
Energy=stats.Energy;
Homogeneity=stats.Homogeneity;
state=regionprops(BWF,'Area');
area=state.Area;
Mean=mean2(BWFF);
Standard_Deviation=std2(BWFF);
Entropy=entropy(BWFF);
RMS=mean2(rms(BWFF));
Kurtosis=kurtosis(double(BWFF(:)));
Skewness= skewness(double(BWFF(:)));
Feature_Matrix2(l,:)=[Contrast,Correlation,Energy,Homogeneity,area,Entropy,Mean,Skewness,Kurtosis];
Out_Matrix2(l,:)='G';
end
Feature_Matrix=[Feature_Matrix1;Feature_Matrix2];
Out_Matrix=[Out_Matrix1;Out_Matrix2];
SVM_classifier=fitcsvm(Feature_Matrix,Out_Matrix);
save('SVM_classifier.mat');
