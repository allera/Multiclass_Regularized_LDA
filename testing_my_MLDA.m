%5 voxels, 4 classes, 250 samples per class, total of 1000 samples (can be for example 1000 subjetcs)

%feature space of dimesnion 5, last dimension (or voxel) has no
%discriminative signal

x=abs(randn(5,1000));% random noyse
%add some signal diferenciating classes
x(1,1:250)=2+x(1,1:250);
x(2,251:500)=3+x(2,251:500);
x(3,501:750)=4+x(3,501:750);
x(4,751:end)=5+x(4,751:end);

imagesc(x)

Y=[ones(1,250) 2.*ones(1,250) 3.*ones(1,250) 4.*ones(1,250) ];% class labels

%classify 
Xtrain=x;
Ytrain=Y;
Xtest=Xtrain;
Ytest=Ytrain;


addpath(genpath('./code'))
lda_method='LDA';%'sim_diag';%'LDA'
[Output  probout, info] = MultiClassRegularizedLDA(Xtest',Xtrain',Ytrain,lda_method);

sum(Output==Ytest)/numel(Ytest)

