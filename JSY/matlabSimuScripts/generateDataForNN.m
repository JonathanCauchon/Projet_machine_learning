%Uncomment if fonction is directly used:
    %batchSize=2;
    %batchNum=1;

%%Wavelength Range
startWL=1500;
endWL=1600;

%Speed/Accuracy
res=1001;        %number of wavelengths (spectral resolution)
segments=101;   %apodization segments

%initialize cdc
cdc = ApodizedContraDC;
cdc.alpha=0;
cdc.kch=0;
cdc.rch=0;
cdc.kappaMin=0;
cdc.period=1550e-9/2.5/2;
cdc.neffwg1=2.6;
cdc.neffwg2=2.4;
cdc.Dneffwg1=-1.0e6;
cdc.Dneffwg2=-1.1e6;
cdc.centralWL=1550*10^-9; %centre de la dispersion Dneff
cdc.starting_wavelength=startWL;
cdc.ending_wavelength=endWL;
cdc.resolution=res;
cdc.N_seg=segments;
cdc.name=cat(2,   'CDC for NN' );
cdc.display_progress=0;

for it=1:batchSize
    pRandA=1+rand*10;
    pRandKappaMax=500+rand*49500;
    pRandN=100+round(rand*2900);
    lambdaB=1525+rand*50;

    customSettings=0;
    if(customSettings)
        pRandA=5;
        pRandKappaMax=10000;
        pRandN=1000;
        lambdaB=1575;
    end

    pRandLCH=(-lambdaB/1550+1)*2;

    %set parameters
    cdc.a=pRandA;
    cdc.kappaMax=pRandKappaMax;
    cdc.N_Corrugations=pRandN;
    cdc.lch=pRandLCH;

    %calculate the cdc
    cdc=cdc.update;
    %save the results
    cdcLine{it}=[cdc.a cdc.N_Corrugations cdc.kappaMax lambdaB*1e-9 cdc.kappa_apod cdc.chirpDev*cdc.period real(cdc.E_Drop) imag(cdc.E_Drop)];
    disp([it cdcLine{it}(1:4).*[1 1/100 1/1000 1e6]])
end

toPlot=0;
if(toPlot)
    cdc.plotKappa;
    figure;
    plot(cdc.period*cdc.chirpDev*1e9*2*2.5);
    figure;
    plot(cdc.Lambda,cdc.drop)
    hold all;
    yyaxis right
    plot(cdc.Lambda,cdc.dropGroupDelay)
end

%save batch
filename=cat(2,'nnData/cdcDataBatch',num2str(batchNum));

fileID = fopen(cat(2,filename,'.txt'),'w');
header='a (float), N (int), kappa (float), lambdaB (float), apodization (1 X 101), period (1 X 101), real(E_drop) (1 X 1001), imag(E_drop) (1 X 1001)';
fprintf(fileID,header);
notation = '%1.10e ';
formatSpec=cat(2,'\n',repmat(notation,1,3),repmat(notation,1,202),repmat(notation,1,2002),notation);

[~,nrows] = size(cdcLine);
for row = 1:nrows
    fprintf(fileID,formatSpec,cdcLine{row});
end
fclose(fileID);

disp('BatchDone')
