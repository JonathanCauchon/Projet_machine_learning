batchSize=20;
iniBatchNum=31;
endBatchNum=90;

for batchNum=iniBatchNum:endBatchNum
    generateDataForNN;
end

disp('=Completely Done=')