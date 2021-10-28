clc;
clear;

load fea_new2

feats=xxx(:,1:89);
label=xxx(:,90);

for i=1:4200
   Feats{i}=feats(i,:);
end
Feats=Feats';
labels=double(label);

%%
accuracy=0;
while accuracy<0.96
data = Feats ;  % your data   
labels1=categorical(label);
% [m,n] = size(data) ; 
% 
% [trainInd,valInd,testInd] = dividerand(4200,0.8,0.1,0.1);
% % load trainInd
% % load valInd
% % load testInd
% Feats_train = data(trainInd,:) ;  
% Feats_test = data(testInd,:) ;
% Feats_val = data(valInd,:) ;
% % 
% labels_train = labels1(trainInd,:) ;  
% labels_test = labels1(testInd,:) ;
% labels_val = labels1(valInd,:) ;

%%

% cv  = cvpartition(size(data,1), "HoldOut", 0.2);
% 
% train_split=cv.training;
% test_split=cv.test;
load train_split
load test_split
Feats_train = data(train_split, :);
labels_train = labels1(train_split, :);
Feats_test = data(test_split, :);
labels_test = labels1(test_split, :);


%%
inputSize = 1;
numClasses = 4;
lgraph = layerGraph();
tempLayers = sequenceInputLayer(1,"Name","sequence");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    gruLayer(300,"Name","gru","OutputMode","last")
    dropoutLayer(0.2,"Name","dropout_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    lstmLayer(300,"Name","lstm","OutputMode","last")
    dropoutLayer(0.2,"Name","dropout_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    fullyConnectedLayer(4,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"sequence","gru");
lgraph = connectLayers(lgraph,"sequence","lstm");
lgraph = connectLayers(lgraph,"dropout_1","addition/in1");
lgraph = connectLayers(lgraph,"dropout_2","addition/in2");

 options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 0.0001, ...
    'SequenceLength', 89, ...
     'Verbose',false);
%       'Plots','training-progress',... 

%      'ExecutionEnvironment','multi-gpu', ...

net = trainNetwork(Feats,labels1,lgraph,options);
[YPred_rnn,scores] = classify(net,Feats_test);
accuracy = mean(YPred_rnn == labels_test)

