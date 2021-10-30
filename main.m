clc;clear;

load data

feats=xxx(:,1:89);
label=xxx(:,90);

for i=1:4200
   Feats{i}=feats(i,:);
end
Feats=Feats';
labels=double(label);

data = Feats ;  % your data   
labels1=categorical(label);

> train-test splitting
%cv  = cvpartition(size(data,1), "HoldOut", 0.2);
%train_split=cv.training;
%test_split=cv.test;
load train_split
load test_split
Feats_train = data(train_split, :);
labels_train = labels1(train_split, :);
Feats_test = data(test_split, :);
labels_test = labels1(test_split, :);


%% ** Model Setup **
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

%% ** Parameter setting **
 options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 0.0001, ...
    'SequenceLength', 89, ...
     'Verbose',false);

%% ** Training of the proposed parallel based RNNs model **
net = trainNetwork(Feats_train,labels_train,lgraph,options);


%% ** The second stage of the proposed model **
% ** Feature Extraction **
for i=1:length(Feats_train)
value=Feats_train{i};
layer='addition';
feat_train(:,i) = activations(net,value,layer);
end

for i=1:length(Feats_test)
value=Feats_test{i};
layer='addition';
feat_test(:,i) = activations(net,value,layer);
end

% ** Classification with SVM method **
featuresTrain=feat_train';
YTrain=double(labels_train);
featuresTest=feat_test';
YTest=double(labels_test);

template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder',3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
    
classifier = fitcecoc(featuresTrain,YTrain,...
    'Learners', template,...
    'Coding', 'onevsall');

% ** Testing of proposed parallel RNNs based SVM model **
YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == YTest);
fprintf('Result: %d', accuracy)
