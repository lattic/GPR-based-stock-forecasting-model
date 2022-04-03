clc;clear;close all;
%data process
data1 = readcell('adbe.csv');
stock1 = cell2mat(data1(:,2));
trainX = (1:1185)';
%trainX = (998:1185)';
testX = (1186:1227)';
trainY = stock1(1:1185);
%trainY = stock1(998:1185);
testYreal = stock1(1186:1227);
%gpr model
gprMdl = fitrgp(trainX, trainY, ...
    'KernelFunction','matern32','BasisFunction','pureQuadratic',...
    'FitMethod','sr','PredictMethod','fic', ...
    'Standardize',true,'ComputationMethod','v', ...
    'ActiveSetMethod','likelihood','Optimizer','quasinewton', ...
    'OptimizeHyperparameters','auto');
[testYpd,~,limit] = predict(gprMdl,testX);
Lower=limit(:,1);
Upper=limit(:,2);%testYpd预测值，limit为上限和下限
%计算误差
erravg=sum(abs(testYpd-testYreal)./testYreal)/length(testYreal);
disp('平均绝对误差为');disp(erravg);
% 计算测试集实际值在上下限的概率
y3=(testYreal-Lower>0)&(Upper-testYreal>0);
errarea=sum(y3)/length(y3);
disp('实际值在预测上下限区间的概率为');disp(errarea);
%作图
figure;
plot(trainX,trainY,'b');xlabel('时间/天');ylabel('收盘价格/美元')
hold on;
plot(testX,testYreal,'b');
plot(testX,testYpd,'m');
fill([testX;flipud(testX)], [Lower;flipud(Upper)],[0.93333, 0.83529, 0.82353],'edgealpha', '0', 'facealpha', '.5');
legend('train','testreal','testpd','uncertainty');