function [net]=ELM_test(Xdata,Ydata,Opts)
%%% save the important data characteristics before normalization
N1=min(min(Ydata));                 % save for denormalization
N2=max(max(Ydata));                 % save for denormalization
%%% initialization
number_neurons=Opts.number_neurons; % get number of neurons
num_runs = Opts.number_runs;
ELM_Type=Opts.ELM_Type;             % get Application Type
activation = Opts.activation;
Bn=Opts.Bn;                         % transform lables into binary codes
%%% Normalize your data according To ELM_Type
if ELM_Type=='Class';
Xdata=scaledata(Xdata,0,1);
App='Classification';
if Bn==1
net.bn='binary Targets';
end
else
Xdata=scaledata(Xdata,0,1);
Ydata=scaledata(Ydata,0,1); 
App='Regression';
end
net.app=App;% save type of application
%%%% divide your data into training and testing sets according to training ratio
[X,Y,Xts,Yts]=divide_data(Opts.Tr_ratio,Xdata,Ydata);
%         X  :    training inputs
%         Y  :    training targets
%         Xts:    testing inputs
%         Yts:    testing targets
% save

net.X=X;                % scaled training inputs
net.Y=Y;                % training targets
net.Xts=Xts;            % scaled testing inputs
net.Yts=Yts;            % testing target

%%%% encode lables for classification only
if ELM_Type=='Class' & Bn==1;
[BY,BYts,label]=encode_lables(Y',Yts');
Y=BY';
Yts=BYts';
end
net.IW ={};
net.OW ={};
Y_hat=0;
Yts_hat =0;
for i =1:num_runs
    if mod(i,100)==0
        disp(round(100*i/num_runs,1))
    end
    %%%% 1st step: generate a random input weights
    input_weights=0.3*(rand(number_neurons,size(X,2))*2-1);
    net.IW{i}=input_weights;    % save Input weights
    %%%% 2nd step: calculate the hidden layer
    ixX= input_weights*X';
    ceta=0;
    %ceta=2*rand(number_neurons,1)-1;
    H=activation(ixX+ceta);
    %%%% 3rd step: calculate the output weights beta
    B=pinv(H') * Y ; % Moore-Penrose pseudoinverse of matrix
    net.OW{i}=B;        % save output weights
    %%%% calculate the actual output of traning and testing 
    Y_hat=Y_hat+(H' * B) ;
    Yts_hat=Yts_hat+(activation(input_weights*Xts'+ceta)'*B);
end
Y_hat = Y_hat/num_runs;
Yts_hat=Yts_hat/num_runs;
%%%% calculate the prefomance of training and testing
if ELM_Type=='Regrs'
    TrAccuracy=sqrt(mse(Y-Y_hat));       % RMSE for regression
    TsAccuracy=sqrt(mse(Yts-Yts_hat));   % RMSE for regression
    Y_hat=scaledata(Y_hat,N1,N2);        % denormalization
    %Yts_hat=scaledata(Yts_hat,N1,N2);    % denormalization
    net.Y_hat=Y_hat;                     % estimated training targets
    %net.Yts_hat=Yts_hat;                 % estimated testing targets
else
    if Bn==1
    Y_hat=round(scaledata(Y_hat,0,1));      % adjust outputs normalization
    Yts_hat=round(scaledata(Yts_hat,0,1));  % adjust outputs normalization
    else
    Y_hat=round(scaledata(Y_hat,N1,N2));    % adjust outputs normalization
    Yts_hat=round(scaledata(Yts_hat,N1,N2));% adjust outputs normalization    
    end

     ClassificationRate_Training=0;

    for i = 1 : size(Y,1)
        [label_index_expected]=(Y(i,:));
        [label_index_actual]=(Y_hat(i,:));
        if label_index_actual==label_index_expected
            ClassificationRate_Training=ClassificationRate_Training+1;
        end
    end
    TrAccuracy=ClassificationRate_Training/size(Y,1);% classification rate   

     ClassificationRate_Testing=0;
    for i = 1 : size(Yts,1)
        [label_index_expected]=(Yts(i,:));
        [label_index_actual]=(Yts_hat(i,:));
        if label_index_actual==label_index_expected
            ClassificationRate_Testing=ClassificationRate_Testing+1;
        end
    end   
    TsAccuracy=ClassificationRate_Testing/size(Yts,1);% classification rate   
% decode  lables
if ELM_Type=='Class'& Bn==1
  [NBY,NBYts]=decode_lables(Y_hat',Yts_hat',label);
  net.Y_hat=NBY';        % estimated training targets
  net.Yts_hat=NBYts';    % estimated testing targets
  net.BnY_hat=Y_hat;     % binary labeles
  net.BnYts_hat=Yts_hat; % binary labeles
else
  net.Y_hat=Y_hat;       % estimated training targets
  net.Yts_hat=Yts_hat;   % estimated testing targets
end
end
% save data
net.min=N1;             % save the min value of Targets
net.max=N2;             % save the max value of Targets
net.Opts=Opts;          % save options
net.tr_acc=TrAccuracy;  % training accuracy
net.ts_acc=TsAccuracy;  % testing accuracy
net.num_runs = num_runs;
end
function y=sigmoid(x)
    y=1./(1+exp(-x));
end