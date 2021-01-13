function [net]=ELM_test(Xdata,Ydata,Opts)
%%% save the important data characteristics before normalization
N1=Opts.N1;                 % save for denormalization
N2=Opts.N2;                 % save for denormalization
%%% initialization
number_neurons=Opts.number_neurons; % get number of neurons
num_runs = Opts.number_runs;
activation = Opts.activation;
App='Regression';
net.app=App;% save type of application
%%%% divide your data into training and testing sets according to training ratio
[X,Y,Xts,Yts]=divide_data(Opts.Tr_ratio,Xdata,Ydata);
%         X  :    training inputs
%         Y  :    training targets
%         Xts:    testing inputs
%         Yts:    testing targets
% save
%X=[ones(size(X,1),1),X];
%Xts =[ones(size(Xts,1),1),Xts];
net.X=X;                % scaled training inputs
net.Y=Y;                % training targets
net.Xts=Xts;            % scaled testing inputs
net.Yts=Yts;            % testing target

%%%% encode lables for classification only
net.activation = activation;
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
    %if i ==50
    %    figure(); plot(1:70,ixX(:,1:1000))
    %end
    ceta=0;
    %ceta=2*rand(number_neurons,1)-1;
    H=[ones(1,size(ixX,2));activation(ixX+ceta)];
    %%%% 3rd step: calculate the output weights beta
    if Opts.norm==2
        B=pinv(H') * Y ; % Moore-Penrose pseudoinverse of matrix
    elseif Opts.norm==1
        for i=1:size(Y,2)
            tmp=fitlm(H',Y(:,i),'RobustOpts','on');
            B(:,i)=tmp.Coefficients.Estimate;
        end
    end
    
    net.OW{i}=B;        % save output weights
    %%%% calculate the actual output of traning and testing 
    %Y_hat=Y_hat+([ones(1,size(H,2));H]' * B) ;
    Y_hat=Y_hat+(H' * B);
    Yts_hat=Yts_hat+([ones(1,size(activation(input_weights*Xts'+ceta),2));activation(input_weights*Xts'+ceta)]'*B);
end
Y_hat = Y_hat/num_runs;%-repmat(mean(Y),size(Y,1),1);
%net.Ymean =mean(Y);
Yts_hat=Yts_hat/num_runs;%-repmat(mean(Y),size(Y,1),1);
%%%% calculate the prefomance of training and testing
TrAccuracy=sqrt(mse(Y-Y_hat));       % RMSE for regression
TsAccuracy=sqrt(mse(Yts-Yts_hat));   % RMSE for regression
Y_hat=scaledata(Y_hat,N1,N2);        % denormalization
if ~isempty(Yts_hat)
    Yts_hat=scaledata(Yts_hat,N1,N2);% denormalization
end
net.Y_hat=Y_hat;                     % estimated training targets
net.Yts_hat=Yts_hat;                 % estimated testing targets
% save data
net.min=N1;             % save the min value of Targets
net.max=N2;             % save the max value of Targets
net.Opts=Opts;          % save options
net.tr_acc=TrAccuracy;  % training accuracy
net.ts_acc=TsAccuracy;  % testing accuracy
net.num_runs = num_runs;
end
