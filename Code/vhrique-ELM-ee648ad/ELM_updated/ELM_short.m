function [net]=ELM_short(X,Y,Opts)
%%% save the important data characteristics before normalization
N1=Opts.N1;                 % save for denormalization
N2=Opts.N2;                 % save for denormalization
%%% initialization
number_neurons=Opts.number_neurons; % get number of neurons
num_runs = Opts.number_runs; %number of random draws
activation = Opts.activation; %activation function as function
net.activation = activation;
net.X=X;                % scaled training inputs
net.Y=Y;                % training targets
net.IW ={};
net.OW ={};
Y_hat=0;

for i =1:num_runs
    %%%% 1st step: generate a random input weights
    input_weights=0.3*(rand(number_neurons,size(X,2))*2-1);
    net.IW{i}=input_weights;    % save Input weights
    %%%% 2nd step: calculate the hidden layer
    ixX= input_weights*X';
    ceta=0;
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
    Y_hat=Y_hat+(H' * B);
end
Y_hat = Y_hat/num_runs;
%net.Ymean =mean(Y);
Y_hat=scaledata(Y_hat,N1,N2);        % denormalization
Y = scaledata(Y,N1,N2);
net.Y = Y;
net.Y_hat=Y_hat;  
% save data
net.mape = reshape(100*mean(abs((Y-Y_hat)./Y),1),9,9)';
net.min=N1;             % save the min value of Targets
net.max=N2;             % save the max value of Targets
net.Opts=Opts;          % save options
net.num_runs = num_runs;
end
