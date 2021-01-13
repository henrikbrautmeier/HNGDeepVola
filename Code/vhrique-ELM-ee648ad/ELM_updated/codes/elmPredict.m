function [output]=elmPredict(net,samples)
%% 
% the function estimate the unknown Targets of samples.
% samples : input samples
% net     : the trained network.
%%% save the important data caracteristics befor normalization
    %%%%    Author:         TAREK BERGHOUT
    %%%%    UNIVERSITY:     BATNA 2, ALGERIA
    %%%%    EMAIL:          berghouttarek@gmail.com
    %%%%    last update:    03/09/2019.day/month/year
%% get options
number_neurons=net.Opts.number_neurons; % get number of neurons
Bn=net.Opts.Bn;                         % transform lables into binary codes
N1=net.min;                             % get denormalizing values
N2=net.max;
% get denormalizing values
input_weights=net.IW;
num_runs = net.num_runs;
B=net.OW;
output=[];
%samples=[ones(size(samples,1),1),samples];
for i =1:num_runs
    %% Activation
    H=net.activation(input_weights{i}*samples');
    H=[ones(1,size(H,2));H];
    %% output
    if i==1
        output =(H' * B{i}) ;
    else
        output=output+(H' * B{i}) ;
    end
end
output=output/num_runs;%-repmat(net.Ymean,size(output,1),1);
%% Adjusting the output according to initial conditions
output=scaledata(output,N1,N2);               % denormalization
end