%%
% 
%  I dedicate this work to my son Lokmane
% 

%%
% 
% <<ELM_TB.png>>
% 
 
clear all;clc
addpath('codes','dataset');
%% Load data
D=load('spambase.data');
A=D(:,1:57);             % Inputs
B=D(:,58);               % Targets
%% define Options
Opts.ELM_Type='Class';    % 'Class' for classification and 'Regrs' for regression
Opts.number_neurons=10;  % Maximam number of neurons 
Opts.Tr_ratio=0.70;       % training ratio
Opts.Bn=1;                % 1 to encode  lables into binary representations
                          % if it is necessary
%% Training
[net]= elm_LB(A,B,Opts);
 net
%% prediction
[output]=elmPredict(net,A);








