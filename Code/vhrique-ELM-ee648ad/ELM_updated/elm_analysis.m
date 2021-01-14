%ELM 
clc;clearvars
rng(42)
load id_Moneyness_SmallGrid_data_price_norm_70387_bigprice.mat
sample_idx = 1:100;
x_clean = scaledata(data_price(sample_idx,15:end),0,1);
y_clean = scaledata(data_price(sample_idx,1:5),0,1);
N1=min(min(data_price(sample_idx,15:end)));                 % save for denormalization
N2=max(max(data_price(sample_idx,15:end)));                 % save for denormalization

N = size(x_clean,1);
n_rep = 1;%500; %number of MV copies per surface
x = zeros(n_rep*N,size(x_clean,2)); % prices
y = zeros(n_rep*N,size(y_clean,2)); % paramters
x_mv = zeros(n_rep*N,size(x_clean,2));%missing value matrix
O =  zeros(n_rep*N,size(x_clean,2)); %observed indecee matrix
M =  zeros(n_rep*N,size(x_clean,2)); % missing value indeccee matrix
random_order = 0;%1; %boolean value indicating if order will be permuted
counter = 0;
idx_order = randperm(81);

% This loop creates the missing values
for i =1:N
    for j=1:n_rep
        counter = counter+1;
        x(n_rep*(i-1)+j,:)  = x_clean(i,:);
        y(n_rep*(i-1)+j,:) = y_clean(i,:);
        num_mv = randi([10,40]);  %number of missing values per surface
        idx = randperm(81);
        idx_mv = sort(idx(1:num_mv));
        idx_keep = sort(idx(num_mv+1:end));
        x_mv(counter,idx_keep) =x(counter,idx_keep);
        O(counter,idx_keep) = ones(size(idx_keep));
        M(counter,idx_mv) = ones(size(idx_mv));
        if random_order
            x(counter,:) = x(counter,idx_order);
            x_mv(counter,:) = x_mv(counter,idx_order);
            O(counter,:) = O(counter,idx_order);
            M(counter,:) = M(counter,idx_order);
        end
    end

end
A=x_mv;             % Inputs
B=x;               % Targets

%% define Options
Opts.number_neurons= 70;     % Maximam number of neurons 
sigmoid = @(x)1./(1+exp(-x));
Opts.activation =@(x) radbas(x);                            
Opts.number_runs =100; %number of random repititions
Opts.N1 =N1;  
Opts.N2 =N2;
Opts.norm =2; %2 normtype
%% Training
[net]= ELM_short(A,B,Opts);%ELM_test(A,B,Opts);
MAPE = net.mape;