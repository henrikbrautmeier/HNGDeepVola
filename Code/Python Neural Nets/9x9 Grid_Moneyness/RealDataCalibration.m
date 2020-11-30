Maturity        = 10:30:250;
Moneyness       = 1.1:-0.025:0.9;
K               = 1./Moneyness;
S               = 2000;%1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
load("data_calib_real.mat")%load("data_calib_real_full.mat")%load("data_calib_real_convexhull.mat")
%% Concentate underlying Data
years     = 2010;
goals     = ["MSE"];%,"MAPE","OptLL"];
path_data = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Calibration Calloption/';

alldata = {};
k = 0;
for y = years
    for goal = goals
        k = k+1;
        file       = strcat(path_data,'params_options_',num2str(y),'_h0_calibrated_',goal,'_interiorPoint_noYield.mat');
        tmp        = load(file);
        alldata{k} = tmp.values;
        year_total(k) =y;
    end
end

alldata = {};
k = 0;
for y = years
    for goal = goals
        k = k+1;
        file       = strcat(path_data,'params_options_',num2str(y),'_h0_calibrated_',goal,'_interiorPoint_noYield.mat');
        tmp        = load(file);
        alldata{k} = tmp.values;
        year_total(k) =y;
    end
end
Ninputs = 0;
for j = 1:k
    for m = 1:length(alldata{1,j})
        if isempty(alldata{1,j}{1,m})
            continue
        end
        Ninputs = Ninputs+1;
        week_vec(Ninputs) = m;
        year_vec(Ninputs) =year_total(j);
        params(Ninputs,:) = alldata{1,j}{1,m}.hngparams;
        sig2_0(Ninputs)   = alldata{1,j}{1,m}.sig20; 
        yields(Ninputs,:) = alldata{1,j}{1,m}.yields;
    end
end
data =[];
weekyear =[];
path                = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Data/Datasets';
%path                = 'D:/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
%path                =  'C:/Users/TEMP/Documents/GIT/HenrikAlexJP/Data/Datasets';
useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
num_voladays        = 6; % if real vola, give the number of historic volas used (6 corresponds to today plus 5 days = 1week);
algorithm           = 'interior-point';% 'sqp'
stock_ind           = 'SP500';
for year = 2010
    path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
    load(path_);

    % load Interest rates
    % load the corresponding data
    if useYield
        path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
        txt = 'useYield';
    else
        path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
        txt = 'noYield';
    end
    load(path_vola);

    % if use realized volatility data then load the corresponding data


    bound                   = [100, 100];
    formatIn                = 'dd-mmm-yyyy';

    % start from the first Wednesday of 2015 and finish with the last Wednesday of 2015

    DateString_start        = strcat('01-January-',num2str(year));
    DateString_end          = strcat('31-December-',num2str(year));
    date_start              = datenum(DateString_start, formatIn);
    date_end                = datenum(DateString_end, formatIn);
    wednessdays             = (weekday(date_start:date_end)==4);
    Dates                   = date_start:date_end;
    Dates                   = Dates(wednessdays);



    % bounds for maturity, moneyness, volumes, interest rates
    Type                    = 'call';
    MinimumVolume           = 100;
    MinimumOpenInterest     = 100;
    IfCleanNans             = 1;
    TimeToMaturityInterval  = [8, 250];
    MoneynessInterval       = [0.9, 1.1];
    [OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
        TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
        TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
        OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
        TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);

    weeksprices             = week(datetime([OptionsStruct.date], 'ConvertFrom', 'datenum'));
    disp(length(unique(weeksprices)))
    year_idx = year*ones(1,length(weeksprices));
    idxj  = 1:length(unique(weeksprices));
    weakyear_tmp = [weeksprices;year_idx];
    weekyear = [weekyear,weakyear_tmp];
    data_year = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
    data = [data,data_year];
end
params_determ = params(2:51,:);
params_determ = [params_determ(:,2:4),params_determ(:,1),params_determ(:,5)];
pers_determ = params_determ(:,1).* params_determ(:,3).^2+ params_determ(:,2);
pers_calib = params_calib(:,1).* params_calib(:,3).^2+ params_calib(:,2);
mape_pers = 100*abs((pers_calib-pers_determ)./pers_determ);
mape_params = 100*abs((params_determ-params_calib)./params_determ);
mape_mean = mean(mape_params);
list =[];
for j=2:51
    data_week_tmp = data(:,and((weekyear(1,:) == j),(weekyear(2,:) == year)))';
    data_vec = data_week_tmp(:,[3,2,4]);
    r_cur = interp1([10:30:250], rates_calib(j-1,:)./252, data_vec(:,2)');
    prices_determ =price_Q_order(params_determ(j-1,:),data_vec,r_cur');
    prices_nn =price_Q_order(params_calib(j-1,:),data_vec,r_cur');
    mape_nn = 100*abs((prices_nn-data_week_tmp(:,1)')./data_week_tmp(:,1)');
    mape_determ = 100*abs((prices_determ-data_week_tmp(:,1)')./data_week_tmp(:,1)');
    short(j,:) = [mean(mape_nn),mean(mape_determ)];
    list{j} = [data_week_tmp(:,1)';prices_nn;prices_determ;mape_nn;mape_determ];
end
errors =[];
for j =2:51
    errors =[errors,list{1,j}([1,4:5,2:3],:)];
end


figure()
scatter(errors(1,:),errors(2,:),"x");hold on
scatter(errors(1,:),errors(3,:),"x");
xlim([0.37,400])
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
legend("MAPE NNvsObs","MAPE HNGvsObs");%"MAPE NNvsHNG")
figure()
plot(1:50,short(2:51,:))
set(gca, 'YScale', 'log')
legend("Weekly MAPE NNvsObs","Weekly MAPE HNGvsObs");%"MAPE NNvsHNG")
figure()
legend_vec = ["a","b","g","w","h0"];
for i =1:5
    subplot(6,1,i)
    plot(1:50,mape_params(:,i))
    if i==4
        set(gca, 'YScale', 'log')
    end
    legend(legend_vec(i));%"MAPE NNvsHNG")
end
subplot(6,1,6)
plot(1:50,mape_pers)
legend("persistency")


figure()
for i =1:5
    subplot(6,1,i)
    scatter(params_calib(:,i),short(2:end,1));hold on
    scatter(params_calib(:,i),short(2:end,2))

end
subplot(6,1,6)
scatter(pers_calib,short(2:end,1)); hold on
scatter(pers_calib,short(2:end,2));


% Theoretical Analysis: 
Maturity        = 10:30:250;
Moneyness       = 1.1:-0.025:0.9;
K               = 1./Moneyness;
S               = 2000;%1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
%missing_values =load("2010_interpolatedgrid_mv_convexhull.mat");
missing_values =load("2010_interpolatedgrid_mv.mat");

missing_values = missing_values.data_2;
true_values = reshape(missing_values(:,:,:,1),50,9,9);
missing_idx = reshape(missing_values(:,:,:,2),50,9,9);

list_net =[];
grids_mape = zeros(50,9,9);
grids_smallmape = zeros(50,9,9);
nanmode = 0;
underlying =unique(data(4,:));
for j=2:51
    Moneyness       = 1.1:-0.025:0.9;
    K               = 1./Moneyness;
    S               = underlying(j-1);%2000;%1;
    K               = K*S;
    data_vec = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';
    r_cur = reshape(repmat(rates_calib(i-1,:),length(Moneyness),1),[],1);
    prices_determ =reshape(price_Q_order(params_determ(j-1,:),data_vec,r_cur),9,9)';
    prices_nn =reshape(price_Q_order(params_calib(j-1,:),data_vec,r_cur),9,9)';
    mape_net = 100*abs((prices_determ-prices_nn)./prices_determ);
    mape_realHNG =  100*abs((prices_determ-reshape(true_values(j-1,:,:),9,9))./reshape(true_values(j-1,:,:),9,9));
    mape_realNN =  100*abs((reshape(true_values(j-1,:,:),9,9)-prices_nn)./reshape(true_values(j-1,:,:),9,9));
    mape_small = -NaN*ones(9,9);
    if nanmode
        mape_small(reshape(~isnan(true_values(j-1,:)),9,9)) = mape_net(reshape(~isnan(true_values(j-1,:)),9,9))
    else
        mape_small(reshape(true_values(j-1,:)~=-999,9,9)) = mape_net(reshape(true_values(j-1,:)~=-999,9,9))
        mape_realNN(reshape(true_values(j-1,:)==-999,9,9))= NaN;
        mape_realHNG(reshape(true_values(j-1,:)==-999,9,9))= NaN
    end
    list_net{j}.gridmape = mape_net;
    list_net{j}.smallgridmape = mape_small;
    grids_mape(j-1,:,:) = mape_net;
    grids_smallmape(j-1,:,:) = mape_small;
    g1(j-1,:,:) = mape_realHNG;
    g2(j-1,:,:) = mape_realNN;
end
figure
subplot(1,3,1)
heatmap(reshape(nanmean(grids_smallmape,1),9,9))
subplot(1,3,2)
heatmap(reshape(nanmean(g1,1),9,9))
subplot(1,3,3)
heatmap(reshape(nanmean(g2,1),9,9))