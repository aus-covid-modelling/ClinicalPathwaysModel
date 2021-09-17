function [AZ1_data,AZ2_data,P1_data,P2_data,M1_data,M2_data]=clinical_pathways_model_release(epi_input_file_name,EDconsult_cap)
% Developed on MATLAB 2019b by James Walker (13 September 2021)
%
%% Inputs:
% epi_input_file_name - The file name associated with the data to be input 
% into the clinical pathways model. The input file is a table of multiple
% simulations with the numbers of daily infections by symptom/asymptomatic 
% status, vaccination status (product and dose) and age (in 5 year age 
% brackets). Note: the code is sensetive to the labels for 
% symptomatic/asymptomatic status, vaccine type, age bracket and column 
% names. See the test file for the appropriate labels.
%
% EDconsult_cap - the ED consult capacity (the maximum number of daily
% arrivals to ED that can receive a consult and hence enter the hospital)

%% Outputs: 
% unvac_data - An object containing output data for unvaccinated people by day, simulation and age. 
% This includes the daily number of new infections, new symptomatic
% infections, first time arrivals to ED, excess demand at ED
% (the amount of arrivals to ED, including returning individuals, that do
% not get admitted), the ED loss (the number of people that arrive at ED 
% and are not admitted and do not return to ED)
% ED arrivals exceed demand), people occupying ICU beds, people occupying
% beds and deaths.
% AZ1_data - As above, but for individuals with a single dose of
% AstraZeneca
% AZ2_data - As above, but for individuals with a two doses of
% AstraZeneca
% P1_data - As above, but for individuals with a single dose of
% Pfizer
% P2_data - As above, but for individuals with a two doses of
% Pfizer
% M1_data - As above, but for individuals with a single dose of
% Moderna
% M2_data - As above, but for individuals with a two doses of
% Moderna

%% 
tic;

sim_table=readtable(epi_input_file_name,'ReadVariableNames',true);

%pulls symptom labels (ordering is asymptomatic and symptomatic)
assymp_symp = unique(sim_table.Symptomatic) %CHECK ORDERING IS CORRECT

%splitting table into symptomatic and asymptomatic groups
[~,n_cols]=size(sim_table);
asymp_table = sim_table(strcmp(sim_table.Symptomatic,assymp_symp(1)),[1:3,5:n_cols]);
symp_table=sim_table(strcmp(sim_table.Symptomatic,assymp_symp(2)),[1:3,5:n_cols]);


non_case_columns = 4;

days_full=symp_table.DateSymptoms;
day_vec=unique(days_full);
num_days = length(day_vec);

age_full = symp_table.AgeBracket;
age_strata = unique(age_full);

%sets age groups in chronological order
age_strata = {age_strata{1},age_strata{10},age_strata{[2:9,11:(end)]}} %CHECK THIS ORDERING IS CORRECT

vac_full = strcat(symp_table.VaccineAtInfection,num2str(symp_table.DosesAtInfection));
vac_strata = unique(vac_full);

% sets vaccines in order: unvvacinated, AZ1, AZ2, P1, P2, M1, M2
vac_strata = vac_strata([5,1,2,6,7,3,4]) %CHECK THIS ORDERING IS CORRECT

num_strata = length(vac_strata)*length(age_strata);

[~,x]=size(symp_table);
num_sims = x-non_case_columns;
num_sims

%% Incorporating the delay distribution (Gamma distribution binned into days and truncated for multinomial sampling)
max_delay = 14;
mean_delay = 4.0 * ones(num_strata, num_sims);
delay_shape = 1 * ones(num_strata, num_sims); 
rate_param = mean_delay ./ delay_shape;  

delay_pdf = zeros(num_strata, max_delay, num_sims);

%discretisation of the distribution
for z =1 : max_delay
    delay_pdf(:,z,:)= gamcdf(z,delay_shape, rate_param) - gamcdf(z-1,delay_shape, rate_param);
end

% normalise the distribution
for s = 1:num_strata
    for sim = 1:num_sims
        delay_pdf(s,:,sim) = delay_pdf(s,:,sim)./sum(delay_pdf(s,:,sim));
    end
end
delay_pdf=permute(delay_pdf,[3,1,2]);   


%% Recording symptomatic cases and potential delays to presentation
temp_delayed_cases = zeros(num_sims,num_strata,num_days+max_delay);
cases_matrix=zeros(num_sims,num_strata,num_days);
asymp_matrix = zeros(num_sims,num_strata,num_days);
for ii=1:num_days
    for vac = 1:length(vac_strata)
        for age = 1:length(age_strata)
            strata_day_index = find(strcmp(vac_full,vac_strata{vac}) & strcmp(age_full,age_strata{age}) & days_full==day_vec(ii));
            cases_matrix(:,age + (vac-1)*length(age_strata),ii) = table2array(symp_table(strata_day_index,(non_case_columns+1):end));
            asymp_matrix(:,age + (vac-1)*length(age_strata),ii) = table2array(asymp_table(strata_day_index,(non_case_columns+1):end));
            temp_delayed_cases(:,age + (vac-1)*length(age_strata),ii:(ii+max_delay-1))=squeeze(temp_delayed_cases(:,age + (vac-1)*length(age_strata),ii:(ii+max_delay-1)))+mnrnd(cases_matrix(:,age + (vac-1)*length(age_strata),ii),squeeze(delay_pdf(:,age + (vac-1)*length(age_strata),:)));
        end
    end
end
%days, sims, strata
cases_matrix = permute(cases_matrix, [3,1,2]);
asymp_matrix = permute(asymp_matrix, [3,1,2]);
temp_delayed_cases = permute(temp_delayed_cases, [3,1,2]);
delayed_cases = temp_delayed_cases(1:num_days,:,:);

%cut off unncessary days of 0 cases
day1_index=find(sum(asymp_matrix + cases_matrix,[2,3]),1); 
last_day = min(num_days,day1_index+400);

cases_matrix=cases_matrix(day1_index:last_day,:,:);
asymp_matrix = asymp_matrix(day1_index:last_day,:,:);
delayed_cases = delayed_cases(day1_index:last_day,:,:);


num_days = size(cases_matrix,1); 


% runs a minimum of 200 samples through the clinical pathways model 
% if fewer simulations files are provided, each input simulation will be
% run multiple times
min_samples = 200;
increase_sample=ceil(min_samples/num_sims);
if increase_sample>1
    cases_matrix = repmat(cases_matrix,[1,increase_sample,1]);
    asymp_matrix = repmat(asymp_matrix, [1,increase_sample,1]);
    delayed_cases = repmat(delayed_cases, [1,increase_sample,1]);
    num_sims = increase_sample*num_sims;
end
display(['the simulations were augmented to ', num2str(num_sims)])

hosp = zeros(size(cases_matrix));




%% Generating Hospitalisations

%Odds ratio of hospitalisation from VOC
OR = 1.42;

% Vaccine Ordering: None, AZ1, AZ2, P1, P2, M1, M2
% relative vaccination of hospitalised cases
rel_symp_vac = 1-[0, 0.33, 0.61, 0.33, 0.83, 0.33, 0.83];
rel_hosp_vac = 1-[0,0.69,0.86,0.71,0.87,0.71,0.87];
rel_hosp_given_symps =(rel_hosp_vac./rel_symp_vac); 

%baseline probability of hospitalisation
hosp_baseline = 0.75 * [0.039 0.001  0.006  0.009 0.026 0.040 0.042 0.045 0.050 0.074 0.138 0.198 0.247 0.414 0.638 1.000 0.873];


%scaled for VOC (from VOC odds ratio)
p_hosp_unvac = min((OR.*hosp_baseline./(1-hosp_baseline))./(1+OR.*(hosp_baseline./(1-hosp_baseline))),1);

%scaled for vac status
temp=(p_hosp_unvac')*rel_hosp_given_symps;
p_hosp = temp(:)';

for s = 1:num_strata
    hosp(:,:,s) = binornd(delayed_cases(:,:,s),p_hosp(s));
end

%delay distribution for 

infection_data = permute(hosp, [3, 2, 1]);


%% ICU probabilities

%relative ICU probabilities given preICU
voc_rel_ICU = 1.99; 
rel_vac_ICUtemp = 1-[0,0.69,0.86,0.71,0.87,0.71,0.87];
rel_vac_ICU =(rel_vac_ICUtemp./rel_hosp_vac); 

%baseline value
ICU_baseline =0.24 * [0.243 0.289 0.338 0.389 0.443 0.503 0.570 0.653 0.756 0.866 0.954 1.000 0.972 0.854 0.645 0.402 0.107];

%scaling for VOC
p_ICU_unvac =min((voc_rel_ICU * ICU_baseline .*hosp_baseline)./p_hosp_unvac,1);

%scaling for vac status
temp=(p_ICU_unvac')*rel_vac_ICU;
temp_p_preICU = temp(:)';

%prob of eventual ICU given hospitalised, age and vac status
p_preICU = (temp_p_preICU').* ones(num_strata, num_sims ,num_days);

%% Death on Ward probabilities

%relative death on ward parameters
voc_rel_death = 1.61; 
rel_vac_deathtemp = 1-[0, 0.69, 0.9, 0.71, 0.92, 0.71, 0.92];
rel_vac_death =(rel_vac_deathtemp./rel_hosp_vac);

%baseline value
death_ward_baseline = 0.46 * [0.039, 0.037, 0.035, 0.035, 0.036, 0.039, 0.045, 0.055, 0.074, 0.107, 0.157, 0.238, 0.353, 0.502, 0.675, 0.832, 1];

%scaling for VOC
p_death_ward_unvac = min((voc_rel_death .* death_ward_baseline .* (1-ICU_baseline).* hosp_baseline)./((1-p_ICU_unvac).*p_hosp_unvac),1);

%scaling for vac status
temp=(p_death_ward_unvac')*rel_vac_death;
temp_p_ward_death = temp(:)';

% probability of death on ward given the are on ward (not preICU), age and vaccine status
p_ward_death = (temp_p_ward_death').* ones(num_strata, num_sims);


%% Death in ICU probabilities

%relative death on ICU parameters
voc_rel_deathICU = 1.61;
rel_vac_deathICUtemp = 1-[0, 0.69, 0.9, 0.71, 0.92, 0.71, 0.92];
rel_vac_deathICU =(rel_vac_deathICUtemp./rel_vac_ICUtemp);

%baseline value
ICU_death_baseline = 0.67 * [0.282, 0.286, 0.291, 0.299, 0.310, 0.328, 0.353, 0.390, 0.446, 0.520, 0.604, 0.705, 0.806, 0.899, 0.969, 1.0, 0.918];

%scaling for voc
p_ICU_death_unvac =min((voc_rel_deathICU * ICU_death_baseline .* ICU_baseline.* hosp_baseline)./(p_ICU_unvac.*p_hosp_unvac),1);

%scaling for vac status
temp=(p_ICU_death_unvac')*rel_vac_deathICU;
temp_p_ICU_death = temp(:)';

% prob of death in ICU given in ICU, age and vac status
p_ICU_death = (temp_p_ICU_death').* ones(num_strata, num_sims);

%% Death postICU probabilities
voc_rel_deathpostICU = 1.61;
rel_vac_deathpostICUtemp = 1-[0, 0.69, 0.9, 0.71, 0.92, 0.71, 0.92];
rel_vac_deathpostICU =(rel_vac_deathpostICUtemp./rel_vac_ICUtemp);

postICU_death_baseline = 0.35 * [0.091 0.083 0.077 0.074 0.074 0.076 0.08 0.086 0.093 0.102 0.117 0.148 0.211 0.332 0.526 0.753 1.0];
p_postICU_death_unvac = min((voc_rel_deathpostICU * postICU_death_baseline .* (1-ICU_death_baseline) .* ICU_baseline.* hosp_baseline)./((1-p_ICU_death_unvac) .* p_ICU_unvac .* p_hosp_unvac),1);

temp=(p_postICU_death_unvac')*rel_vac_deathpostICU;
temp_p_postICU_death = temp(:)';

p_postICU_death = (temp_p_postICU_death').* ones(num_strata, num_sims);

save('transition_probs.mat','p_hosp_unvac','p_ICU_unvac','p_death_ward_unvac','p_ICU_death_unvac','p_postICU_death_unvac')

%% Los Parameters (using full posterior mean LoS conditional upon shape)
num_comps = 8;
mean_los_params = cell(num_comps,1);
shape_los_params = cell(num_comps,1);
max_los_params = cell(num_comps,1);

%ward to dc
mean_los_params{1} = 10.7 * ones(num_strata, num_sims);
shape_los_params{1} = 1 * ones(num_strata, num_sims);
max_los_params{1} = 60;

%ward to death
mean_los_params{2} = 10.3 * ones(num_strata, num_sims);
shape_los_params{2} = 2 * ones(num_strata, num_sims);
max_los_params{2} = 35;

%preICU
mean_los_params{3} = 2.5 * ones(num_strata, num_sims);
shape_los_params{3} = 1 * ones(num_strata, num_sims);
max_los_params{3} = 15;

%ICUtodeath
post_mean = 11.8;
upper = 12;
st_dev = (upper-post_mean)/(norminv(0.975,0,1));
mean_los_params{4} = normrnd(post_mean,st_dev,[1,num_sims]).*ones(num_strata,num_sims);
shape_los_params{4} = 2 * ones(num_strata, num_sims);
max_los_params{4} = 40;


%ICUtostepdowndeath
post_mean = 7;
upper = 8;
st_dev = (upper-post_mean)/(norminv(0.975,0,1));
mean_los_params{5} = normrnd(post_mean,st_dev,[1,num_sims]).*ones(num_strata,num_sims);
shape_los_params{5} = 1 * ones(num_strata, num_sims);
max_los_params{5} = 35;

%ICUtostepdowndc
post_mean = 15.6;
upper = 16.1;
st_dev = (upper-post_mean)/(norminv(0.975,0,1));
mean_los_params{6} = normrnd(post_mean,st_dev,[1,num_sims]).*ones(num_strata,num_sims);
shape_los_params{6} = 1 * ones(num_strata, num_sims);
max_los_params{6} = 75;

%stepdown_death
post_mean = 8.1;
upper = 10;
st_dev = (upper-post_mean)/(norminv(0.975,0,1));
mean_los_params{7} = normrnd(post_mean,st_dev,[1,num_sims]).*ones(num_strata,num_sims);
shape_los_params{7} = 1 * ones(num_strata, num_sims);
max_los_params{7} = 45;

%stepdown_dc
post_mean = 12.2;
upper = 12.4;
st_dev = (upper-post_mean)/(norminv(0.975,0,1));
mean_los_params{8} = normrnd(post_mean,st_dev,[1,num_sims]).*ones(num_strata,num_sims);
shape_los_params{8} = 2 * ones(num_strata, num_sims);
max_los_params{8} = 45;


%% parameters for ED at capacity

%orderng of consultations for ED
vac_pref = [1,2,4,6,3,5,7]'; %unvac, az1, p1, m1, az2, p2, m2
age_pref = [length(age_strata):(-1):1]; %oldest to youngest
strata_preferencing =[];
for ii=1:age_pref
   strata_preferencing = [strata_preferencing;age_pref(ii) + length(age_strata)*(vac_pref-1)];
end

%probability vector associated with returning to ED after no consults available
p_EDreturn = 0.5*ones(num_strata, num_sims); 



display('Data and parameter initialisation complete')
toc;


%% Parameters (these are made large matricies so that they can be sampled from priors later)
% vector of probabilities that individuals in a strata will need ICU given
% that they've presented at the hospital 
p_ICU_from_ED =repmat(0.0,num_strata,1) .* ones(num_strata, num_sims ,num_days);  % probability that a case needs ICU immediately upon arrival to ED (given that they will need ICU)



%% Precalculations for LOS distributions
tic;

los_pdf = cell(num_comps,1);
for comp = 1:num_comps
    rate_param = mean_los_params{comp} ./ shape_los_params{comp};
    
    los_pdf{comp} = zeros(num_strata, max_los_params{comp}, num_sims);

    %discretization of the distribution
    for z =1 : max_los_params{comp}
        los_pdf{comp}(:,z,:)= gamcdf(z,shape_los_params{comp}, rate_param) - gamcdf(z-1,shape_los_params{comp}, rate_param);
    end
    
    % normalise the distribution
    for s = 1:num_strata
        for sim = 1:num_sims
            los_pdf{comp}(s,:,sim) = los_pdf{comp}(s,:,sim)./sum(los_pdf{comp}(s,:,sim));
        end
    end
    
end

display('LoS distribution calculations complete')
toc;


%% ED presentations
%some pre-processing function for entry times into ICU (possible delays due
%to not enough consult availabilty), could potentially also do pre-processing to
%turn multiple compartmental information into presentations
tic;
[preICU_demand, ICU_demand, ward_demand, EDconsults_made, EDconsults_left, ED_loss, ED_excess_demand] = admittance_process(infection_data, p_preICU, p_ICU_from_ED, num_days, num_strata, num_sims, EDconsult_cap, p_EDreturn, strata_preferencing);

display('admittance process complete')
toc;

%determine the eventual trajectories of people entering in ICU (if anyone
%went directly to ICU)
ICU_death_demand =binornd(ICU_demand(:,:,1:num_days), repmat(p_ICU_death,[1,1,num_days]));
ICU_stepdown_demand = ICU_demand(:,:,1:num_days) - ICU_death_demand;
ICU_stepdowndeath_demand = binornd(ICU_stepdown_demand, repmat(p_postICU_death,[1,1,num_days]));
ICU_stepdowndc_demand = ICU_stepdown_demand - ICU_stepdowndeath_demand;


stepdown_dc_demand = zeros(num_strata, num_sims, num_days);
stepdown_death_demand = zeros(num_strata, num_sims, num_days);


%everything is over slighly more days than the forecast (this avoids the
%need to truncate results towards the end of the forecast horizon)
occupancy = cell(num_comps,1);
departures = cell(num_comps,1);
for comp = 1:num_comps
   occupancy{comp} = zeros(num_sims, num_days + max_los_params{comp}, num_strata);
   departures{comp} = zeros(num_sims, num_days + max_los_params{comp}, num_strata);
end

tic;
for t= 1:num_days
    %tic;

    %% Ward
    ward_to_death_demand=binornd(ward_demand(:,:,t),p_ward_death);
    ward_to_dc_demand = ward_demand(:,:,t) - ward_to_death_demand;
    
    %ward discharges
    [occupancy{1}, departures{1}] = los_function(ward_to_dc_demand, occupancy{1}, departures{1}, t, los_pdf{1}, max_los_params{1});
    
    %ward (non ICU) deaths 
    [occupancy{2}, departures{2}] = los_function(ward_to_death_demand, occupancy{2}, departures{2}, t, los_pdf{2}, max_los_params{2});
    
    % pre ICU ward
    [occupancy{3}, departures{3}, new_preICU_departures] = los_function(preICU_demand(:, :, t), occupancy{3}, departures{3}, t, los_pdf{3}, max_los_params{3});
    
    %update ICU demand
    %determine the eventual trajectories of people entering in ICU
    newICU_death_demand =binornd(new_preICU_departures(:, :, 1:num_days), repmat(p_ICU_death,[1,1,num_days]));
    newICU_stepdown_demand = new_preICU_departures(:, :, 1:num_days) - newICU_death_demand;
    newICU_stepdown_death_demand = binornd(newICU_stepdown_demand, repmat(p_postICU_death,[1,1,num_days]));
    newICU_stepdown_dc_demand = newICU_stepdown_demand - newICU_stepdown_death_demand;
    
    ICU_death_demand=ICU_death_demand+newICU_death_demand;
    ICU_stepdowndeath_demand = ICU_stepdowndeath_demand + newICU_stepdown_death_demand;
    ICU_stepdowndc_demand = ICU_stepdowndc_demand + newICU_stepdown_dc_demand;
    
    %% ICU
    %ICUdeath
    [occupancy{4}, departures{4}] = los_function(ICU_death_demand(:,:,t), occupancy{4}, departures{4}, t, los_pdf{4}, max_los_params{4});
    
    %ICU -> stepdown death
    [occupancy{5}, departures{5}, new_stepdown_death] = los_function(ICU_stepdowndeath_demand(:,:,t), occupancy{5}, departures{5}, t, los_pdf{5}, max_los_params{5});
    stepdown_death_demand = stepdown_death_demand + new_stepdown_death(:, :, 1:num_days);
    
    %ICU ->stepdown dc
    [occupancy{6}, departures{6}, new_stepdown_dc] = los_function(ICU_stepdowndc_demand(:,:,t), occupancy{6}, departures{6}, t, los_pdf{6}, max_los_params{6});
    stepdown_dc_demand = stepdown_dc_demand + new_stepdown_dc(:, :, 1:num_days);
    
    
    %% post ICU
    %death
    [occupancy{7}, departures{7}] = los_function(stepdown_death_demand(:,:,t), occupancy{7}, departures{7}, t, los_pdf{7}, max_los_params{7});
    
    %discharge
    [occupancy{8}, departures{8}] = los_function(stepdown_dc_demand(:,:,t), occupancy{8}, departures{8}, t, los_pdf{8}, max_los_params{8});
    
    if mod(t,25)==0
        display(['day ' ,num2str(t), ' complete.'])
        toc;
    end
end

toc;




%% Save all of the outputs by vaccine status into tables in the form: [simulation index, time index, output by age]

day_indices=1:num_days;
temp=[1:num_sims].*ones(length(day_indices),num_sims);
sim_index = temp(:);
cases_matrix=cases_matrix(day_indices,:,:);
asymp_matrix=asymp_matrix(day_indices,:,:);
infection_data=infection_data(:,:,day_indices);
preICU_demand=preICU_demand(:,:,day_indices);
ICU_demand=ICU_demand(:,:,day_indices);
ward_demand=ward_demand(:,:,day_indices);
ED_excess_demand=ED_excess_demand(:,:,day_indices);
ED_loss=ED_loss(:,:,day_indices);
for ii=1:8
    occupancy{ii}=occupancy{ii}(:,day_indices,:);
    departures{ii}=departures{ii}(:,day_indices,:);
end

vac_status_str = 1:length(age_strata);
unvac_data.infections = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str)+asymp_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.incidence = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.new_ED_arrivals = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(infection_data(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.new_admitted = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(preICU_demand(vac_status_str,:,:) + ICU_demand(vac_status_str,:,:) + ward_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.ED_excess_demand = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(ED_excess_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.ED_loss = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(ED_loss(vac_status_str,:,:), [3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.ICU_occupancy = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(occupancy{4}(:,:,vac_status_str) + occupancy{5}(:,:,vac_status_str) + occupancy{6}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.total_beds_occupied = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(occupancy{1}(:,:,vac_status_str)+occupancy{2}(:,:,vac_status_str)+occupancy{3}(:,:,vac_status_str)+occupancy{4}(:,:,vac_status_str)+occupancy{5}(:,:,vac_status_str)+occupancy{6}(:,:,vac_status_str)+occupancy{7}(:,:,vac_status_str)+occupancy{8}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
unvac_data.new_deaths = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(departures{4}(:,:,vac_status_str)+departures{7}(:,:,vac_status_str)+departures{2}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];

vac_status_str = (length(age_strata)+1):(2*length(age_strata));
AZ1_data.infections = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str)+asymp_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.incidence = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(cases_matrix(:,:,(length(age_strata)+1):(2*length(age_strata))),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.new_ED_arrivals = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(infection_data(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.new_admitted = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(preICU_demand(vac_status_str,:,:) + ICU_demand(vac_status_str,:,:) + ward_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.ED_excess_demand = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(ED_excess_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.ED_loss = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(ED_loss(vac_status_str,:,:), [3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.ICU_occupancy = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(occupancy{4}(:,:,vac_status_str) + occupancy{5}(:,:,vac_status_str) + occupancy{6}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.total_beds_occupied = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(occupancy{1}(:,:,vac_status_str)+occupancy{2}(:,:,vac_status_str)+occupancy{3}(:,:,vac_status_str)+occupancy{4}(:,:,vac_status_str)+occupancy{5}(:,:,vac_status_str)+occupancy{6}(:,:,vac_status_str)+occupancy{7}(:,:,vac_status_str)+occupancy{8}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
AZ1_data.new_deaths = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(departures{4}(:,:,vac_status_str)+departures{7}(:,:,vac_status_str)+departures{2}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];

vac_status_str = (2*length(age_strata)+1):(3*length(age_strata));
AZ2_data.infections = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str)+asymp_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.incidence = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(cases_matrix(:,:,(length(age_strata)+1):(2*length(age_strata))),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.new_ED_arrivals = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(infection_data(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.new_admitted = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(preICU_demand(vac_status_str,:,:) + ICU_demand(vac_status_str,:,:) + ward_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.ED_excess_demand = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(ED_excess_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.ED_loss = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(ED_loss(vac_status_str,:,:), [3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.ICU_occupancy = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(occupancy{4}(:,:,vac_status_str) + occupancy{5}(:,:,vac_status_str) + occupancy{6}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.total_beds_occupied = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(occupancy{1}(:,:,vac_status_str)+occupancy{2}(:,:,vac_status_str)+occupancy{3}(:,:,vac_status_str)+occupancy{4}(:,:,vac_status_str)+occupancy{5}(:,:,vac_status_str)+occupancy{6}(:,:,vac_status_str)+occupancy{7}(:,:,vac_status_str)+occupancy{8}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
AZ2_data.new_deaths = [sim_index,repmat((1:(num_days))',[num_sims,1]),reshape(permute(departures{4}(:,:,vac_status_str)+departures{7}(:,:,vac_status_str)+departures{2}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];

vac_status_str = (3*length(age_strata)+1):(4*length(age_strata));
P1_data.infections = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str)+asymp_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.incidence = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,(length(age_strata)+1):(2*length(age_strata))),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.new_ED_arrivals = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(infection_data(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.new_admitted = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(preICU_demand(vac_status_str,:,:) + ICU_demand(vac_status_str,:,:) + ward_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.ED_excess_demand = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_excess_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.ED_loss = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_loss(vac_status_str,:,:), [3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.ICU_occupancy = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{4}(:,:,vac_status_str) + occupancy{5}(:,:,vac_status_str) + occupancy{6}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.total_beds_occupied = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{1}(:,:,vac_status_str)+occupancy{2}(:,:,vac_status_str)+occupancy{3}(:,:,vac_status_str)+occupancy{4}(:,:,vac_status_str)+occupancy{5}(:,:,vac_status_str)+occupancy{6}(:,:,vac_status_str)+occupancy{7}(:,:,vac_status_str)+occupancy{8}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
P1_data.new_deaths = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(departures{4}(:,:,vac_status_str)+departures{7}(:,:,vac_status_str)+departures{2}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];

vac_status_str = (4*length(age_strata)+1):(5*length(age_strata));
P2_data.infections = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str)+asymp_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.incidence = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,(length(age_strata)+1):(2*length(age_strata))),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.new_ED_arrivals = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(infection_data(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.new_admitted = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(preICU_demand(vac_status_str,:,:) + ICU_demand(vac_status_str,:,:) + ward_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.ED_excess_demand = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_excess_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.ED_loss = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_loss(vac_status_str,:,:), [3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.ICU_occupancy = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{4}(:,:,vac_status_str) + occupancy{5}(:,:,vac_status_str) + occupancy{6}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.total_beds_occupied = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{1}(:,:,vac_status_str)+occupancy{2}(:,:,vac_status_str)+occupancy{3}(:,:,vac_status_str)+occupancy{4}(:,:,vac_status_str)+occupancy{5}(:,:,vac_status_str)+occupancy{6}(:,:,vac_status_str)+occupancy{7}(:,:,vac_status_str)+occupancy{8}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
P2_data.new_deaths = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(departures{4}(:,:,vac_status_str)+departures{7}(:,:,vac_status_str)+departures{2}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];

vac_status_str = (5*length(age_strata)+1):(6*length(age_strata));
M1_data.infections = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str)+asymp_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.incidence = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,(length(age_strata)+1):(2*length(age_strata))),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.new_ED_arrivals = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(infection_data(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.new_admitted = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(preICU_demand(vac_status_str,:,:) + ICU_demand(vac_status_str,:,:) + ward_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.ED_excess_demand = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_excess_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.ED_loss = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_loss(vac_status_str,:,:), [3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.ICU_occupancy = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{4}(:,:,vac_status_str) + occupancy{5}(:,:,vac_status_str) + occupancy{6}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.total_beds_occupied = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{1}(:,:,vac_status_str)+occupancy{2}(:,:,vac_status_str)+occupancy{3}(:,:,vac_status_str)+occupancy{4}(:,:,vac_status_str)+occupancy{5}(:,:,vac_status_str)+occupancy{6}(:,:,vac_status_str)+occupancy{7}(:,:,vac_status_str)+occupancy{8}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
M1_data.new_deaths = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(departures{4}(:,:,vac_status_str)+departures{7}(:,:,vac_status_str)+departures{2}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];

vac_status_str = (6*length(age_strata)+1):(7*length(age_strata));
M2_data.infections = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,vac_status_str)+asymp_matrix(:,:,vac_status_str),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.incidence = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(cases_matrix(:,:,(length(age_strata)+1):(2*length(age_strata))),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.new_ED_arrivals = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(infection_data(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.new_admitted = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(preICU_demand(vac_status_str,:,:) + ICU_demand(vac_status_str,:,:) + ward_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.ED_excess_demand = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_excess_demand(vac_status_str,:,:),[3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.ED_loss = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(ED_loss(vac_status_str,:,:), [3,2,1]),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.ICU_occupancy = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{4}(:,:,vac_status_str) + occupancy{5}(:,:,vac_status_str) + occupancy{6}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.total_beds_occupied = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(occupancy{1}(:,:,vac_status_str)+occupancy{2}(:,:,vac_status_str)+occupancy{3}(:,:,vac_status_str)+occupancy{4}(:,:,vac_status_str)+occupancy{5}(:,:,vac_status_str)+occupancy{6}(:,:,vac_status_str)+occupancy{7}(:,:,vac_status_str)+occupancy{8}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];
M2_data.new_deaths = [sim_index,repmat(((1:(num_days)))',[num_sims,1]),reshape(permute(departures{4}(:,:,vac_status_str)+departures{7}(:,:,vac_status_str)+departures{2}(:,:,vac_status_str),[2,1,3]),[length(day_indices)*num_sims,length(age_strata)])];



%% Admittance and LoS fuctions
function [preICU_demand, ICU_demand, ward_demand, consults_made, consults_left, loss, excess_demand] = admittance_process(infection_data, p_severe_ED, p_ICU_from_ED, num_days, num_strata, num_sims, ED_daily_consultation_capacity, p_EDreturn,strata_preferencing)
    %simple binomial split model for requiring ICU vs not requiring ICU

    %swaps around strata based on how the ED will preference consults when
    %near capacity
    [~,unsort]=sort(strata_preferencing);
    
    infection_data = infection_data(strata_preferencing,:,:);
    p_severe_ED = p_severe_ED(strata_preferencing,:,:);
    p_ICU_from_ED=p_ICU_from_ED(strata_preferencing,:,:);
    p_EDreturn=p_EDreturn(strata_preferencing,:);
    
    %pre-allocating variables
    consults_made=zeros(num_strata, num_sims, num_days);
    excess_demand = zeros(num_strata, num_sims, num_days);
    consults_left = ED_daily_consultation_capacity*ones(num_days,num_sims);
    return_presentations = zeros(num_strata, num_sims, num_days+1);
    loss = zeros(num_strata, num_sims, num_days);
    for t=1:num_days
       
        
        %total consult demand on day t
        ED_consult_demand = infection_data(: ,: , t) + return_presentations(:, :, t);
        
        %assigning consults according to strata order
        for s = 1:num_strata
            consults_made(s,:,t)=min(consults_left(t,:),ED_consult_demand(s,:));
            consults_left(t,:)=consults_left(t,:)-consults_made(s,:,t);
        end
        
        %number of people that miss out on a consultation at ED
        excess_demand(:, :, t) = ED_consult_demand - consults_made(:, :, t); 

        %number that return tomorrow (note the time independence so the number of returns conditional upon a full ED is geometrically distributed)
        % p_return_consult can be strata dependent 
        return_presentations(:, :, t+1) = binornd(excess_demand(:, :, t), p_EDreturn);
        
        %number that leave the system without entering the hospital at this time (they may die or
        %recover, but these calculations can be done later). 
        loss(:, :, t) = excess_demand(:, :, t) - return_presentations(:, :, t+1);
    end

    %note this currently does not increase severity for returning cases
    severe = binornd(consults_made, p_severe_ED);
    ICU_demand = binornd(severe, p_ICU_from_ED);
    preICU_demand = severe - ICU_demand;
    ward_demand = consults_made-severe;
    
    consults_made=consults_made(unsort,:,:);
    ICU_demand = ICU_demand(unsort,:,:);
    preICU_demand = preICU_demand(unsort,:,:);
    ward_demand = ward_demand(unsort,:,:);
    loss = loss(unsort,:,:);
    excess_demand = excess_demand(unsort,:,:);
end 


function [occupancy,departures,new_departures] = los_function(new, occupancy,  departures,t, los_pdf, max_stay)
    
    %function for sampling length of stay and adding to the recorded
    %occupancy, departures and new_departures (this is the same function for each compartment in the model)
    
    new=permute(new, [2,1]);
    los_pdf=permute(los_pdf,[3,2,1]);
    
    [num_sim, days, num_strata]=size(departures);
    new_departures=zeros(num_sim, days, num_strata);
    
    %they stay on ward for at least a day 
    time_range = (t+1):(t + max_stay);
    time_range2 = t:(t + max_stay);
    
    for s = 1:num_strata
    %parfor s = 1:num_strata
        new_los = mnrnd(new(:, s),los_pdf(:, :, s));
        new_departures(:, time_range, s) = new_los;
        departures(:, time_range, s) =departures(:, time_range, s) + new_los;
        occupancy(:, time_range2, s) = occupancy(:, time_range2, s) + new(:, s) - cumsum([zeros(num_sim,1),new_los],2);
    end

    new_departures = permute(new_departures, [3, 1, 2]);
    
end
end
