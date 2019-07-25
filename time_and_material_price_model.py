"""
Statistical Model of Platform Cost https://jira.intgdc.com/browse/NPR-4
Stanislav Vohnik
2019-07-25
Model version 'Time and Material'
"""
from re import sub
from pickle import dump
import pandas_profiling
from pandas import HDFStore, Series, DataFrame
from numpy import log, exp
from theano import shared, dot
from pymc3 import Model, Normal, HalfNormal, sample, traceplot, sample_posterior_predictive,\
summary as df_summary, StudentT, Deterministic, Exponential
from gooddata import Gdc, Metadatafactory, Datafactory

##################################
# Defaults and Attr, tiers Scoping
##################################
ATTS = Series({0:'workspace_size_gb',
               1:'etl_input_size_gb',
               2:'no_etl_integrations',
               3:'etl_task_time_last7d_sec',
               4:'sli_task_time_last7d_sec',
               5:'sj_task_time_last7d_sec',
               6:'avg_etl_upload_data_mb',
               7:'no_distinct_users_last7d',
               8:'all_size_gb',
               9:'comp_time'
               })
MEASURE = 'workspace_cost_simple'

# select ATT to use in model by a number
ATT = Series([ATTS[8], ATTS[5], ATTS[3]]).values
#To run all variables use
#ATT = ATTS.values

##############################################################
# Trace parameters influence quality of posterior distribution
# and Sigma as well
##############################################################
TUNE = 10000
SAMPLES = 500

#LIMIT for zeros
LIMIT = 1e-6 #  we will se it in plots

#Files base name
F_ATT = '-'.join([str(a) for a, b in ATTS.items() if b in ATT])
F_BASENAME = f"-({LIMIT})-{SAMPLES}-{TUNE}-{F_ATT}-{MEASURE}"

##############################################
# Login into GDC to get Current Variables and Report data from GDC Workspace
# Load the DATASET, LIMIT/cat zero values and Profile input data
##############################################
GDC = Gdc()
GDC.login('stanislav.vohnik+PP@gooddata.com')

######################################################
# Downloading Variables Report Data From GDC Workspace
######################################################
PROMPTS = Metadatafactory(GDC).get_prompts('y2rdzf5tapkpolaog5nkbxcw9nk9j3qf')
PP = Datafactory(GDC).get_report_data('/gdc/md/y2rdzf5tapkpolaog5nkbxcw9nk9j3qf/obj/5761')
PP.fillna(LIMIT, inplace=True)
PP.replace(to_replace=0.0, value=LIMIT, inplace=True)
MRR = Datafactory(GDC).get_report_data('/gdc/md/y2rdzf5tapkpolaog5nkbxcw9nk9j3qf/obj/5826')
MRR.fillna(LIMIT, inplace=True)
MRR.replace(to_replace=0.0, value=LIMIT, inplace=True)
print('Done Downloading Reports Data From GDC Workspace')

#Create Order Out of Chaos
COLUMNS = {i:sub(r'[\s\W\.]', '_', i).lower() for i in sorted(PP.COLUMNS)}
PP.rename(COLUMNS=COLUMNS, inplace=True)
PP.rename(COLUMNS={'workspace_cost___v4':'workspace_cost'}, inplace=True)

#Computation of another metric "Workspace Cost Simple"
PP['comp_time'] = PP.sj_task_time_last7d_sec+PP.etl_task_time_last7d_sec+PP.sli_task_time_last7d_sec
_ALL_COMP_BY_MONTH = PP.comp_time.sum()
_COMPUTATION_COST = PROMPTS.get('CPU Platform Costs') * PP['comp_time'] / _ALL_COMP_BY_MONTH
_COST_STORAGE = (PP.workspace_size_gb + PP.etl_input_size_gb) * PROMPTS.get('Storage Price per GB')
PP['workspace_cost_simple'] = _COMPUTATION_COST + _COST_STORAGE
PP['all_size_gb'] = PP.workspace_size_gb + PP.etl_input_size_gb

#Setting proper datatypes
NUMERICAL = PP.select_dtypes(include=['float', 'bool'])
CATEGORICAL = list(set(PP.COLUMNS) - set(NUMERICAL))
DTYPE = {**{i:'float32' for i in NUMERICAL},
         **{i:'str' for i in CATEGORICAL}}
PP = PP.astype(DTYPE)

#Saving profiling of imput data
pandas_profiling.ProfileReport(PP).to_file(outputfile=f"ProfileReport_{F_BASENAME}.html")


#Print For reference
print(ATT, 'F_BASENAME:', F_BASENAME)

#Assigning input/output LOG or not LOG ;-)
#we normalized it o bites an setting minimum for times to 1s 
X = log(PP[ATT]).copy()
X['all_size_gb'] = X['all_size_gb'] + log(10)*6
X['all_size_gb'] = X['all_size_gb'].astype('float32')
X['sj_task_time_last7d_sec'] = X['sj_task_time_last7d_sec'] + log(1)
X['sj_task_time_last7d_sec'] = X['sj_task_time_last7d_sec'].astype('float32')
X['etl_task_time_last7d_sec'] = X['etl_task_time_last7d_sec'] + log(1)
X['etl_task_time_last7d_sec'] = X['etl_task_time_last7d_sec'].astype('float32')
X_INPUT = shared(X.values)# numpy array
Y = log(PP[MEASURE]).copy()
Y_OUTPUT = shared(Y.values)# numpy array

#######################################
# Model definition
# MEASURE = exp(ALPHA)*exp(BETA)**log(X)
# mu = log(MEASURE) = ALPHA+BETA*log(X)
#######################################

with Model() as cost_model:
    # Priors for unknown cost model parameters
    ALPHA = Normal('ALPHA', mu=0, sigma=1000)
    BETA = Normal('BETA', mu=0, sigma=1000, shape=len(ATT))
    SIGMA = HalfNormal('SIGMA', sigma=100)

    # Model
    MU = ALPHA + dot(X_INPUT, BETA)
    NU = Deterministic('NU', Exponential('nu_', 1/29))

    # Likelihood (sampling distribution) of observations
#     Y_OBS = Normal('Y_OBS', mu=mu, sigma=sigma, observed=Y_OUTPUT)
    Y_OBS = StudentT('Y_OBS', mu=MU, sigma=SIGMA, observed=Y_OUTPUT, nu=NU)

with cost_model:
    TRACE = sample(SAMPLES, tune=TUNE, cores=6)
    traceplot(TRACE)

with cost_model:
    Y_PRED = sample_posterior_predictive(TRACE, 1000, cost_model)
    Y_ = Y_PRED['Y_OBS'].mean(axis=0)
    PP['model_cost'] = exp(Y_)  # depends on imput/output
    SUMMARY = df_summary(TRACE)

with open('Time_and_Material_cost_model.pkl', 'wb') as f:
    dump({'model': cost_model, 'TRACE': TRACE}, f)

PROMPTS['F_BASENAME'] = F_BASENAME
with HDFStore('Time_and_Material_pricing_version_fp_2.h5') as store:
    store['PP'] = PP
    store['X'] = X
    store['Y'] = PP[MEASURE]
    store['MRR'] = MRR
    store['PROMPTS'] = DataFrame(PROMPTS, index=[1])
    store['SUMMARY'] = SUMMARY

_DELTA = 100*(1-(exp(Y_).sum()/PP[MEASURE].sum()))
print('*'*80+'\n'+'*'*80)
print(f"SAMPLES={SAMPLES}; tune={TUNE}; attributes: '{', '.join(ATT)}'")
print(f"'ModelCost:{(exp(Y_).sum())}, 'ObservedCost': {(PP[MEASURE].sum())}, 'delta':{_DELTA}%")
print('*'*80+'\n'+'*'*80)
print(SUMMARY)
