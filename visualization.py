'''
Visualization of Cost Model
Stanislav Vohnik
2019-07-24
'''
from pickle import load
from seaborn import pairplot, kdeplot
import matplotlib.pyplot as plt
from pandas import HDFStore, concat, Series
from numpy import log
from pymc3 import summary as df_summary

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

ATTS = Series({0:'workspace_size_gb',
               1:'etl_input_size_gb',
               2:'no_etl_integrations',
               3:'etl_task_time_last7d_sec',
               4:'sli_task_time_last7d_sec',
               5:'sj_task_time_last7d_sec',
               6:'avg_etl_upload_data_mb',
               7:'no_distinct_users_last7d',
               8:'all_size_gb',
               9:'comp_time',
               10:'share'
               })
MEASURE = 'workspace_cost_simple_subs'
ATT = Series([ATTS[8], ATTS[10]]).values

#####################
# Load Data From Disk
#####################
# Loading DATA from disk previously saved by Pricing model

MODEL_PREFIX = 'subscription'

TIERS = ['data_power_tier', 'data_frequency_tier', 'workspace_size_bucket']

with HDFStore(f'{MODEL_PREFIX}_pricing_version_fp_2.h5') as store:
    PP = store['PP']
    X = store['X']
    Y = store['Y']
    Y_ = PP['model_cost']
    PROMPTS = store['PROMPTS']
    F_BASENAME = PROMPTS['F_BASENAME'].values[0]

with open(f'{MODEL_PREFIX}_cost_model.pkl', 'rb') as fn:
    DATA = load(fn)
    COST_MODEL, TRACE = DATA['model'], DATA['trace']



########################
# Model visualization
########################

#Plot Cost model KDE (ln x scale)
FIG2, _ = plt.subplots(1, 1, figsize=(13, 6))
plt.title(f'KDE Workstaion Cost {MODEL_PREFIX} Observation versus Model', fontsize=16)
kdeplot(log(Y), label='Observation')
kdeplot(log(Y_), label='Model')
plt.xlabel('Wrks Cost Ln()', fontsize=16)
plt.ylabel('Density', fontsize=16)
FIG2.savefig(f'Cost_model_KDE_{MODEL_PREFIX}_{F_BASENAME}.png')

#Plot TIERS visualization relationships
for tier in TIERS:
    cols = list(ATT) + [MEASURE, 'model_cost']
    ppp = log(PP[cols]).copy()
    ppp = concat([ppp, PP[tier]], axis=1)
    tvrf_name = f'Visualizing relationships-{MODEL_PREFIX}-{tier}-{F_BASENAME}.png'
    pairplot(ppp,
             hue=tier,
             height=3,
             kind='scatter'
             ).savefig(tvrf_name)

pairplot(ppp,
         height=3,
         kind='scatter',
         diag_kind='kde'
         ).savefig(f'Visualizing relationships-{MODEL_PREFIX}-{F_BASENAME}.png')

SUMMARY = df_summary(TRACE)
print(SUMMARY)
