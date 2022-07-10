# -*- coding: utf-8 -*-
"""Shuffle_Normalized Y_ Quanreg _ OLS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qNoyfyncIsigXa9n5CrRtlsmSVGkkj-h

# Setup
"""

import pandas as pd
import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from scipy.stats import norm

from pylab import rcParams
import seaborn as sns

"""## Graph Options"""

!wget https://github.com/MaxGhenis/random/raw/master/Roboto-Regular.ttf -P /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf
mpl.font_manager._rebuild()

# Seaborn beauty

sns.set_style('white')
DPI = 200
mpl.rc('savefig', dpi=DPI)
mpl.rcParams['figure.dpi'] = DPI
mpl.rcParams['figure.figsize'] = 6.4, 4.8  # Default.
mpl.rcParams['font.sans-serif'] = 'Roboto'
mpl.rcParams['font.family'] = 'sans-serif'

#Set title text color to dark gray (https://material.io/color) not black.
TITLE_COLOR = '#212121'
mpl.rcParams['text.color'] = TITLE_COLOR

#Axis titles and tick marks are medium gray.
AXIS_COLOR = '#757575'
mpl.rcParams['axes.labelcolor'] = AXIS_COLOR
mpl.rcParams['xtick.color'] = AXIS_COLOR
mpl.rcParams['ytick.color'] = AXIS_COLOR

#sns.set_palette(sns.color_palette('Paired', len(QUANTILES)))
sns.set_palette(sns.color_palette('Paired'))
sns.color_palette("Paired", as_cmap=True)
# Set dots to a light gray
#dot_color = sns.color_palette('Blues', 3)[1]
dot_color = sns.color_palette("Paired", as_cmap=True)

"""# Data"""

from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_csv('/content/gdrive/MyDrive/datasets/df_2606.csv')
pd.set_option('display.max_columns', 50)
df=df.sort_values(by="Date").reset_index(drop=True)

# df=df.sort_values(by="Date").reset_index(drop=True)
# #df['DA_price_lag1'] = df['DA_price'].shift(1)
# #df = df[df['DA_price_lag1'].notna()]
# df.describe()

# df=df.sort_values(by="Date").reset_index(drop=True)
# train_df, test_df = train_test_split(df, test_size=0., shuffle=True)

# Normalization
# mean = train_df_var.mean(axis=0)
# std = train_df_var.std(axis=0)
# train_df_var = (train_df_var - mean) / std
# test_df_var = (test_df_var - mean) / std

# mean_label = train_labels.mean()
# std_label = train_labels.std()
# train_labels = (train_labels - mean_label) / std_label
# test_labels = (test_labels - mean_label) / std_label

"""# Model 1:
###IndexPrice ~ hours + seasons (0h and Spring as a base)
"""

y = df[['IndexPrice']]

# base: 0 and spring
pred_h_s = df[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                         '16', '17', '18', '19', '20', '21', '22', '23', 'autumn', 'summer', 'winter']]

X = sm.add_constant(pred_h_s)
X_1 = sm.add_constant(pred_h_s)

# Model
quantiles = np.arange(0.1,1,0.1)
models = []
params_1 = []

quantreg_1 = sm.QuantReg(y, X_1)

# for q in QUANTILES:
#   b = quantreg_3.fit(q=q)
#   print(b.summary())

for qt in quantiles:
#	print(qt)
	res = quantreg_1.fit(q = qt)
	params_1.append([qt, res.params['const'],
                res.params['1'], res.pvalues['1'], res.params['2'], res.pvalues['2'],
                res.params['3'], res.pvalues['3'], res.params['4'], res.pvalues['4'], res.params['5'], res.pvalues['5'],
                res.params['6'], res.pvalues['6'], res.params['7'], res.pvalues['7'], res.params['8'], res.pvalues['8'],
                res.params['9'], res.pvalues['9'], res.params['10'], res.pvalues['10'], res.params['11'], res.pvalues['11'],
                res.params['12'], res.pvalues['12'], res.params['13'], res.pvalues['13'], res.params['14'], res.pvalues['14'],
                res.params['15'], res.pvalues['15'], res.params['16'], res.pvalues['16'], res.params['17'], res.pvalues['17'],
                res.params['18'], res.pvalues['18'], res.params['19'], res.pvalues['19'], res.params['20'], res.pvalues['20'],
                res.params['21'], res.pvalues['21'], res.params['22'], res.pvalues['22'], res.params['23'], res.pvalues['23'],
                res.params['autumn'], res.pvalues['autumn'], res.params['summer'], res.pvalues['summer'], res.params['winter'], res.pvalues['winter']])

params_1 = pd.DataFrame(data = params_1, columns = ['qt','intercept',
                                                '1', '1_pvalue', '2', '2_pvalue', '3', '3_pvalue',
                                                '4', '4_pvalue', '5', '5_pvalue', '6', '6_pvalue', '7', '7_pvalue', '8', '8_pvalue',
                                                '9', '9_pvalue', '10', '10_pvalue', '11', '11_pvalue', '12', '12_pvalue', '13', '13_pvalue',
                                                '14', '14_pvalue', '15', '15_pvalue', '16', '16_pvalue', '17', '17_pvalue', '18', '18_pvalue',
                                                '19', '19_pvalue', '20', '20_pvalue', '21', '21_pvalue', '22', '22_pvalue', '23', '23_pvalue',
                                                'autumn', 'autumn_pvalue', 'summer', 'summer_pvalue', 'winter', 'winter_pvalue'])
#print(params)

plt.figure(0)
params_1.plot(x = 'qt', y = ['autumn', 'summer', 'winter'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(0)
params_1.plot(x = 'qt', y = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                         '16', '17', '18', '19', '20', '21', '22', '23'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

"""# Model 2:
### IndexPrice ~ hours + seasons + 'Wind_Onshore', 'Solar', 'Wind_Offshore', 'Load_forecast', 'DA_price’ (0h and Spring as a base)
"""

# base: 0 and spring
pred = df[['Wind_Onshore', 'Solar', 'Wind_Offshore', 'Load_forecast', 'DA_price', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                         '16', '17', '18', '19', '20', '21', '22', '23', 'autumn', 'summer', 'winter']]

X_2 = sm.add_constant(pred)

# Multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

#less than 10 should be ok?
#vif_data

## Build the model for other quantiles
quantiles = np.arange(0.1,1,0.1)
models = []
params_2 = []

quantreg_2 = sm.QuantReg(y, X_2)

for qt in quantiles:
#	print(qt)
	res = quantreg_2.fit(q = qt)
	models.append(res)
	params_2.append([qt, res.params['const'], res.params['Wind_Onshore'], res.pvalues['Wind_Onshore'], res.params['Solar'], res.pvalues['Solar'], res.params['Wind_Offshore'], res.pvalues['Wind_Offshore'],
	               res.params['Load_forecast'], res.pvalues['Load_forecast'], res.params['DA_price'], res.pvalues['DA_price'], res.params['1'], res.pvalues['1'], res.params['2'], res.pvalues['2'], res.params['3'], res.pvalues['3'], res.params['4'], res.pvalues['4'], res.params['5'], res.pvalues['5'], res.params['6'], res.pvalues['6'], res.params['7'], res.pvalues['7'], res.params['8'], res.pvalues['8'], res.params['9'], res.pvalues['9'], res.params['10'], res.pvalues['10'], res.params['11'], res.pvalues['11'], res.params['12'], res.pvalues['12'], res.params['13'], res.pvalues['13'], res.params['14'], res.pvalues['14'], res.params['15'], res.pvalues['15'], res.params['16'], res.pvalues['16'], res.params['17'], res.pvalues['17'], res.params['18'], res.pvalues['18'], res.params['19'], res.pvalues['19'], res.params['20'], res.pvalues['20'], res.params['21'], res.pvalues['21'], res.params['22'], res.pvalues['22'], res.params['23'], res.pvalues['23'], res.params['autumn'], res.pvalues['autumn'], res.params['summer'], res.pvalues['summer'], res.params['winter'], res.pvalues['winter']])


params_2 = pd.DataFrame(data = params_2, columns = ['qt','intercept','Wind_Onshore', 'Wind_Onshore_pvalue', 'Solar', 'Solar_pvalue', 'Wind_Offshore', 'Wind_Offshore_pvalue', 'Load_forecast',
                                                'Load_forecast_pvalue', 'DA_price', 'DA_price_pvalue', '1', '1_pvalue', '2', '2_pvalue', '3', '3_pvalue', '4', '4_pvalue', '5', '5_pvalue', '6', '6_pvalue', '7', '7_pvalue', '8', '8_pvalue', '9', '9_pvalue', '10', '10_pvalue', '11', '11_pvalue', '12', '12_pvalue', '13', '13_pvalue', '14', '14_pvalue', '15', '15_pvalue', '16', '16_pvalue', '17', '17_pvalue', '18', '18_pvalue', '19', '19_pvalue', '20', '20_pvalue', '21', '21_pvalue', '22', '22_pvalue', '23', '23_pvalue', 'autumn', 'autumn_pvalue', 'summer', 'summer_pvalue', 'winter', 'winter_pvalue'])

print(params_2)

## Plot the changes in the quantile coefficients
plt.figure(1)
params_2.plot(x = 'qt', y = ['Wind_Onshore', 'Solar', 'Wind_Offshore', 'Load_forecast'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(2)
params_2.plot(x = 'qt', y = 'DA_price', 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(3)
params_2.plot(x = 'qt', y = ['autumn', 'summer', 'winter'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

"""# Model 3:
### IndexPrice ~ hours + seasons + 'DA_price’+  Residual Load (0h and Spring as a base)
"""

# base: 0 and spring
pred = df[['DA_price', 'Residual_Load', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                         '16', '17', '18', '19', '20', '21', '22', '23', 'autumn', 'summer', 'winter']]

X_3 = sm.add_constant(pred)

# Model
quantiles = np.arange(0.1,1,0.1)
models = []
params_3 = []

quantreg_3 = sm.QuantReg(y, X_3)

# for q in QUANTILES:
#   b = quantreg_3.fit(q=q)
#   print(b.summary())

for qt in quantiles:
	print(qt)
	res = quantreg_3.fit(q = qt)
	params_3.append([qt, res.params['const'], res.params['Residual_Load'], res.pvalues['Residual_Load'],
                res.params['DA_price'], res.pvalues['DA_price'], res.params['1'], res.pvalues['1'], res.params['2'], res.pvalues['2'],
                res.params['3'], res.pvalues['3'], res.params['4'], res.pvalues['4'], res.params['5'], res.pvalues['5'],
                res.params['6'], res.pvalues['6'], res.params['7'], res.pvalues['7'], res.params['8'], res.pvalues['8'],
                res.params['9'], res.pvalues['9'], res.params['10'], res.pvalues['10'], res.params['11'], res.pvalues['11'],
                res.params['12'], res.pvalues['12'], res.params['13'], res.pvalues['13'], res.params['14'], res.pvalues['14'],
                res.params['15'], res.pvalues['15'], res.params['16'], res.pvalues['16'], res.params['17'], res.pvalues['17'],
                res.params['18'], res.pvalues['18'], res.params['19'], res.pvalues['19'], res.params['20'], res.pvalues['20'],
                res.params['21'], res.pvalues['21'], res.params['22'], res.pvalues['22'], res.params['23'], res.pvalues['23'],
                res.params['autumn'], res.pvalues['autumn'], res.params['summer'], res.pvalues['summer'], res.params['winter'], res.pvalues['winter']])

params_3 = pd.DataFrame(data = params_3, columns = ['qt','intercept','Residual_Load', 'Residual_Load_pvalue',
                                                'DA_price', 'DA_price_pvalue', '1', '1_pvalue', '2', '2_pvalue', '3', '3_pvalue',
                                                '4', '4_pvalue', '5', '5_pvalue', '6', '6_pvalue', '7', '7_pvalue', '8', '8_pvalue',
                                                '9', '9_pvalue', '10', '10_pvalue', '11', '11_pvalue', '12', '12_pvalue', '13', '13_pvalue',
                                                '14', '14_pvalue', '15', '15_pvalue', '16', '16_pvalue', '17', '17_pvalue', '18', '18_pvalue',
                                                '19', '19_pvalue', '20', '20_pvalue', '21', '21_pvalue', '22', '22_pvalue', '23', '23_pvalue',
                                                'autumn', 'autumn_pvalue', 'summer', 'summer_pvalue', 'winter', 'winter_pvalue'])
#print(params)

plt.figure(4)
params_3.plot(x = 'qt', y = 'Residual_Load', 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(5)
params_3.plot(x = 'qt', y = 'DA_price', 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(6)
params_3.plot(x = 'qt', y = ['autumn', 'summer', 'winter'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

# plt.figure(7)
# params.plot(x = 'qt', y = ['1', '2', '3'], 
# 	title = 'Slope for different quantiles', kind ='line')

"""# Model 4:
### Price_dev ~ hours + seasons
"""

y_diff = df[['Price_dev']]

# Model
quantiles = np.arange(0.1,1,0.1)
print(quantiles)
models = []
params_4 = []

quantreg_4 = sm.QuantReg(y_diff, X_1)

# for q in QUANTILES:
#   b = quantreg_3.fit(q=q)
#   print(b.summary())

for qt in quantiles:
	print(qt)
	res = quantreg_4.fit(q = qt)
	params_4.append([qt, res.params['const'],
                res.params['1'], res.pvalues['1'], res.params['2'], res.pvalues['2'],
                res.params['3'], res.pvalues['3'], res.params['4'], res.pvalues['4'], res.params['5'], res.pvalues['5'],
                res.params['6'], res.pvalues['6'], res.params['7'], res.pvalues['7'], res.params['8'], res.pvalues['8'],
                res.params['9'], res.pvalues['9'], res.params['10'], res.pvalues['10'], res.params['11'], res.pvalues['11'],
                res.params['12'], res.pvalues['12'], res.params['13'], res.pvalues['13'], res.params['14'], res.pvalues['14'],
                res.params['15'], res.pvalues['15'], res.params['16'], res.pvalues['16'], res.params['17'], res.pvalues['17'],
                res.params['18'], res.pvalues['18'], res.params['19'], res.pvalues['19'], res.params['20'], res.pvalues['20'],
                res.params['21'], res.pvalues['21'], res.params['22'], res.pvalues['22'], res.params['23'], res.pvalues['23'],
                res.params['autumn'], res.pvalues['autumn'], res.params['summer'], res.pvalues['summer'], res.params['winter'], res.pvalues['winter']])

params_4 = pd.DataFrame(data = params_4, columns = ['qt','intercept',
                                                '1', '1_pvalue', '2', '2_pvalue', '3', '3_pvalue',
                                                '4', '4_pvalue', '5', '5_pvalue', '6', '6_pvalue', '7', '7_pvalue', '8', '8_pvalue',
                                                '9', '9_pvalue', '10', '10_pvalue', '11', '11_pvalue', '12', '12_pvalue', '13', '13_pvalue',
                                                '14', '14_pvalue', '15', '15_pvalue', '16', '16_pvalue', '17', '17_pvalue', '18', '18_pvalue',
                                                '19', '19_pvalue', '20', '20_pvalue', '21', '21_pvalue', '22', '22_pvalue', '23', '23_pvalue',
                                                'autumn', 'autumn_pvalue', 'summer', 'summer_pvalue', 'winter', 'winter_pvalue'])
#print(params)

plt.figure(8)
params_4.plot(x = 'qt', y = ['autumn', 'summer', 'winter'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

"""# Model 5:
### Price_dev ~ hours + seasons + Residual_Load
"""

y_diff = df[['Price_dev']]

# base: 0 and spring
pred = df[['Residual_Load', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                         '16', '17', '18', '19', '20', '21', '22', '23', 'autumn', 'summer', 'winter']]

X_5 = sm.add_constant(pred)

# Model
quantiles = np.arange(0.1,1,0.1)
print(quantiles)
models = []
params_5 = []

quantreg_5 = sm.QuantReg(y_diff, X_5)

# for q in QUANTILES:
#   b = quantreg_3.fit(q=q)
#   print(b.summary())

for qt in quantiles:
	print(qt)
	res = quantreg_5.fit(q = qt)
	params_5.append([qt, res.params['const'], res.params['Residual_Load'], res.pvalues['Residual_Load'],
                res.params['1'], res.pvalues['1'], res.params['2'], res.pvalues['2'],
                res.params['3'], res.pvalues['3'], res.params['4'], res.pvalues['4'], res.params['5'], res.pvalues['5'],
                res.params['6'], res.pvalues['6'], res.params['7'], res.pvalues['7'], res.params['8'], res.pvalues['8'],
                res.params['9'], res.pvalues['9'], res.params['10'], res.pvalues['10'], res.params['11'], res.pvalues['11'],
                res.params['12'], res.pvalues['12'], res.params['13'], res.pvalues['13'], res.params['14'], res.pvalues['14'],
                res.params['15'], res.pvalues['15'], res.params['16'], res.pvalues['16'], res.params['17'], res.pvalues['17'],
                res.params['18'], res.pvalues['18'], res.params['19'], res.pvalues['19'], res.params['20'], res.pvalues['20'],
                res.params['21'], res.pvalues['21'], res.params['22'], res.pvalues['22'], res.params['23'], res.pvalues['23'],
                res.params['autumn'], res.pvalues['autumn'], res.params['summer'], res.pvalues['summer'], res.params['winter'], res.pvalues['winter']])

params_5 = pd.DataFrame(data = params_5, columns = ['qt','intercept', 'Residual_Load', 'Residual_Load_pvalue',
                                                '1', '1_pvalue', '2', '2_pvalue', '3', '3_pvalue',
                                                '4', '4_pvalue', '5', '5_pvalue', '6', '6_pvalue', '7', '7_pvalue', '8', '8_pvalue',
                                                '9', '9_pvalue', '10', '10_pvalue', '11', '11_pvalue', '12', '12_pvalue', '13', '13_pvalue',
                                                '14', '14_pvalue', '15', '15_pvalue', '16', '16_pvalue', '17', '17_pvalue', '18', '18_pvalue',
                                                '19', '19_pvalue', '20', '20_pvalue', '21', '21_pvalue', '22', '22_pvalue', '23', '23_pvalue',
                                                'autumn', 'autumn_pvalue', 'summer', 'summer_pvalue', 'winter', 'winter_pvalue'])
#print(params)

plt.figure(8)
params_5.plot(x = 'qt', y = ['autumn', 'summer', 'winter'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(8)
params_5.plot(x = 'qt', y = "Residual_Load", 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

"""# Model 6:
### Price_dev ~ hours + seasons + 'Wind_Onshore', 'Solar', 'Wind_Offshore', 'Load_forecast'
"""

y_diff = df[['Price_dev']]

# base: 0 and spring
pred = df[['Wind_Onshore', 'Solar', 'Wind_Offshore', 'Load_forecast', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                         '16', '17', '18', '19', '20', '21', '22', '23', 'autumn', 'summer', 'winter']]

X_6 = sm.add_constant(pred)

# Model
quantiles = np.arange(0.1,1,0.1)
print(quantiles)
models = []
params_6 = []

quantreg_6 = sm.QuantReg(y_diff, X_6)

# for q in QUANTILES:
#   b = quantreg_3.fit(q=q)
#   print(b.summary())

for qt in quantiles:
	print(qt)
	res = quantreg_6.fit(q = qt)
	params_6.append([qt, res.params['const'], res.params['Wind_Onshore'], res.pvalues['Wind_Onshore'], res.params['Solar'],
                res.pvalues['Solar'], res.params['Wind_Offshore'], res.pvalues['Wind_Offshore'],
                res.params['Load_forecast'], res.pvalues['Load_forecast'],
                res.params['1'], res.pvalues['1'], res.params['2'], res.pvalues['2'],
                res.params['3'], res.pvalues['3'], res.params['4'], res.pvalues['4'], res.params['5'], res.pvalues['5'],
                res.params['6'], res.pvalues['6'], res.params['7'], res.pvalues['7'], res.params['8'], res.pvalues['8'],
                res.params['9'], res.pvalues['9'], res.params['10'], res.pvalues['10'], res.params['11'], res.pvalues['11'],
                res.params['12'], res.pvalues['12'], res.params['13'], res.pvalues['13'], res.params['14'], res.pvalues['14'],
                res.params['15'], res.pvalues['15'], res.params['16'], res.pvalues['16'], res.params['17'], res.pvalues['17'],
                res.params['18'], res.pvalues['18'], res.params['19'], res.pvalues['19'], res.params['20'], res.pvalues['20'],
                res.params['21'], res.pvalues['21'], res.params['22'], res.pvalues['22'], res.params['23'], res.pvalues['23'],
                res.params['autumn'], res.pvalues['autumn'], res.params['summer'], res.pvalues['summer'], res.params['winter'], res.pvalues['winter']])

params_6 = pd.DataFrame(data = params_6, columns = ['qt','intercept', 'Wind_Onshore', 'Wind_Onshore_pvalue', 'Solar', 'Solar_pvalue', 'Wind_Offshore',
                                                'Wind_Offshore_pvalue', 'Load_forecast','Load_forecast_pvalue',
                                                '1', '1_pvalue', '2', '2_pvalue', '3', '3_pvalue',
                                                '4', '4_pvalue', '5', '5_pvalue', '6', '6_pvalue', '7', '7_pvalue', '8', '8_pvalue',
                                                '9', '9_pvalue', '10', '10_pvalue', '11', '11_pvalue', '12', '12_pvalue', '13', '13_pvalue',
                                                '14', '14_pvalue', '15', '15_pvalue', '16', '16_pvalue', '17', '17_pvalue', '18', '18_pvalue',
                                                '19', '19_pvalue', '20', '20_pvalue', '21', '21_pvalue', '22', '22_pvalue', '23', '23_pvalue',
                                                'autumn', 'autumn_pvalue', 'summer', 'summer_pvalue', 'winter', 'winter_pvalue'])
#print(params)

plt.figure(9)
params_6.plot(x = 'qt', y = ['autumn', 'summer', 'winter'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(10)
params_6.plot(x = 'qt', y = ['Wind_Onshore', 'Solar', 'Wind_Offshore', 'Load_forecast'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(9)
params_6.plot(x = 'qt', y = 'intercept', 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(9)
params_5.plot(x = 'qt', y = 'intercept', 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

plt.figure(9)
params_1.plot(x = 'qt', y = 'intercept', 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')

"""# Compare Quantile Loss

In the regression loss equation above, as q has a value between 0 and 1, the first term will be positive and dominate when under-predicting, yi > yip, and the second term will dominate when over-predicting, yi < yip. For q equal to 0.5, under-prediction and over-prediction will be penalized by the same factor, and the median is obtained. The larger the value of q, the more under-predictions are penalized compared to over-predictions. For q equal to 0.75, under-predictions will be penalized by a factor of 0.75, and over-predictions by a factor of 0.25. The model will then try to avoid under-predictions approximately three times as hard as over-predictions, and the 0.75 quantile will be obtained.

https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/
"""

def quantile_loss(q, y, f):
    # q: Quantile to be evaluated, e.g., 0.5 for median.
    # y: True value.
    # f: Fitted or predicted value.
    e = y - f
    return np.maximum(q * e, (q - 1) * e)

METHODS = ['Model_1', 'Model_2', 'Model_3']


preds = np.array([(method, q, x) 
                  for method in METHODS 
                  for q in QUANTILES
                  for x in X['const']])

preds = pd.DataFrame(preds)
preds.columns = ['method', 'q', 'x']
preds = preds.apply(lambda x: pd.to_numeric(x, errors='ignore'))

preds['label'] = np.resize(y, preds.shape[0])

preds.loc[preds.method == 'Model_1', 'pred'] = np.concatenate(
    [quantreg_1.fit(q=q).predict(X_1) for q in QUANTILES]) 

preds.loc[preds.method == 'Model_2', 'pred'] = np.concatenate(
    [quantreg_2.fit(q=q).predict(X_2) for q in QUANTILES]) 

preds.loc[preds.method == 'Model_3', 'pred'] = np.concatenate(
    [quantreg_3.fit(q=q).predict(X_3) for q in QUANTILES])

preds['quantile_loss'] = quantile_loss(preds.q, preds.label, preds.pred)

def plot_loss_comparison(preds):
    overall_loss_comparison = preds[~preds.quantile_loss.isnull()].\
      pivot_table(index='method', values='quantile_loss').\
      sort_values('quantile_loss')
    # Show overall table.
    print(overall_loss_comparison)
  
    # Plot overall.
    with sns.color_palette('Blues', 1):
        ax = overall_loss_comparison.plot.barh()
        plt.title('Total Quantile Loss', loc='left')
        sns.despine(left=True, bottom=True)
        plt.xlabel('Quantile loss')
        plt.ylabel('')
        ax.legend_.remove()
  
    # Per quantile.
    per_quantile_loss_comparison = preds[~preds.quantile_loss.isnull()].\
        pivot_table(index='q', columns='method', values='quantile_loss')
    # Sort by overall quantile loss.
    per_quantile_loss_comparison = \
        per_quantile_loss_comparison[overall_loss_comparison.index]
    print(per_quantile_loss_comparison)
  
    # Plot per quantile.
    with sns.color_palette('Blues'):
        ax = per_quantile_loss_comparison.plot.barh()
        plt.title('Quantile loss per quantile', loc='left')
        sns.despine(left=True, bottom=True)
        handles, labels = ax.get_legend_handles_labels()
        plt.xlabel('Quantile loss')
        plt.ylabel('Quantile')
        # Reverse legend.
        ax.legend(reversed(handles), reversed(labels))


plot_loss_comparison(preds)

METHODS = ['Model_4', 'Model_5', 'Model_6']


preds_diff = np.array([(method, q, x) 
                  for method in METHODS 
                  for q in QUANTILES
                  for x in X['const']])

preds_diff = pd.DataFrame(preds_diff)
preds_diff.columns = ['method', 'q', 'x']
preds_diff = preds_diff.apply(lambda x: pd.to_numeric(x, errors='ignore'))

preds_diff['label'] = np.resize(y_diff, preds_diff.shape[0])

preds_diff.loc[preds_diff.method == 'Model_4', 'pred'] = np.concatenate(
    [quantreg_4.fit(q=q).predict(X_1) for q in QUANTILES]) 

preds_diff.loc[preds_diff.method == 'Model_5', 'pred'] = np.concatenate(
    [quantreg_5.fit(q=q).predict(X_5) for q in QUANTILES]) 

preds_diff.loc[preds_diff.method == 'Model_6', 'pred'] = np.concatenate(
    [quantreg_6.fit(q=q).predict(X_6) for q in QUANTILES])

preds_diff['quantile_loss'] = quantile_loss(preds_diff.q, preds_diff.label, preds_diff.pred)

plot_loss_comparison(preds_diff)
