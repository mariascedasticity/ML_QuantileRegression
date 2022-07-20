df_abs = pd.read_csv('/content/gdrive/MyDrive/datasets/df_RL.csv')
pd.set_option('display.max_columns', 50)
df_abs=df_abs.sort_values(by="Date").reset_index(drop=True)

y_diff = df_abs[['Price_dev']]

# base: 0 and spring
pred = df_abs[['RL_low', 'RL_high', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
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
	params_5.append([qt, res.params['const'], res.params['RL_low'], res.pvalues['RL_low'], res.params['RL_high'], res.pvalues['RL_high'],
                res.params['1'], res.pvalues['1'], res.params['2'], res.pvalues['2'],
                res.params['3'], res.pvalues['3'], res.params['4'], res.pvalues['4'], res.params['5'], res.pvalues['5'],
                res.params['6'], res.pvalues['6'], res.params['7'], res.pvalues['7'], res.params['8'], res.pvalues['8'],
                res.params['9'], res.pvalues['9'], res.params['10'], res.pvalues['10'], res.params['11'], res.pvalues['11'],
                res.params['12'], res.pvalues['12'], res.params['13'], res.pvalues['13'], res.params['14'], res.pvalues['14'],
                res.params['15'], res.pvalues['15'], res.params['16'], res.pvalues['16'], res.params['17'], res.pvalues['17'],
                res.params['18'], res.pvalues['18'], res.params['19'], res.pvalues['19'], res.params['20'], res.pvalues['20'],
                res.params['21'], res.pvalues['21'], res.params['22'], res.pvalues['22'], res.params['23'], res.pvalues['23'],
                res.params['autumn'], res.pvalues['autumn'], res.params['summer'], res.pvalues['summer'], res.params['winter'], res.pvalues['winter']])

params_5 = pd.DataFrame(data = params_5, columns = ['qt','intercept', 'RL_low', 'RL_low_pvalue', 'RL_high', 'RL_high_pvalue',
                                                '1', '1_pvalue', '2', '2_pvalue', '3', '3_pvalue',
                                                '4', '4_pvalue', '5', '5_pvalue', '6', '6_pvalue', '7', '7_pvalue', '8', '8_pvalue',
                                                '9', '9_pvalue', '10', '10_pvalue', '11', '11_pvalue', '12', '12_pvalue', '13', '13_pvalue',
                                                '14', '14_pvalue', '15', '15_pvalue', '16', '16_pvalue', '17', '17_pvalue', '18', '18_pvalue',
                                                '19', '19_pvalue', '20', '20_pvalue', '21', '21_pvalue', '22', '22_pvalue', '23', '23_pvalue',
                                                'autumn', 'autumn_pvalue', 'summer', 'summer_pvalue', 'winter', 'winter_pvalue'])
#print(params)

preds_5['quantile loss'] = quantile_loss(preds_5.q, preds_5.label_diff, preds_5.pred)
preds_5

plt.figure(8)
params_5.plot(x = 'qt', y = ["RL_low", 'RL_high'], 
	title = 'Coefficients based on quantiles', kind ='line', xlabel = 'Qunatiles', ylabel = 'Coefficients')
