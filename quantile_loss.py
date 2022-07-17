#In the regression loss equation above, as q has a value between 0 and 1, the first term will be positive and dominate when under-predicting, yi > yip, and the second term will dominate when over-predicting, yi < yip. For q equal to 0.5, under-prediction and over-prediction will be penalized by the same factor, and the median is obtained. The larger the value of q, the more under-predictions are penalized compared to over-predictions. For q equal to 0.75, under-predictions will be penalized by a factor of 0.75, and over-predictions by a factor of 0.25. The model will then try to avoid under-predictions approximately three times as hard as over-predictions, and the 0.75 quantile will be obtained.

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

# PLOT

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
