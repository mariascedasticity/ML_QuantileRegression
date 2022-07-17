def loss(q, y, f):
    # q: Quantile to be evaluated, e.g., 0.5 for median.
    # y: True value.
    # f: Fitted or predicted value.
    e = y - f
    return e
  
preds_3['residuals'] = loss(preds_3.q, preds_3.label, preds_3.pred)
preds_3

g = sns.displot(preds_5, x="residuals", hue=None, col="q", kind='kde', fill=True, col_wrap=3, height=3)
plt.show()

#from wide to long for visualisations in seabborn
melted_5 = pd.melt(preds_5, id_vars=['method', 'q','index'], value_vars=['label_diff', 'pred'])
sns.displot(data=melted_5, x="value", hue="variable", col="q", kind='kde', col_wrap=3, height=3, fill = True)
