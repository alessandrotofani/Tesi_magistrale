import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#####################################################################################################
## Functions for mice ##
#####################################################################################################

def split_by_day(dataset, days): 
    
    day = 86400 # secondi in un giorno

    # indici per il loop
    start = day
    end = day * 2 

    data_splitted = {} # dizionario che contiene i vari set splittati per giorno
    # loop per riempire il dizionario
    for i in range(days):
        data_splitted[i] = dataset[(dataset['TransactionDT'] >= (start)) & (dataset['TransactionDT'] < (end - 1))]
        start += day
        end += day
        
    return data_splitted

def select_col_by_nan(dataset, tresh): # funzione per selezionare le colonne di un dataset in base al numero di NaN
    cols = [] # lista che contiene le colonne con numero di NaN inferiore alla soglia 
    for col in dataset.columns:
        if dataset[col].isna().sum() < tresh:
            cols.append(col)
    return cols

def mice(dataset, days, tresh = 100000): # funzione che performa il MICE sul dataset selezionato
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    cols = select_col_by_nan(dataset, tresh) # lista che contiene le colonne con meno nan della soglia
    print(len(cols))
    data = dataset[dataset.columns.intersection(cols)]
    data_splitted = split_by_day(data, days) # dizionario con dataset splittato per giorno

    fitted = {} # dizionario che contiene i dataset splittati con gli imputed values
    for day in range(days): # faccio l'imputation su ogni dataset riguardante le transazioni giornaliere 
        # subset = data_splitted[day][[col for col in dataset.columns if col in cols]]
        subset = data_splitted[day]
        imp = IterativeImputer(missing_values=np.nan, random_state=0, n_nearest_features=5, max_iter = 40)                          
        imp.fit(subset)
        subset = imp.transform(subset)
        fitted_set = subset
        fitted[day] = pd.DataFrame(fitted_set, columns = cols).round(2) # trasformo la matrice ottenuta in un dataframe 

    return fitted # ritorno il dizionario i cui elementi sono i dataset giornalieri con i valori imputed

#####################################################################################################
## Functions for eda ##
#####################################################################################################

def get_stat(dataset, mean = False, std = False):
    col_to_drop = ['TransactionAmt','TransactionID','TransactionDT']
    if mean:
        means = dataset.groupby(['isFraud']).mean()
        means = means.drop(col_to_drop, axis = 1)
        return means
    if std:
        stds = dataset.groupby(['isFraud']).std()
        stds = stds.drop(col_to_drop, axis = 1)
        return stds
    else:
        print('Specificy if you want the mean or std')
    
def get_subFrame(dataset, safe = False, fraud = False):
    col_to_drop = ['TransactionAmt','TransactionID','TransactionDT','isFraud']
    if safe:
        safe_dataset = dataset[dataset['isFraud']==0].drop(col_to_drop, axis = 1)
        return safe_dataset
    if fraud:
        fraud_dataset = dataset[dataset['isFraud']==1].drop(col_to_drop, axis = 1)
        return fraud_dataset
    else:
        print('Specificy if you want the safe or fraud rows')

from scipy import stats

def diff(df):
    res = {}
    for col in df.columns:
        res[col] = df[col][0] - df[col][1]
    return res

def s(df_safe, df_fraud, stds):
    res = {}
    for col in df_safe.columns:
        s0_2 = stds[col][0]**2
        s1_2 = stds[col][1]**2
        n0 = len(df_safe)
        n1 = len(df_fraud)
        res[col] = np.sqrt(s0_2 /n0 + s1_2 /n1)
    return res

def t(mean, std, df_safe, df_fraud):
    res = {}
    s_ = s(df_safe, df_fraud, std)
    for col in mean.columns:
        res[col] = diff(mean)[col] / s_[col]
    return res

def v(df_safe, df_fraud, stds):
    res = {}
    for col in df_safe.columns:
        s0_2 = stds[col][0]**2
        s1_2 = stds[col][1]**2
        n0 = len(df_safe)
        n1 = len(df_fraud)
        v0 = n0 - 1
        v1 = n1 - 1
        v = ((s0_2 / n0 + s1_2 / n1)**2)/(s0_2**2/(n0**2 * v0) + s1_2**2/(n1**2 * v1))
        res[col] = np.ceil(v)
    return res

def sig_cols(t_variable, dataset, liv_sign = 0.95):
    p_value = {}
    sig_cols = 0
    num_sign_col = []
#     num_col_not_sign = []

    for col in dataset.columns:    
        p_value[col] = 1 - stats.t.cdf(t_variable[col], df = dof[col])
        if p_value[col] > liv_sign:
            num_sign_col.append(col)
            print('Feature ', col, 'has a pvalue of: ', p_value[col])
            sig_cols += 1
#         else:
#             num_col_not_sign.append(col)
#             print('Feature ', col, 'is below significancy level')
    print(len(num_sign_col), ' significative columns on ', len(dataset.columns), 'total columns')
    return num_sign_col

def get_sign_cols(count, liv_sign = 0.95):
    from scipy.stats import chi2_contingency 

    stat = {}
    p = {}
    dof = {}
    expected = {}

    # dizionari che contengono i nomi delle feature significative e non 
    cat_col_sign = []
#     cat_col_not_sign = []

    for col in count:
        stat[col], p[col], dof[col], expected[col] = chi2_contingency(count[col]) 
        if p[col] < liv_sign:
            cat_col_sign.append(col)
            print('Feature ', col,' is significant \t Chi square: ', stat[col], '\t dof: ', dof[col], '\n')
#         else:
#             cat_col_not_sign.append(col)
#             print('Feature ', col,' is NOT significant \t Chi square: ', stat[col], '\t dof: ', dof[col], '\n')

    print('Number of significative features: ', len(cat_col_sign))
    return cat_col_sign

#####################################################################################################
## Analisi correlazioni ##
#####################################################################################################

def dropColNotSign(dataset, col_sign, not_ignore = None):
  col_not_sign = []
  for col in dataset.columns:
    if col not in col_sign and col != not_ignore:
        col_not_sign.append(col)
  dataset = dataset.drop(col_not_sign, axis=1)            
  return dataset

def corr_matrix_plot(dataset, corr_matrix):
    f = plt.figure(figsize=(20, 20))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(range(dataset.shape[1]), dataset.columns, fontsize=10, rotation=90)
    plt.yticks(range(dataset.shape[1]), dataset.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    
def highest_correlations(corr_matrix, tresh = 0.8):
    corr_matrix_abs = corr_matrix.abs()
    corr_matrix_abs = corr_matrix_abs.unstack()
    corr_sorted = corr_matrix_abs.sort_values(kind="quicksort")
    corr_sorted = corr_sorted[:-len(num_sign_col)] # levo i termini sulla diagonale
    corr_sorted = corr_sorted.drop_duplicates() # levo i termini doppi
    corr_sorted = corr_sorted[corr_sorted>tresh]
    return corr_sorted

def corr_dict(corr):
    corr_dict = {}
    corr_list = []
    for col1, df in corr.groupby(level=0):
        for col2 in df:
           if not np.isnan(corr.loc[col1,col2]):
                corr_dict[col1,col2] = corr.loc[col1,col2]
                corr_list.append([col1,col2])
                corr_list.append([col2,col1])
                
    return corr_dict, corr_list

#####################################################################################################
## Preprocessing ##
#####################################################################################################

def min_max_scaling(data):
  col_not_to_scale = ['isFraud','TransactionID', 'TransactionDT']
  for col in data.columns:
    if col not in col_not_to_scale:
      data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
       
  return data

def split(dataset, test_size):  
  from sklearn.model_selection import train_test_split
  y = dataset['isFraud']
  X = dataset.drop(['isFraud'], axis = 1)

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
  return X_train, X_val, y_train, y_val

def ap_metric(clf, X_val, y_val, proba):
  from sklearn.metrics import precision_recall_curve
  from sklearn.metrics import plot_precision_recall_curve
  from sklearn.metrics import average_precision_score

  if proba:
    y_score = clf.predict_proba(X_val)
    average_precision = average_precision_score(y_val, y_score[:,1])

  else:
    y_score = clf.predict(X_val)
    average_precision = average_precision_score(y_val, y_score)
   
  disp = plot_precision_recall_curve(clf, X_val, y_val)
  disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
  return

#####################################################################################################
## Performance measure ##
#####################################################################################################

def performance(clf, X_val, y_val, proba = True):
  from sklearn.metrics import plot_roc_curve

  plot_roc_curve(clf, X_val, y_val)
  plt.show()

  ap_metric(clf, X_val, y_val, proba)
  return

def conf_matrix(clf, X_val, y_val):
  from sklearn.metrics import plot_confusion_matrix

  # disp = plot_confusion_matrix(classifier, X_val, y_val, display_labels=class_names, cmap=plt.cm.Blues, normalize=False)
  disp = plot_confusion_matrix(clf, X_val, y_val, cmap=plt.cm.Blues, normalize=None)
  disp.ax_.set_title('Confusion matrix')
  # print(title)
  print(disp.confusion_matrix)
  return

# def save_model(clf):
#   import pickle
#   with open('model.pkl','wb') as f:
#     pickle.dump(clf,f)
#   return

# def load_model():
#   import pickle
#   with open('model.pkl', 'rb') as f:
#     clf = pickle.load(f)
#   return clf

def col_not_sign(dataset, sign_cols):
  col_not_sign = []
  for col in dataset.columns:
    if col not in sign_cols and col != 'TransactionID':
      col_not_sign.append(col)
  col_not_sign.append('isFraud')
  return col_not_sign

def get_col(data):
  cols = []
  for col in data.columns:
      if col != 'isFraud':
        cols.append(col)
  return cols

def select_days(dataset, days):
  sec = 86400
  tot = days * sec
  dataset = dataset[dataset['TransactionDT'] < tot]
  return dataset

def easy_ensemble(n_subsets, X_train, y_train):
  from imblearn.ensemble import EasyEnsemble 
  ee = EasyEnsemble(random_state=42, n_subsets=n_subsets)
  X_trainres, y_trainres = ee.fit_sample(X_train, y_train)
  return X_trainres, y_trainres

def roc_auc_subset(clf, X_train, y_train, X_val, y_val, n_subsets = 5):
  from sklearn.metrics import auc
  from sklearn.metrics import plot_roc_curve
  from xgboost import XGBClassifier

  # classifier = RandomForestClassifier(max_depth=2, random_state=0)
  classifier = clf
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)

  fig, ax = plt.subplots()
  for i in range(n_subsets):
      classifier.fit(X_train[i], y_train[i])
      viz = plot_roc_curve(classifier, X_val, y_val,
                          name='ROC fold {}'.format(i),
                          alpha=0.3, lw=1, ax=ax)
      interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
      interp_tpr[0] = 0.0
      tprs.append(interp_tpr)
      aucs.append(viz.roc_auc)

  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Chance', alpha=.8)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(mean_fpr, mean_tpr, color='b',
          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
          lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                  label=r'$\pm$ 1 std. dev.')

  ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example")
  ax.legend(loc="lower right")
  plt.show()
  return

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
def f1(model, X_val, y_val):
  from sklearn.metrics import f1_score
  y_pred = model.predict(X_val) 
  return f1_score(y_val, y_pred, average='macro')

#####################################################################################################
## Autoencoder  ##
#####################################################################################################

def mse_calc(X, X_pred):
  from sklearn.metrics import mean_squared_error
  err = []

  for i in range(X.shape[0]):
    err.append(mean_squared_error(X.iloc[i], X_pred[i]))

  return err

def mse(safe, fraud, autoencoder):

  safe_predicted = autoencoder.predict(safe)
  fraud_predicted = autoencoder.predict(fraud)

  safe_errors = mse_calc(safe, safe_predicted)
  fraud_errors = mse_calc(fraud, fraud_predicted)

  return safe_errors, fraud_errors

def make_df(X_val, fraud_train, autoencoder):
  safe_errors, fraud_errors = mse(X_val, fraud_train, autoencoder) 

  safe_df = pd.DataFrame({'mse': safe_errors, 'anomaly': np.zeros(len(safe_errors))})
  fraud_df = pd.DataFrame({'mse': fraud_errors,	'anomaly': np.ones(len(fraud_errors))})

  mse_df = pd.concat([safe_df, fraud_df])

  return mse_df

# confusion matrix
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def performance_autoencoder(X_test, fraud_test, autoencoder, soglia):
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import recall_score
  from sklearn.metrics import average_precision_score
  from sklearn.metrics import f1_score

  mse_df = make_df(X_test, fraud_test, autoencoder)
  y_pred = []
  for mse in mse_df['mse']:
    if mse > soglia:
      y_pred.append(0)
    if mse < soglia:
      y_pred.append(1)

  y_safe = np.zeros(X_test.shape[0])
  y_fraud = np.ones(fraud_test.shape[0])
  y_true = np.concatenate((y_safe, y_fraud))

  recall =  recall_score(y_true, y_pred, average='macro')
  print('Recall: ', recall)
  average_precision = average_precision_score(y_true, y_pred)
  print('Average Precision: ', average_precision)
  f1 = f1_score(y_true, y_pred, average='macro')
  print('F1 score: ', f1)
  cm = confusion_matrix(y_true, y_pred)
  print(cm)
  
  return cm, recall, average_precision, f1







