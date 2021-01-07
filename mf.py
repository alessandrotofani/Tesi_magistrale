import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#####################################################################################################
## MICE ##
#####################################################################################################

def split_by_day(dataset, days): # splitta il dataset in più giorni
    '''
    in
    dataset: dataset da splittare
    days: numero di giorni totali che si vogliono selezionare
    out
    data_splitted: dizionario i cui elementi sono i dataset relativi ad un determinato giorno (giorno: dataset_giornaliero)
    '''  
    day = 86400 # secondi in un giorno
    # indici per il loop
    start = day
    end = day * 2 

    data_splitted = {} # inizializzo il dizionario
    for i in range(days): # loop per riempire il dizionario
        data_splitted[i] = dataset[(dataset['TransactionDT'] >= (start)) & (dataset['TransactionDT'] < (end - 1))]
        start += day
        end += day
    return data_splitted

def select_col_by_nan(dataset, tresh): # funzione per selezionare le colonne di un dataset in base al numero di NaN
    '''
    in
    dataset: dataset con i nan
    tresh: soglia sotto la quale le colonne vengono ignorate
    out
    cols: lista con le colonne con numero di nan inferiore alla soglia
    '''
    cols = [] # inizializzo la lista 
    for col in dataset.columns:
        if dataset[col].isna().sum() < tresh:
            cols.append(col)
    return cols

def mice(dataset, days, tresh = 100000): # funzione che performa il MICE sul dataset selezionato
    '''
    in
    dataset: dataset sul quale si vuole performare il mice
    days: giorni totali sui quali si vuole fare la imputation
    tresh: soglia sul numero di nan. Le colonne che hanno un numero di nan superiore alla soglia verranno eliminate
    out
    fitted: dizionario i cui elementi sono i dataset giornalieri su cui è stata fatta l'imputation (day: imputed_dataset)
    '''
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

def get_stat(dataset, mean = False, std = False): # calcola la media o la deviazione standard delle feature 
    '''
    in
    dataset: dataset 
    mean: se True, calcola le medie
    std: se True, calcola le deviazioni standard
    out
    means: dataframe con le medie calcolate per ogni feature
    stds: dataframe con le deviazioni standard calcolate per ogni feature
    '''
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
    
def get_subFrame(dataset, safe = False, fraud = False): # serve per eliminare delle colonne dal dataframe e selezionare un sottoinsieme a seconda del tipo di transazione
    '''
    in
    dataset: dataframe
    safe: se True, seleziona le transazioni safe
    fraud: se True, seleziona le transazioni fraudolente
    out
    safe_dataset: dataframe con le transazioni safe senza le colonne col_to_drop
    fraud_dataset: dataframe con le transazioni fraudolente senza le colonne col_to_drop
    '''
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

def diff(df): # calcola la differenza tra elementi di un dataframe
    '''
    in 
    df: dataframe
    out
    res: differenza
    '''
    res = {}
    for col in df.columns:
        res[col] = df[col][0] - df[col][1]
    return res

def s(df_safe, df_fraud, stds): # calcola la deviazione standard secondo la statistica della variabile t 
    '''
    in
    df_safe: dataset con le transazioni safe
    df_fraud: dataset con le transazioni fraudolente
    stds: dizionario con le deviazioni standard    
    out
    res: dizionario con la deviazione std secondo la statistica di t per feature (feature: s)
    '''
    res = {}
    for col in df_safe.columns:
        s0_2 = stds[col][0]**2
        s1_2 = stds[col][1]**2
        n0 = len(df_safe)
        n1 = len(df_fraud)
        res[col] = np.sqrt(s0_2 /n0 + s1_2 /n1)
    return res

def t(mean, std, df_safe, df_fraud): # calcola la variabile t di Student
    ''' 
    in
    mean: dataframe con le medie per ogni feature
    std: dizionario con le deviazioni standard
    df_safe: dataset con le transazioni safe
    df_fraud: dataset con le transazioni fraudolente
    out
    res: dizionario con la variabile t per ogni feature (feature: t)
    '''
    res = {}
    s_ = s(df_safe, df_fraud, std) # calcolo delle deviazioni standard
    for col in mean.columns:
        res[col] = diff(mean)[col] / s_[col]
    return res

def v(df_safe, df_fraud, stds): # calcola il parametro v, necessario per il test statistico
    '''
    in
    df_safe: dataset con le transazioni safe
    df_fraud: dataset con le transazioni fraudolente
    stds: dizionario che contiene le deviazioni standard per feature
    out
    res: dizionario con parametro v calcolato per ogni feature (feature: v)
    '''
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

def sig_cols(t_variable, dataset, dof, liv_sign = 0.95): # restituisce la lista con le feature numeriche significative
    '''
    in
    t_variable: dizionario contenete i valori della variabile t per feature (feature: t)
    dataset: dataset contente le feature numeriche
    dof: dizionario contenente i gradi di libertà per feature (feature: dof)
    liv_sign: livello di significatività del test statistico
    out
    num_sign_col: lista con le feature numeriche significative
    '''
    p_value = {}
    sig_cols = 0
    num_sign_col = []
#     num_col_not_sign = []

    for col in dataset.columns:    
        p_value[col] = 1 - stats.t.cdf(t_variable[col], df = dof[col]) # questa funzione permette di ricavare il valore della cumulata dat t e i dof 
        if p_value[col] > liv_sign:
            num_sign_col.append(col)
            print('Feature ', col, 'has a pvalue of: ', p_value[col])
            sig_cols += 1
#         else:
#             num_col_not_sign.append(col)
#             print('Feature ', col, 'is below significancy level')
    print(len(num_sign_col), ' significative columns on ', len(dataset.columns), 'total columns')
    return num_sign_col

def get_sign_cols(count, liv_sign = 0.95): # restituisce la lista con le feature categoriche significative
    '''
    in 
    count: dizionario con i conteggio per ogni feature
    liv_sign: livello di significatività del test statistico
    out
    cat_col_sign: lista con i nomi delle feature significative
    '''
    from scipy.stats import chi2_contingency 

    stat = {}
    p = {}
    dof = {}
    expected = {}
    cat_col_sign = []

    for col in count: # tramite la funzione di scikit calcola il valore del chi-quadro, il p-value, i gradi di libertà e il valore atteso del chi-quadro. 
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

def dropColNotSign(dataset, col_sign, not_ignore = None): # rimuove le feature non significative
  '''
  in
  dataset: dataset
  cols_sign: lista con le efature significative
  not_ignore: feature da non rimuovere
  out
  dataset: dataset con solo le feature significative
  '''
  col_not_sign = []
  for col in dataset.columns:
    if col not in col_sign and col != not_ignore:
        col_not_sign.append(col)
  dataset = dataset.drop(col_not_sign, axis=1)            
  return dataset

def corr_matrix_plot(dataset, corr_matrix): # plotta la matrice di correlazione
    '''
    in 
    dataset: dataset
    corr_matrix: matrice di correlazione
    '''
    f = plt.figure(figsize=(20, 20))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(range(dataset.shape[1]), dataset.columns, fontsize=10, rotation=90)
    plt.yticks(range(dataset.shape[1]), dataset.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    return 
    
def highest_correlations(corr_matrix, tresh = 0.8): # restituisce le feature con correlazione sopra soglia
    '''
    in
    corr_matrix: matrice di correlazione
    tresh: soglia sulla correlazione
    out
    corr_sorted: dizionario con le correlazioni con valore sopra soglia
    '''
    corr_matrix_abs = corr_matrix.abs()
    corr_matrix_abs = corr_matrix_abs.unstack()
    corr_sorted = corr_matrix_abs.sort_values(kind="quicksort")
    corr_sorted = corr_sorted[:-len(num_sign_col)] # levo i termini sulla diagonale
    corr_sorted = corr_sorted.drop_duplicates() # levo i termini doppi
    corr_sorted = corr_sorted[corr_sorted>tresh]
    return corr_sorted

def corr_dict(corr): # restituisce il dizionario con le correlazioni
    '''
    corr: matrice con le correlaioni
    out
    corr_dict: dizionario con le correlazioni
    corr_list: lista con le features correlate
    '''
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

def min_max_scaling(data): # fa il min max scaling del dataset
  '''
  in
  data: dataset
  out
  data: dataset con i feature values scalati 
  '''
  col_not_to_scale = ['isFraud','TransactionID', 'TransactionDT']
  for col in data.columns:
    if col not in col_not_to_scale:
      data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
       
  return data

def split(dataset, test_size): # splitta il dataset in train set e labels
  '''
  in
  dataset: dataset
  test_size: size del test set
  out
  X_train: train set
  X_val: validation set
  y_train, y_val: labels
  '''
  from sklearn.model_selection import train_test_split
  y = dataset['isFraud']
  X = dataset.drop(['isFraud'], axis = 1)

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
  return X_train, X_val, y_train, y_val

def ap_metric(clf, X_val, y_val, proba): # plotta la mean average precision
  '''
  in
  clf: modello trainato
  X_val: validation set
  y_val: labels
  proba: True per i modelli che supportano predict_proba
  '''
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

def performance(clf, X_val, y_val, proba = True): # plotta la curva roc e la mean average precision
  '''
  in
  clf: modello trainato
  X_val: validation set
  y_val: labels
  proba: True per i modelli che supportano predict_proba
  '''
  from sklearn.metrics import plot_roc_curve

  plot_roc_curve(clf, X_val, y_val)
  plt.show()

  ap_metric(clf, X_val, y_val, proba)
  return

def conf_matrix(clf, X_val, y_val): # plotta la confusion matrix
  '''
  in
  clf: modello trainato
  X_val: validation set
  y_val: labels 
  '''
  from sklearn.metrics import plot_confusion_matrix

  # disp = plot_confusion_matrix(classifier, X_val, y_val, display_labels=class_names, cmap=plt.cm.Blues, normalize=False)
  disp = plot_confusion_matrix(clf, X_val, y_val, cmap=plt.cm.Blues, normalize=None)
  disp.ax_.set_title('Confusion matrix')
  # print(title)
  print(disp.confusion_matrix)
  return

def col_not_sign(dataset, sign_cols): # restituisce le colonne non significative
  '''
  in
  dataset: dataset
  sign_cols: lista col nome delle feature significative
  out
  col_not_sign: lista col nome delle feature non significative
  '''
  col_not_sign = []
  for col in dataset.columns:
    if col not in sign_cols and col != 'TransactionID':
      col_not_sign.append(col)
  col_not_sign.append('isFraud')
  return col_not_sign

def get_col(data): # restituisce le colonne dal dataframe
  '''
  in
  data: dataset
  out
  cols: lista con le colonne del dataset 
  '''
  cols = []
  for col in data.columns:
      if col != 'isFraud':
        cols.append(col)
  return cols

def select_days(dataset, days): # seleziona i dati fino al giorno indicato
  '''
  in 
  dataset: dataset
  days: numero di giorni che si vogliono selezionare
  out
  dataset: dataset con il numero di giorni selezionato
  '''
  sec = 86400
  tot = days * sec
  dataset = dataset[dataset['TransactionDT'] < tot]
  return dataset

def easy_ensemble(n_subsets, X_train, y_train): # applica l'easy ensemble al set specificato
  '''
  in
  n_subsets: numero di subset che si vogliono formare
  X_train: train set
  y_train: labels
  out
  X_trainres: dizionario con train set resampled
  y_trainres: dizionario con le labels 
  '''
  from imblearn.ensemble import EasyEnsemble 
  ee = EasyEnsemble(random_state=42, n_subsets=n_subsets)
  X_trainres, y_trainres = ee.fit_sample(X_train, y_train)
  return X_trainres, y_trainres

def roc_auc_subset(clf, X_train, y_train, X_val, y_val, n_subsets = 5): # plotta la curva roc auc sul numero di subset specificato
  '''
  in
  clf: modello trainato
  X_train: train set
  y_train: labels
  X_val: validation set
  y_val: labels
  n_subsets: numero di subsets di cui plottare la roc
  '''
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
def f1(model, X_val, y_val): # calcola lo score f1 
  '''
  in
  model: modello trainato
  X_val: validation set
  y_val: labels
  out
  f1_score: f1 score
  '''
  from sklearn.metrics import f1_score
  y_pred = model.predict(X_val) 
  return f1_score(y_val, y_pred, average='macro')

def loss_by_day(X_test, y_test, clf): # calcola la lossa e la cm per giorno
  '''
  in
  X_test: test set 
  y_test: labels
  clf: modello trainato
  out
  loss: dizionario con le loss calcolate per giorno
  cm: dizionario con le confusion matrix calcolate per giorno
  '''
  from sklearn.metrics import log_loss, confusion_matrix
  y_pred = clf.predict_proba(X_test) # labels predette (probabilità)
  pred_df = pd.DataFrame(data=y_pred, index = X_test.index) # dataframe che conterrà labels, labels predette, day
  pred_df['day'] = X_test['day']
  pred_df['label'] = y_test
  pred_df['pred_label'] = clf.predict(X_test)

  splitted = {} # dizionario contenete il dataset splittato per giorni 
  for i in range(7): # loop per splittare il dataset 
    splitted[i] = pred_df[pred_df['day'] == i]
  
  loss = {}
  cm = {}
  for i in range(7):
    labels = splitted[i]['label'] # seleziono la label
    pred_vector =  splitted[i][[0,1]] # seleziono il vettore con le probabilità
    pred_label = splitted[i]['pred_label'] # seleziono la label predetta
    loss[i] = log_loss(labels, pred_vector).round(4) # calcolo la loss
    cm[i] = confusion_matrix(labels, pred_label, normalize = 'true') # calcolo la consufion matrix
  
  return loss, cm

def fraud_ratio_per_feature(data, feature, verbose = False): # calcola i rate di transazioni fraudolente 
  '''
  in 
  data: dataset 
  feature: feature (categorica) di cui si vuole calcolare il rate 
  verbose: se True, restituisce più valori
  out
  ratio: rate di transazioni fraudolente sul totale
  fraud_count: numero di transazioni fraudolente
  tot: numero di transazioni totali in cui la feature ha valore pari a 1 
  '''
  subset = data[data[feature] == 1]
  tot = subset.shape[0]
  fraud_count = subset[subset['isFraud'] == 1].sum()[0]
  ratio = fraud_count/tot
  if not verbose:
    return ratio
  else:
    return ratio, fraud_count, tot

def ratio_dictionary(clf, data, n_features): # calcola i rate e restituisce il dizionario 
  '''
  in
  clf: modello trainato
  data: dataset
  n_features: numero massimo di features da considerare
  out
  ratio_dict: dizionario con il rate di transazioni fraudolente per feature ordinate per score 
  '''
  feature_score = clf.get_booster().get_score(importance_type='gain') # calcolo lo score 
  important_features = sorted(feature_score, key=feature_score.get, reverse=True)[:n_features] #ordino le features 
  ratio_dict = {} # inizializzo il dizionario 
  for feature in important_features: # loop calcolare il rate 
    ratio_dict[feature] = list(fraud_ratio_per_feature(data, feature, verbose = True))
    if ratio_dict[feature][0] == 0:
      del ratio_dict[feature]
  return ratio_dict

#####################################################################################################
## Autoencoder  ##
#####################################################################################################

def mse_calc(X, X_pred): # calcola gli mse 
  '''
  in
  X: labels
  X_pred: label predette
  out
  err: mse 
  '''
  from sklearn.metrics import mean_squared_error
  err = []

  for i in range(X.shape[0]):
    err.append(mean_squared_error(X.iloc[i], X_pred[i]))

  return err

def mse(safe, fraud, autoencoder): # calcola gli mse 
  '''
  in
  safe: dataset con transazioni safe
  fraud: dataset con transazioni fraudolente
  autoencoder: AE trainato
  out
  safe_errors: mse per le transazioni safe
  fraud_errors: mse per le transazioni fraud 
  '''

  safe_predicted = autoencoder.predict(safe) # label predette 
  fraud_predicted = autoencoder.predict(fraud)

  safe_errors = mse_calc(safe, safe_predicted) # calcola gli mse 
  fraud_errors = mse_calc(fraud, fraud_predicted)

  return safe_errors, fraud_errors

def make_df(X_val, fraud_train, autoencoder): # crea un dataframe con i mse
  '''
  in 
  X_val: validation set solo con le transazioni safe
  fraud_train: train set di transazioni fraudolente
  autoencoder: AE trainato
  out
  mse_df: dataframe contenete gli mse
  '''
  safe_errors, fraud_errors = mse(X_val, fraud_train, autoencoder) 

  safe_df = pd.DataFrame({'mse': safe_errors, 'anomaly': np.zeros(len(safe_errors))})
  fraud_df = pd.DataFrame({'mse': fraud_errors,	'anomaly': np.ones(len(fraud_errors))})

  mse_df = pd.concat([safe_df, fraud_df])

  return mse_df

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def performance_autoencoder(X_test, fraud_test, autoencoder, soglia): # permette di valutare le performance dell'autoencoder
  '''
  in
  X_test: test set solo con transazioni safe
  fraud_test: test set solo con transazioni fraudolente
  autoencoder: AE trainato
  soglia: soglia usata per discriminare le transazioni a seconda del mse 
  out
  cm: confusion matrix
  recall: recall del modello
  average_precision: mean average precision del modello
  f1: f1 score del modello
  '''
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import recall_score
  from sklearn.metrics import average_precision_score
  from sklearn.metrics import f1_score

  mse_df = make_df(X_test, fraud_test, autoencoder) # dataframe contenente i mse 
  y_pred = [] # labels predette 
  for mse in mse_df['mse']: # loop che confronta gli mse con la soglia per calcolare le labels
    if mse > soglia:
      y_pred.append(0)
    if mse < soglia:
      y_pred.append(1)

  y_safe = np.zeros(X_test.shape[0]) 
  y_fraud = np.ones(fraud_test.shape[0])
  y_true = np.concatenate((y_safe, y_fraud)) # array con le label 

  recall =  recall_score(y_true, y_pred, average='macro')
  print('Recall: ', recall)
  average_precision = average_precision_score(y_true, y_pred)
  print('Average Precision: ', average_precision)
  f1 = f1_score(y_true, y_pred, average='macro')
  print('F1 score: ', f1)
  cm = confusion_matrix(y_true, y_pred)
  print(cm)
  
  return cm, recall, average_precision, f1

#####################################################################################################
## Rete neurale  ##
#####################################################################################################

# Riferimento cm: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def plot_cm(labels, predictions, p=0.5): # plotta la confusion matrix
  '''
  in
  labels: labels
  predictions: labels predette dal modello
  p: treshold 
  '''
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  return

# Riferimento auc: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
def plot_roc(name, labels, predictions, **kwargs): # plotta la roc auc curve 
  '''
  in
  name: label della funzione plottata
  labels: labels
  predictions: labels predette dal modello
  '''
  from sklearn.metrics import roc_curve, auc

  fp, tp, _ = roc_curve(labels, predictions)
  auc_nn = auc(fp, tp)
  print('AUC: ', auc_nn)
  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.legend(loc='lower right')
  plt.grid(True)
  ax = plt.gca()
  return

def plot_ap(name, y_test, y_pred, **kwargs): # plotta la mean average precision
  '''
  in 
  name: label della funzione plottata
  y_test: labels
  y_pred: labels predette
  '''
  from sklearn.metrics import precision_recall_curve, average_precision_score

  precision, recall, _ = precision_recall_curve(y_test, y_pred)
  ap = average_precision_score(y_test, y_pred)
  print('Average precision: ', ap)
  plt.plot(100*recall, 100*precision, label=name, linewidth=2, **kwargs)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='lower right')
  plt.grid(True)
  ax = plt.gca()
  return


#####################################################################################################
## Catboost ##
#####################################################################################################

# viene fatto in un secondo file nel caso in cui fosse necessario fare ee o smote 
def feature_scaling(data): # permette di fare il min max scaling della feature temporale
  '''
  in
  data: dataset
  out
  data: dataset il tempo scalato 
  '''
  from sklearn import preprocessing

  min_max_scaler = preprocessing.MinMaxScaler()
  time = data['TransactionDT'].values.reshape(-1,1)
  data['TransactionDT'] = min_max_scaler.fit_transform(time)
  data.drop(columns=['TransactionID'], inplace=True)
  return data


def ratio(data): # permette di ottenere il rate di eventi safe su quelli fraudolenti
  '''
  input 
  data: dataset
  output
  r: rate 
  '''
  fraud = (data['isFraud'] == 1).sum()
  safe = (data['isFraud'] == 0).sum()
  r = np.ceil(safe/fraud)
  return r

def encoding(data, cat_cols): # serve per il label encoding
  '''
  input
  data: dataset
  cat_cols: lista contenente i nomi delle colonne categoriche
  output
  data: dataset con le feature categoriche encoded
  categorical_names: dizionare che mappa il label encoding al feature value
  '''
  from sklearn import preprocessing
  categorical_col=load_list(cat_cols)
  categorical_features = [data.columns.get_loc(col) for col in categorical_col if col in data]
  categorical_names = {}
  for feature in categorical_features:
      le = preprocessing.LabelEncoder()
      le.fit(data.iloc[:, feature])
      data.iloc[:, feature] = le.transform(data.iloc[:, feature])
      categorical_names[feature] = le.classes_
  return data, categorical_names

# Riferimento: https://stackoverflow.com/questions/46966690/change-a-column-string-into-int-in-python-list
def to_list(data, cat_feature_list): # permette di ritornare i valori nel datfrae ome lista
  '''
  input
  data: dataset
  cat_feature_list: lista contente i nomi delle feature categoriche
  output
  mylist: dataset trasformato in lista, in cui gli elementi nelle colonne categoriche sono interi
  '''
  my_list = data.values.tolist()
  for row in my_list:
    for col in cat_feature_list:
      row[col] = int(row[col])
  return my_list


#####################################################################################################
## XAI ##
#####################################################################################################


def save_list(filename, list):  # salva la lista list nel path specificato assegnando filename come nome del file
  '''
  input
  filename: nome che si vuole assegnare al file contenente la lista da salvare
  list: lista che si vuole salvare
  '''
  import os
  if not os.path.isfile('/content/drive/MyDrive/Tesi_magistrale/Dataset/IEEE/Output/'+filename+'.txt'):
      with open(filename+'.txt', 'w') as f:
          for item in list:
              f.write("%s " % item)
  return


def load_list(filename, alg = None): # serve per caricare la lista con gli id delle righe del dataframe
  '''
  input
  filename: nome del file che contiene la lista che si vuole caricare
  alg: nome dell'algoritmo 
  output
  list: lista contente gli id delle righe del dataframe su cui l'algoritmo si è allenato/validato/testato
  '''
  if alg is None:       
    file = open('/content/drive/MyDrive/Tesi_magistrale/Dataset/IEEE/Output/'+filename+'.txt', "r")
  else:
    file = open('/content/drive/MyDrive/Tesi_magistrale/Dataset/IEEE/Output/'+alg+'/'+filename+'.txt', "r")
  list = file.read() # importo il file
  list = list.split(" ") # le colonne sono separate dallo spazio
  file.close() 
  list.pop() # levo l'ultimo elemento che è vuoto
  return list

def get_set(filename, data, alg, labels = False): # serve per selezionare il dataset dati gli id; bisogna usare lo stesso set usato nella parte di training del modello
  ''' 
  input
  filename: nome del file che contiene la lista degli id
  alg: nome dell'algoritmo
  data: train, test o val. Set che si vuole selezionare
  output
  X: set senza la label
  y: labels
  '''
  list = load_list(filename, alg)
  list = [int(i) for i in list] 
  X = data.iloc[list, :]
  if labels:
    y = X['isFraud']
    X.drop(columns=['isFraud'], axis = 1, inplace = True)
    return X, y
  else:
    X.drop(columns=['isFraud'], axis = 1, inplace = True)
    return X



#####################################################################################################
## Feature Engineering  ##
#####################################################################################################

# Riferimento: https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm
def feature_engineering(data):
  '''
  input
  data: dataset su cui performare il feature engineering
  output
  data: dataset connuove feature ingegnerizzate
  '''
  data = device_name(data) # raggruppo il nome del device 
  data = device(data) # seleziono la versione del device
  data = os_(data) # seleziono il nome e la versione del sistema operativo
  data = browser(data) # seleziono nome e versione del browser
  data = screen(data) # seleziono sreen heigth e width 
  data = id_23_34(data) # seleziono la seconda parte di questi id
  data.drop(columns = ['DeviceInfo', 'id_30', 'id_31', 'id_33', 'id_34', 'id_23'], inplace=True) # levo le colonne da cui ho creato le nuove features

  data = date(data) # costrusco una colonna con l'ora e una con il giorno corrispondente alla transazione
  return data

def device_name(data):
  data['device_name'] = data['DeviceInfo'].str.split('/', expand=True)[0]
  device_dict = {
      'SM': 'Samsung',
      'SAMSUNG': 'Samsung',
      'GT-': 'Samsung',
      'Moto G': 'Motorola',
      'Moto': 'Motorola',
      'moto': 'Motorola',
      'LG-': 'LG',
      'rv:': 'RV',
      'HUAWEI': 'Huawei',
      'ALE-': 'Huawei',
      '-L': 'Huawei',
      'Blade': 'ZTE',
      'BLADE': 'ZTE',
      'Linux': 'Linux',
      'XT': 'Sony',
      'HTC': 'HTC',
      'ASUS': 'Asus'
        }
  for device in device_dict:
    data.loc[data['device_name'].str.contains(device, na=False), 'device_name'] = device_dict[device]
  data.loc[data.device_name.isin(data.device_name.value_counts()[data.device_name.value_counts() < 200].index), 'device_name'] = "Others"
  return data

def device(data):
  data['device_version'] = data['DeviceInfo'].str.split('/', expand=True)[1]
  return data

def os_(data):
  data['os_name'] = data['id_30'].str.split(' ', expand=True)[0]
  data['os_version'] = data['id_30'].str.split(' ', expand=True)[1]
  return data

def browser(data):
  data['browser_name'] = data['id_31'].str.split(' ', expand=True)[0]
  data['browser_version'] = data['id_31'].str.split(' ', expand=True)[1]
  return data

def screen(data):
  data['screen_w'] = data['id_33'].str.split('x', expand=True)[0]
  data['screen_h'] = data['id_33'].str.split('x', expand=True)[1]
  return data

def id_23_34(data):
  data['id_23'] = data['id_23'].str.split(':', expand=True)[1]
  data['id_34'] = data['id_34'].str.split(':', expand=True)[1]  
  return data

def date(data):
  data['day'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1) % 7)
  data['hour'] = np.floor(data['TransactionDT'] / 3600) % 24
  return data

def fraud_ratio_per_time(data, day = False, hour = False): # 
  '''
  input
  data: dataset dopo la fase di feature_engineering
  day: se True, il conteggio viene fatto per giorni
  hour: se True, il conteggio viene fatto per ore
  output
  ratio: dizionario con (unità_di_tempo: rate)
  '''
  ratio = {}
  if day:
    col = 'day'
    n = 7
  if hour:
    col = 'hour'
    n = 24
  for i in range(n):
    fraud = data[data[col] == i].sum()[0]
    tot = data[data[col] == i].shape[0]
    ratio[i] = (fraud / tot).round(3)
  return ratio

def get_important_features_by_ratio(data, tresh=0.4): # seleziona le feature categoriche che individuano le transazioni fraudolente con rate superiore alla soglia 
  ''' 
  input
  data: dataset dopo la fase di one hoe encoding e feature engineering
  tresh: soglia sotto la quale eliminare le feature che presentano un rate inferiore 
  output
  important_features: dizionario contenente le feature con rate di transazioni fradolente ordinate per valore e sopra soglia
  feature_ratio: dizionario che contiene i rate per feature (feature: rate)
  '''
  cat_features = load_list('cat_sign_col') # lista con i nomi delle vecchie colonne contenenti le features categoriche 
  eng_features = ['device_version', 'os_name', 'os_version', 'browser_name', 'browser_version', 'screen_w', 'screen_h', 'day', 'hour'] # nomi delle nuove feature ottenute tramite la funzione feature_engineering
  cat_features_complete = cat_features + eng_features # lista con i nomi di tutte le features categoriche prima della fase di one hot encoding

  cat_cols = [] # lista che conterrà i nomi delle features categoriche dopo la fase di one hot encoding e feature engineering
  cols = data.columns.tolist()  # colonne presenti nel dataset

  for col in cols: # loop per ottenere la lista di feature categoriche 
    for string in cat_features_complete:
      if string in col: # controllo che il nome della feature pre encoding sia nella lista delle nuove features
        cat_cols.append(col)

  feature_ratio = {} # dizionario che conterrà il rate per feature (feature: rate)
  for col in cat_cols:
    feature_ratio[col] = fraud_ratio_per_feature(data, col)

  important_features = sorted(feature_ratio, key=feature_ratio.get, reverse=True) # ordino il dizionario per valore del rate

  for feature in feature_ratio: # loop per eliminare le feature con rate sotto soglia
    if feature_ratio[feature] < tresh:
      important_features.remove(feature)

  return important_features, feature_ratio