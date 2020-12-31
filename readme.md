# Tesi Magistrale
## Link utili
[Worklog](https://docs.google.com/document/d/1u_Q3iAA3DFf81A097LBxNq6zbCLvt9MxL3HV_aMxuLM/edit?usp=sharing): contiene il diario delle attività e gli altri link per la bibliografia e le tabelle. 

[Dataset](https://www.kaggle.com/c/ieee-fraud-detection): dataset utilizzato per l'analisi e la costruzione dei modelli.

## File list
### Dataset exploration
`1_mice.ipynb`: import e merge del dataset. Contiene la parte di imputation dei missing values, realizzata attreverso MICE. 

`2_eda.ipynb`: analisi esplorativa del dataset. Vengono selezionate le feature significative attraverso il test di Welch per le feature numeriche e il test $\chi^{2}$ per le feature categoriche. Infine is analizzano le correlazioni. Vedere il file `fraud_detection_with_corr_plots` nel folder `img` per i plot delle correlazioni. 

### Models
* Tutti i seguenti file hanno la parte di preprocessing necessaria per passare il dataset al modello specificato.
* Le performance dei modelli sono valutate attraverso la confusion matrix, la ROC-AUC e la mean average precision (AP). La scelta della AP come metrica di performance è data dalla necessità di far si che il modello tenga sotto controllo il numero di falsi positivi. 

`3_Autoencoder.ipynb`: costruzione di un autoencoder per fare anomaly detection sui primi 60 giorni. Si allena l'AE sulle transazioni safe, cercando di minimizzare l'errore di ricostruzione. Poi si valida l'AE sul validation set contenente anche le transazioni fraudolente, per scegliere il valore soglia dell'errore di ricostruzione. Infine si testa sul test set e si valutano le performance dell'algoritmo. 

`3_Neural_Net.ipynb`: costruzione di una rete neurale per fare classificazione. Dopo il training, si valutano le performance del modello. 

`3_Pre_&_CatBoost.ipynb`: costruzione di un modello ad ensemble, CatBoost, per fare classificazione. Col catboost si evita di fare il one hot encoding, che invece è necessario per tutti gli altri modelli. Dopo il training, esuito sull'intero dataset, si valutano le performance del modello. 

`3_XGBoost.ipynb`: costruzione di un modello ad ensemble, XGBoost, per fare classificazione. Training eseguito su tutto il dataset, vengono valutate le performance. C'è anche una fase di feature engineering che migliora sensibilmente le prestazioni. Al momento è il modello più performante. 
 
`3_SVM.ipynb`: costruzione di una support vector machine per fare classificazione. Viene implementata una LinearSVC di scikit learn, che si comporta meglio per dataset con più di 20-30 features. 

### XAI
`4_LIME.ipynb`: contiene l'implementazione di lime. Vengono caricati i modelli, e poi viene eseguito l'algoritmo. 

`4_SHAP.ipynb`: contiene l'implementazione del calcolo degli shapley values. Vengono caricati i modelli, e poi viene eseguito l'algoritmo. 

### Modulo 
`mf.py` contiene tutte le funzioni custom utilizzate nei notebook precedenti. 

## Folder list
`/img`: contiene i notebok eseguiti per vedere i plot. 

`/old`: contiene le versioni precedenti dei notebook. 

`/removed`: contiene i notebook con le parti rimosse.

`/trial`: contiene i notebook di parti provate ma implementate con scarso successo. 

