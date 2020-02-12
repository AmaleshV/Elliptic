# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 07:25:25 2020

@author: AMALESH
"""

from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
import node2vec
import numpy as np
import numpy
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,cohen_kappa_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import ParameterGrid, ParameterSampler
from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynAE        import DynAE
from time import time
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from scipy.stats import itemfreq


edges = pd.read_csv(askopenfilename(),engine='python')
features = pd.read_csv(askopenfilename(),engine='python',header=None)
classes = pd.read_csv(askopenfilename(),engine='python')

#3
len(edges),len(features),len(classes)
#4
display(edges.head(5),features.head(5),classes.head(5))
#5
tx_features = ["tx_feat_"+str(i) for i in range(2,95)]
agg_features = ["agg_feat_"+str(i) for i in range(1,73)]
features.columns = ["txId","time_step"] + tx_features + agg_features
features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)
#6
features.groupby('class').size()
#7
count_by_class = features[["time_step",'class']].groupby(['time_step','class']).size().to_frame().reset_index()
illicit_count = count_by_class[count_by_class['class'] == '1']
licit_count = count_by_class[count_by_class['class'] == '2']
unknown_count = count_by_class[count_by_class['class'] == "0"]

#9
ids = features['txId']
short_edges = edges[edges['txId1'].isin(ids)]
graph = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.DiGraph())
#pos = nx.spring_layout(graph)
nx.write_gpickle(graph, r"C:\Users\AMALESH\Desktop\ellip.gpickle")
graph= nx.read_gpickle(r"C:\Users\AMALESH\Desktop\ellip.gpickle")
#hyper_node2vec = {
#        'p': [0.25,0.5,0.75,1],
#        'q': [0.1, 0.5, 1, 2, 5, 10, 100]
#        }
#
#h_node2vec = Hyperconfig('grid_search', hyper_node2vec)
#h_node2vec.get_grid()
#h_node2vec.fit_loop_node2vec(updat)
#
#for p,q in zip(range(i),range(j)):
#    print(p,q)
#for p in i:
#    for q in j:
#        print(p,q)
#DG= graph
#DG.add_weighted_edges_from(add_weighted_edges)

i= [0.25,0.5]
j= [ 0.1,0.5,1,10, 100]
for p in i:
    for q in j:
        node2vec = Node2Vec(graph, dimensions = 128, walk_length=20, 
                                        num_walks = 50, p=p, q=q, 
                                        workers=1)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        node2vec_dict = {}
        for node in graph.nodes():
            node = str(node)
            node2vec_dict[node] = model.wv[node]
        node2vec_pd = pd.DataFrame.from_dict(node2vec_dict, orient='index')
        #node2vec_pd.to_csv(r'C:\Users\AMALESH\Desktop\node2vec.csv')
        
        
        x=node2vec_pd
        classifiers = {
            "Isolation Forest":IsolationForest(n_estimators=500, max_samples=len(x), 
                                               contamination=0.108,random_state=123, verbose=0,behaviour="new"),
            "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                                      leaf_size=30, metric='minkowski',
                                                      p=2, metric_params=None, contamination=0.108),
        }
        #n_outliers = len(SAR)
        for i, (clf_name,clf) in enumerate(classifiers.items()):
            #Fit the data and tag outliers
            if clf_name == "Local Outlier Factor":
                y_pred = clf.fit_predict(x)
                scores_prediction = clf.negative_outlier_factor_
            else:    
                clf.fit(x)
                scores_prediction = clf.decision_function(x)
                y_pred = clf.predict(x)
            #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
            y_pred[y_pred == 1] = 0
            y_pred[y_pred == -1] = 1
            y_num= classes['class']
        
            y_num=y_num.replace('unknown', numpy.NaN)
            y_num=y_num.replace('2','0' )
        
            y_num= y_num.to_frame()
            y_num['y_pred']= y_pred
        
            y_num=y_num.dropna()
        
            y_num['class']=pd.to_numeric(y_num['class'])
            y_num['y_pred']=pd.to_numeric(y_num['y_pred'])
            n_errors = (y_num['y_pred'] != y_num['class']).sum()
            # Run Classification Metrics
            print("node2vec parameters p:{},q:{}".format(node2vec.p,node2vec.q), file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print("{}: {}".format(clf_name,n_errors), file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print("Accuracy Score :", file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print(accuracy_score(y_num['class'],y_num['y_pred']), file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print("Classification Report :", file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print(classification_report(y_num['class'],y_num['y_pred']), file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print("confusion matrix:", file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print(confusion_matrix(y_num['class'],y_num['y_pred']), file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print("ROC_AUC:", file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
            print(roc_auc_score(y_num['class'],y_num['y_pred']), file=open(r"C:\Users\AMALESH\Desktop\output.txt", "a"))
