from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import numpy as np
import functools
from typing import List, Union
import os
import requests
import zipfile
import io
from itertools import combinations

from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from typing import Literal
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
import pandas as pd

this_dir, this_filename = os.path.split(__file__) 


SamplingScheme = Literal['rand', 'stratified', '1vN', 'MvN']
CLASSIFIERS = [Ridge,LinearRegression,BayesianRidge,MLPRegressor,GradientBoostingRegressor,RandomForestRegressor,HistGradientBoostingRegressor,SVR]
CLASSIFIER_NAMES = ['Ridge','LinearRegression','BayesianRidge','MLPRegressor','GradientBoostingRegressor','RandomForestRegressor', 'HistGradientBoostingRegressor','SVR']
HPS = {
    'Ridge' : {
        'estimator__alpha' : [0.1,0.5,1,3],
        'estimator__solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
        'estimator__fit_intercept' : [True,False]
    },
    'LinearRegression' : { 
        'estimator__fit_intercept' : [True,False]
    },
    'BayesianRidge' : {
        'estimator__alpha_init' : [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
        'estimator__lambda_init' : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9],
    },
    'MLPRegressor' : {
        'estimator__hidden_layer_sizes' : [(256,128,64),(128,64),(128)],
        'estimator__activation' : ['identity','logistic','tanh','relu'],
        'estimator__solver' : ['lbfgs', 'sgd', 'adam'],
        'estimator__learning_rate' : ['constant','adaptive'],
        'estimator__max_iter' : [500]
    },
    'GradientBoostingRegressor' : {
    },
    'RandomForestRegressor' : {
        'estimator__n_estimators' : [100,200,300,400,500],
        'estimator__max_depth' : [10,20,30,40,50],
        'estimator__min_samples_split' : [2,5,10],
        'estimator__min_samples_leaf' : [1,2,4],
        'estimator__bootstrap' : [True,False]
    },
    'HistGradientBoostingRegressor' : {
        'estimator__max_iter' : [100,200,300,400,500],
        'estimator__max_depth' : [10,20,30,40,50],
        'estimator__min_samples_leaf' : [1,2,4],
        'estimator__l2_regularization' : [0.0,0.1,0.5,1.0]
    },
    'SVR' : {
        'estimator__kernel' : ['linear', 'rbf', 'sigmoid', 'precomputed'],
        'estimator__gamma' : ['auto', 'scale'],
        'estimator__epsilon' : [0.1,0.5,1.0,3.0],
        'estimator__C' : [0.01,0.1,0.5,1]
    }
}

def get_remote_data(remote_data: str):
    r = requests.get(remote_data)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(Path(this_dir) / 'data')
    z.close()

@functools.lru_cache(maxsize=None)
def get_histogram(file_name: str):
    img_base_path = Path('data') / 'Original'
    img = Image.open(img_base_path / file_name)
    # get histogram
    og_hist = np.array(img.histogram())
    # normalize
    og_hist = og_hist / og_hist.sum()
    return og_hist

def load_and_partition_data(
        df: Union[str, pd.DataFrame] = Path('data') / 'gt.csv', 
        train_proteins: Union[List,str] = None, 
        test_size=0.2, 
        strategy: SamplingScheme = 'rand' , 
        random_state=42):
    
    if isinstance(df, str):
        df = pd.read_csv(df)
    
    df = df
    # remove empty images
    df = df.dropna()
    y = df[['High', 'Low', 'protein']]
    

    if strategy == 'rand':
        # rand split
        X = df.drop(['High', 'Low', 'protein'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y, 
            test_size=test_size, 
            random_state=random_state)

    elif strategy == 'stratified':
        X = df.drop(['High', 'Low', 'protein'], axis=1)
        # stratified split on protein channel
        X_train, X_test, y_train, y_test =  train_test_split(
            X,
            y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y.protein)

    elif strategy == '1vN':
        # 1vN split
        assert train_proteins is not None, 'train_proteins must be provided for 1vN split'
        assert isinstance(train_proteins, str), 'train_proteins must be a string for 1vN split'
        X = df
        X_train = X[X.protein == train_proteins].drop(['High', 'Low','protein'], axis=1)
        y_train = y[y.protein == train_proteins]
        X_test = X[X.protein != train_proteins].drop(['High', 'Low','protein'], axis=1)
        y_test = y[y.protein != train_proteins]
    
    elif strategy == 'MvN':
        # MvN split
        assert train_proteins is not None, 'train_proteins must be provided for MvN split'
        assert isinstance(train_proteins, list), 'train_proteins must be a list for MvN split'
        X = df
        X_train = X[X.protein.isin(train_proteins)].drop(['High', 'Low','protein'], axis=1)
        y_train = y[y.protein.isin(train_proteins)]
        X_test = X[~X.protein.isin(train_proteins)].drop(['High', 'Low','protein'], axis=1)
        y_test = y[~y.protein.isin(train_proteins)]
    else:
        raise ValueError(f'Invalid sampling strategy: {strategy}')
    

    return (np.array(X_train.ImageID.apply(get_histogram).tolist()),
    np.array(X_test.ImageID.apply(get_histogram).tolist()),
    y_train.drop('protein', axis=1).to_numpy(),
    y_test.drop('protein', axis=1).to_numpy())




def get_all_combinations(lst):
    result = []
    for i in range(1, len(lst)):
        for combo in combinations(lst, i):
            m = list(combo)
            n = [item for item in lst if item not in m]
            result.append((m, n))
    return result

def _fit_model(X_train, y_train, X_test, y_test,cv=5):
    cnames = []
    cbestparams = []
    ctrain_scores = []
    ctest_scores = []
    

    for clf,clf_name in zip(CLASSIFIERS,CLASSIFIER_NAMES):
        final_clf = GridSearchCV(
            MultiOutputRegressor(clf()),
            cv=5,param_grid=HPS[clf_name],
            n_jobs=1,
            verbose=3).fit(X_train,y_train)


        cnames.append(clf_name)
        cbestparams.append(final_clf.best_params_)
        ctrain_scores.append(final_clf.score(X_train,y_train))
        ctest_scores.append(final_clf.score(X_test,y_test))

    return cnames, cbestparams, ctrain_scores, ctest_scores


def _train_model(sampling_strategy, results_dir, test_size=0.2):
    # 

    if sampling_strategy == '1vN':
        _cnames, _cbestparams, _ctrain_scores, _ctest_scores, P1, SS = [],[],[],[],[],[]
        all_p = pd.read_csv(Path('data') / 'gt.csv').protein.unique()
        for p in all_p:
            X_train, X_test, y_train, y_test = load_and_partition_data(
                                                        df=Path('data') / 'gt.csv',
                                                        strategy=sampling_strategy,
                                                        train_proteins=p
                                                    )
            
            cnames, cbestparams, ctrain_scores, ctest_scores = _fit_model(X_train, y_train, X_test, y_test)
            _cnames.extend(cnames)
            _cbestparams.extend(cbestparams)
            _ctrain_scores.extend(ctrain_scores)
            _ctest_scores.extend(ctest_scores)
            P1.extend([p]*len(cnames))
            SS.extend([sampling_strategy]*len(cnames))
            

        
        pd.DataFrame({
            'Protein_Trained' : P1,
            'SamplingStrategy' : SS,
            'Classifier' : _cnames,
            'BestParams' : _cbestparams,
            'TrainScores' : _ctrain_scores,
            'TestScores' : _ctest_scores
        }).to_csv(results_dir + '/1vN_results.csv', index=False)




    
    elif sampling_strategy == 'MvN':
        all_p = pd.read_csv(Path('data') / 'gt.csv').protein.unique()
        _cnames, _cbestparams, _ctrain_scores, _ctest_scores, PN, SS = [],[],[],[],[],[]
        for p in get_all_combinations(all_p):
            X_train, X_test, y_train, y_test = load_and_partition_data(
                                                        df=Path('data') / 'gt.csv',
                                                        strategy=sampling_strategy,
                                                        train_proteins=p[0]
                                                    )
            
            cnames, cbestparams, ctrain_scores, ctest_scores = _fit_model(
                X_train, 
                y_train, 
                X_test, 
                y_test)
            
            _cnames.extend(cnames)
            _cbestparams.extend(cbestparams)
            _ctrain_scores.extend(ctrain_scores)
            _ctest_scores.extend(ctest_scores)
            PN.extend([p]*len(cnames))
            SS.extend([sampling_strategy]*len(cnames))

        pd.DataFrame({
            'Proteins_Trained' : PN,
            'SamplingStrategy' : SS,
            'Classifier' : _cnames,
            'BestParams' : _cbestparams,
            'TrainScores' : _ctrain_scores,
            'TestScores' : _ctest_scores
        }).to_csv(results_dir + '/MvN_results.csv', index=False)


    else:
        X_train, X_test, y_train, y_test = load_and_partition_data(
            df=Path('data') / 'gt.csv',
            strategy=sampling_strategy,
            test_size=test_size
        )


        cnames, cbestparams, ctrain_scores, ctest_scores = _fit_model(
            X_train, 
            y_train,
            X_test, 
            y_test)
        
        pd.DataFrame({
            'SamplingStrategy' : [sampling_strategy]*len(cnames),
            'Classifier' : cnames,
            'BestParams' : cbestparams,
            'TrainScores' : ctrain_scores,
            'TestScores' : ctest_scores
        }).to_csv(results_dir + f'/{sampling_strategy}_results.csv', index=False)
        

def train_model(sampling_strategy, test_size=0.2, results_dir='results'):
    if sampling_strategy == 'all':
            for strat in SamplingScheme.__args__:
                _train_model(strat,results_dir, test_size=test_size)

    else:
        _train_model(sampling_strategy,results_dir, test_size=test_size)


    

    