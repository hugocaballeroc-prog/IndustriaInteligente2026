"""HAI22.04 OSA árboles; no usa etiquetas TEST para ajustar modelos o umbrales."""
import os, time, warnings, platform, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experimento as base

SEED=7
# Ventana temporal 30 s, mismas cuatro posiciones para TODOS los árboles.
L=30
OFFSETS=(0,1,8,29)
MODELS=('DT','RF','ET','XGB')
CONFIGS=base.CONFIGS
TARGETS=base.OUTPUTS
Q=base.Q
SMOKE_ROWS=900

def config_path():
    # El directorio indicado por el usuario puede contener los CSV directamente.
    default=r'C:\Users\hugoc\Local\PUCV\Doctorado\Cursos\Semestre-2-2026\Industria-Inteligente\archive'
    os.environ.setdefault('HAI_DATA', default)
    os.environ.setdefault('HAI_OUTPUT',str(Path.cwd()/'resultados_hai_arboles'))
    return Path(os.environ['HAI_DATA'])

def raw_window(data,features):
    # X(t-29), X(t-8), X(t-1), X(t) -> Y(t+1)
    a=data[features].to_numpy(dtype=np.float32,copy=False)
    idx=np.arange(L-1,len(a)-1)
    x=np.concatenate([a[idx-o] for o in OFFSETS],axis=1).astype(np.float32,copy=False)
    y=data[TARGETS].iloc[idx+1].to_numpy(dtype=np.float32)
    return x,y,idx+1

def make_model(name, smoke=False):
    if name=='DT':return DecisionTreeRegressor(max_depth=12,min_samples_leaf=8,random_state=SEED)
    if name=='RF':return RandomForestRegressor(n_estimators=4 if smoke else 35,max_depth=14,min_samples_leaf=5,max_features=.5,max_samples=.7,n_jobs=2,random_state=SEED)
    if name=='ET':return ExtraTreesRegressor(n_estimators=4 if smoke else 35,max_depth=14,min_samples_leaf=5,max_features=.5,n_jobs=2,random_state=SEED)
    if name=='XGB':
        try: from xgboost import XGBRegressor
        except ImportError as e:raise ImportError('Instale xgboost: %pip install xgboost') from e
        one=XGBRegressor(n_estimators=4 if smoke else 65,max_depth=5,learning_rate=.07,subsample=.8,colsample_bytree=.8,tree_method='hist',n_jobs=2,random_state=SEED,objective='reg:squarederror')
        return MultiOutputRegressor(one,n_jobs=1)
    raise ValueError(name)

def save_predictions(model,method,conf,features,scaler,ymu,ysd,smoke=False):
    out=base.dirs()
    train=base.read_csv('train1.csv');cal=base.segments(train)['calibration']
    datasets={'calibration':cal}
    if not smoke:
        datasets.update({name:base.read_csv(name+'.csv') for name in ('test2','test3','test4','train2','train3','train4','train5','train6')})
    records=[]
    for name,data in datasets.items():
        if smoke:data=data.iloc[:SMOKE_ROWS].copy()
        raw,y,index=raw_window(data,features)
        pred=model.predict(scaler.transform(raw))*ysd+ymu
        pred=np.asarray(pred,dtype=np.float32)
        err=y-pred
        frame=pd.DataFrame({'archivo':name,'indice':index,'timestamp':data.timestamp.iloc[index].to_numpy()})
        if name.startswith('test'):frame['Attack']=data.Attack.iloc[index].to_numpy(dtype=np.int8)
        for j,k in enumerate(TARGETS):
            frame['real__'+k]=y[:,j];frame['pred__'+k]=pred[:,j];frame['error__'+k]=err[:,j]
            records.append({'metodo':method,'configuracion':conf,'archivo':name,'variable':k,'componente':k[:2],
                'rmse':float(np.sqrt(np.mean(err[:,j]**2))),'mae':float(np.mean(np.abs(err[:,j])))})
        frame.to_csv(out/f'predicciones/{method}_{conf}_{name}.csv.gz',index=False,compression='gzip')
        print(f'{method} {conf} {name}: {len(frame):,} OSA')
    pd.DataFrame(records).to_csv(out/f'metricas/prediccion_{method}_{conf}.csv',index=False)

def fit_one(method,conf='FULL',smoke=False):
    config_path();out=base.dirs()
    if method not in MODELS:raise ValueError(method)
    train,features_all=base.get_data();ss=base.segments(train)
    if smoke:ss={k:v.iloc[:SMOKE_ROWS].copy() for k,v in ss.items()}
    features=base.inputs(features_all,conf)
    xfit,yfit,_=raw_window(ss['fit'],features)
    xsel,ysel,_=raw_window(ss['selection'],features)
    scaler=StandardScaler().fit(xfit)
    xf=scaler.transform(xfit);xs=scaler.transform(xsel)
    ymu=yfit.mean(axis=0);ysd=np.maximum(yfit.std(axis=0),base.EPS)
    model=make_model(method,smoke)
    start=time.perf_counter(); model.fit(xf,(yfit-ymu)/ysd)
    duration=time.perf_counter()-start
    pred=np.asarray(model.predict(xs))*ysd+ymu
    er=ysel-pred
    metrics=pd.DataFrame([{'metodo':method,'configuracion':conf,'variable':k,'componente':k[:2],
        'rmse_selection':float(np.sqrt(np.mean(er[:,j]**2))),'mae_selection':float(np.mean(np.abs(er[:,j]))),
        'segundos_ajuste':duration} for j,k in enumerate(TARGETS)])
    metrics.to_csv(out/f'metricas/seleccion_{method}_{conf}.csv',index=False)
    if not smoke:
        dump({'model':model,'scaler':scaler,'ymean':ymu,'ystd':ysd,'features':features,'targets':TARGETS,'offsets':OFFSETS,'window':L,'seed':SEED,'method':method,'config':conf},out/f'modelos/{method}_{conf}.joblib',compress=3)
    base.write_json(out/f'logs/config_{method}_{conf}.json',{'method':method,'config':conf,'fit_rows':len(xfit),'selection_rows':len(xsel),
        'features':features,'targets':TARGETS,'seed':SEED,'lag_offsets':OFFSETS,'window':L,'model_parameters':model.get_params(),
        'elapsed_seconds':duration,'smoke':smoke,'python':platform.python_version()})
    save_predictions(model,method,conf,features,scaler,ymu,ysd,smoke)
    base.detect(method,conf,smoke=smoke)
    return metrics

def run_all(smoke=False):
    config_path()
    if not smoke:base.audit()
    for conf in (['FULL'] if smoke else CONFIGS):
        for model in MODELS:
            fit_one(model,conf,smoke)
    if not smoke:
        base.persistence(False);base.detect('PERSIST','FULL',False)
        base.summarize()

def audit():
    config_path()
    return base.audit()

def figure_selection():
    out=base.dirs();files=sorted((out/'metricas').glob('seleccion_*.csv'))
    if not files:raise FileNotFoundError('Faltan métricas de selección')
    data=pd.concat([pd.read_csv(x) for x in files]); data['experimento']=data.metodo+' '+data.configuracion
    q=data.groupby('experimento').rmse_selection.median().sort_values()
    plt.figure(figsize=(12,5));q.plot(kind='bar');plt.ylabel('Mediana RMSE selección (unidades mezcladas; solo orientativa)');plt.tight_layout();plt.savefig(out/'figuras/rmse_seleccion_arboles.png',dpi=170);plt.close()
    return data
