"""HAI22.04 OSA; version 1.0. No se ajusta con datos TEST."""
import os, json, csv, hashlib, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGETS={'P1':['P1_FT01','P1_FT02','P1_FT03','P1_LIT01','P1_PIT01','P1_PIT02','P1_TIT01','P1_TIT02','P1_TIT03'], 'P2':['P2_SIT01','P2_VIBTR01','P2_VIBTR02','P2_VIBTR03','P2_VIBTR04','P2_VT01'], 'P3':['P3_FIT01','P3_LIT01','P3_PIT01'], 'P4':['P4_HT_FD','P4_HT_PO','P4_ST_FD','P4_ST_PO','P4_ST_PT01','P4_ST_TT01']}
OUTPUTS=sum(TARGETS.values(),[])
CONFIGS=['FULL','SIN_P1','SIN_P2','SIN_P3','SIN_P4']
DATA_FILES=['train1.csv','train2.csv','train3.csv','train4.csv','train5.csv','train6.csv','test1.csv','test2.csv','test3.csv','test4.csv']
SEED=7
# Ventana 30 segundos; la selección se restringe a offsets dentro de ella para Elastic Net.
L=30
OFFSETS=[0,1,2,4,8,16,29]
Q=0.995 # cuantil PREDEFINIDO; no se escoge por desempeño TEST
EPS=1e-6

def dirs():
    root=Path(os.environ.get('HAI_OUTPUT','resultados_hai')).resolve()
    for s in ['auditoria','modelos','predicciones','metricas','deteccion','figuras','logs']:(root/s).mkdir(parents=True,exist_ok=True)
    return root

def locate():
    p=os.environ.get('HAI_DATA','test2.zip')
    if not Path(p).exists(): raise FileNotFoundError(f'No existe HAI_DATA={p}. Establezca ruta al ZIP test2.zip o directorio CSV.')
    return Path(p)

def read_csv(name):
    p=locate()
    if p.is_dir(): d=pd.read_csv(p/name)
    else:
        import zipfile
        with zipfile.ZipFile(p) as z: d=pd.read_csv(z.open(name))
    return d

def cols(d):return [c for c in d if c.startswith(('P1_','P2_','P3_','P4_'))]
def inputs(c,config):return [x for x in c if config=='FULL' or not x.startswith(config.replace('SIN_','')+'_')]
def segments(d):
    n=len(d); a=int(n*.6);b=int(n*.8)
    return {'fit': d.iloc[:a].copy(),'selection':d.iloc[a:b].copy(),'calibration':d.iloc[b:].copy()}
def get_data():
    d=read_csv('train1.csv'); expected=cols(d)
    assert len(expected)==86 and all(x in expected for x in OUTPUTS)
    return d,expected

def audit():
    out=dirs();rows=[];headers=None
    for f in DATA_FILES:
        d=read_csv(f); t=pd.to_datetime(d['timestamp']); step=t.diff().dropna()
        if headers is None:headers=list(d.columns)
        if list(d.columns)!=headers:raise ValueError(f'Columnas diferentes: {f}')
        rec={'archivo':f,'filas':len(d),'columnas':len(d.columns),'faltantes':int(d.isna().sum().sum()),'inicio':str(t.iloc[0]),'fin':str(t.iloc[-1]),'duplicados_timestamp':int(t.duplicated().sum()),'intervalos_irregulares':int((step!=pd.Timedelta(seconds=1)).sum()),'ataques_filas':int(d.Attack.sum()),'episodios':int(((d.Attack.eq(1)) & (~d.Attack.shift(fill_value=0).eq(1))).sum())}
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/'auditoria/inventario.csv',index=False)
    pd.DataFrame([{'componente':k,'salida':v} for k,vals in TARGETS.items() for v in vals]).to_csv(out/'auditoria/salidas.csv',index=False)
    write_json(out/'auditoria/protocolo.json',{'training':'train1 60% fit, 20% selection, 20% calibration','test':['test2.csv','test3.csv','test4.csv'],'other':['test1 retrospectivo excluido','train2-6 posterior solo estabilidad'],'configs':CONFIGS,'osa_window':L,'enet_offsets':OFFSETS,'threshold_quantile':Q,'targets':TARGETS,'seed':SEED})
    return pd.DataFrame(rows)

def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf8')
def atomic_csv(df,p):df.to_csv(p,index=False)

def prepare(tab, features, xsc, ysc=None, mode='EN'):
    # Estrictamente X[:t] -> y[t+1]. No se cruzan segmentos.
    arr=tab[features].to_numpy(dtype=np.float32)
    if mode=='EN':
        ix=np.arange(L-1,len(tab)-1)
        X=np.hstack([arr[ix-o] for o in OFFSETS])
        X=xsc.transform(X).astype(np.float32)
    else:
        ix=np.arange(L-1,len(tab)-1)
        X=np.stack([arr[t-L+1:t+1] for t in ix])
        X=((X-xsc.mean_.astype(np.float32))/np.maximum(xsc.scale_,EPS).astype(np.float32)).astype(np.float32)
    y=tab[OUTPUTS].iloc[ix+1].to_numpy(dtype=np.float32)
    return X,y,ix+1

def fit_en(config,smoke=False):
    from joblib import dump
    out=dirs();d,c=get_data();s=segments(d); ff=inputs(c,config)
    if smoke:s={k:v.iloc[:max(1000,min(len(v),1600))].copy() for k,v in s.items()}
    # train only scaler, constant features handled by StandardScaler
    raw=np.hstack([s['fit'][ff].to_numpy(dtype=np.float32)[L-1-o:-1-o if o else -1] for o in OFFSETS])
    scaler=StandardScaler().fit(raw); xfit,yfit,_=prepare(s['fit'],ff,scaler)
    xsel,ysel,_=prepare(s['selection'],ff,scaler)
    best=[];preds=[];r=[]
    alphas=[0.001,0.01] if not smoke else [0.01]
    for j,target in enumerate(OUTPUTS):
        # escala salida sobre segmento fit; no hay información de selección/calibración
        mu=float(yfit[:,j].mean());std=float(yfit[:,j].std());std=max(std,EPS)
        local=[]
        for alpha in alphas:
            mdl=ElasticNet(alpha=alpha,l1_ratio=.2,max_iter=1200,tol=.002,selection='cyclic',random_state=SEED)
            mdl.fit(xfit,(yfit[:,j]-mu)/std)
            pred=mdl.predict(xsel)*std+mu
            local.append((np.sqrt(mean_squared_error(ysel[:,j],pred)),alpha,mdl))
        rmse,alpha,mdl=min(local,key=lambda z:z[0]); best.append((mdl,mu,std));r.append({'config':config,'target':target,'rmse_selection':rmse,'alpha':alpha})
    dump({'models':best,'scaler':scaler,'features':ff,'config':config,'window':L,'offsets':OFFSETS},out/f'modelos/EN_{config}.joblib')
    atomic_csv(pd.DataFrame(r),out/f'metricas/seleccion_EN_{config}.csv')
    evaluate_predictions('EN',config,lambda X:np.column_stack([mdl.predict(X)*sd+mu for mdl,mu,sd in best]),ff,scaler,smoke)

def evaluate_predictions(method,config,predict,features,scaler,smoke=False):
    out=dirs(); d,_=get_data(); sets={'calibration':segments(d)['calibration']}
    if not smoke:
        for file in ['test2.csv','test3.csv','test4.csv','train2.csv','train3.csv','train4.csv','train5.csv','train6.csv']:
            sets[file.replace('.csv','')]=read_csv(file)
    metrics=[]
    for name,tab in sets.items():
        X,y,indices=prepare(tab,features,scaler, mode='EN' if method=='EN' else 'TCN')
        pred=predict(X).astype(np.float32)
        if pred.shape!=y.shape:raise ValueError((pred.shape,y.shape))
        # Ancho, una fila por t; Attack SOLO guardada en predicciones TEST, nunca usada en ajustes
        residual=(y-pred)
        frame=pd.DataFrame({'archivo':name,'indice':indices,'timestamp':tab.timestamp.iloc[indices].values})
        if name.startswith('test'):frame['Attack']=tab.Attack.iloc[indices].to_numpy(dtype=int)
        for j,target in enumerate(OUTPUTS):
            frame[f'real__{target}']=y[:,j];frame[f'pred__{target}']=pred[:,j];frame[f'error__{target}']=residual[:,j]
            metrics.append({'metodo':method,'configuracion':config,'archivo':name,'variable':target,'componente':target[:2],'rmse':float(np.sqrt(np.mean(residual[:,j]**2))),'mae':float(np.mean(np.abs(residual[:,j])))})
        atomic_csv(frame,out/f'predicciones/{method}_{config}_{name}.csv.gz')
        print(f'{method} {config} {name}: {len(frame):,} pronosticos')
    atomic_csv(pd.DataFrame(metrics),out/f'metricas/prediccion_{method}_{config}.csv')

def train_tcn(config,smoke=False):
    import torch
    from torch import nn
    torch.manual_seed(SEED);np.random.seed(SEED)
    out=dirs();d,c=get_data();s=segments(d);ff=inputs(c,config)
    if smoke:s={k:v.iloc[:max(1200,min(len(v),1400))].copy() for k,v in s.items()}
    scaler=StandardScaler().fit(s['fit'][ff].to_numpy(dtype=np.float32))
    x,y,_=prepare(s['fit'],ff,scaler,mode='TCN');xs,ys,_=prepare(s['selection'],ff,scaler,mode='TCN')
    ymu=y.mean(axis=0);ysd=np.maximum(y.std(axis=0),EPS)
    class Block(nn.Module):
        def __init__(self,cin,cout,dil):
            super().__init__();self.c1=nn.Conv1d(cin,cout,3,dilation=dil,padding=2*dil);self.c2=nn.Conv1d(cout,cout,3,dilation=dil,padding=2*dil);self.short=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity();self.dil=dil;self.act=nn.ReLU()
        def forward(self,x):
            a=self.act(self.c1(x)[...,:x.shape[-1]]);b=self.c2(a)[...,:x.shape[-1]]
            return self.act(b+self.short(x))
    class TCN(nn.Module):
        def __init__(self,n):
            super().__init__();self.net=nn.Sequential(Block(n,16,1),Block(16,16,2),Block(16,16,4),Block(16,16,8));self.out=nn.Linear(16,len(OUTPUTS))
        def forward(self,x):return self.out(self.net(x.transpose(1,2))[:,:,-1])
    device='cuda' if torch.cuda.is_available() else 'cpu';model=TCN(len(ff)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001);lossfn=nn.MSELoss();batch=256;epochs=2 if smoke else 35;patience=5
    best=float('inf');bad=0;log=[];state=None
    for epoch in range(epochs):
        model.train();ids=np.random.default_rng(SEED+epoch).permutation(len(x));losses=[]
        for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/batch)))):
            xx=torch.from_numpy(x[chunk]).to(device);yy=torch.from_numpy(((y[chunk]-ymu)/ysd)).to(device)
            opt.zero_grad();loss=lossfn(model(xx),yy);loss.backward();opt.step();losses.append(loss.item())
        model.eval();v=[]
        with torch.no_grad():
            for i in range(0,len(xs),batch):
                yp=model(torch.from_numpy(xs[i:i+batch]).to(device)).cpu().numpy();v.append(np.mean((yp-(ys[i:i+batch]-ymu)/ysd)**2))
        vl=float(np.mean(v));log.append({'epoch':epoch+1,'loss_train':float(np.mean(losses)),'loss_selection':vl})
        print('TCN',config,epoch+1,vl)
        if vl<best-1e-5:
            best=vl;bad=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience:break
    model.load_state_dict(state);model.eval()
    torch.save({'state':state,'features':ff,'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,'ymean':ymu,'ystd':ysd,'config':config,'seed':SEED},out/f'modelos/TCN_{config}.pt')
    atomic_csv(pd.DataFrame(log),out/f'logs/entrenamiento_TCN_{config}.csv')
    plt.figure(figsize=(7,4));plt.plot([r['epoch'] for r in log],[r['loss_selection'] for r in log]);plt.xlabel('Época');plt.ylabel('MSE selección');plt.tight_layout();plt.savefig(out/f'figuras/aprendizaje_TCN_{config}.png',dpi=160);plt.close()
    def predictor(data):
        p=[]
        with torch.no_grad():
            for i in range(0,len(data),1024):p.append(model(torch.from_numpy(data[i:i+1024]).to(device)).cpu().numpy())
        return np.vstack(p)*ysd+ymu
    evaluate_predictions('TCN',config,predictor,ff,scaler,smoke)

def detect(method,config,smoke=False):
    out=dirs();p=out/f'predicciones/{method}_{config}_calibration.csv.gz'
    if not p.exists():raise FileNotFoundError(f'Primero entrenar: {p}')
    cal=pd.read_csv(p);errors=np.column_stack([cal['error__'+j].to_numpy() for j in OUTPUTS]);mu=errors.mean(axis=0);std=np.maximum(errors.std(axis=0),EPS)
    score=lambda e:np.sqrt(np.mean(((e-mu)/std)**2,axis=1))
    sc=score(errors);threshold=float(np.quantile(sc,Q));write_json(out/f'deteccion/umbral_{method}_{config}.json',{'cuantil':Q,'umbral':threshold,'mu':mu.tolist(),'std':std.tolist(),'variables':OUTPUTS,'calibracion':'train1 ultimos 20%, sin etiquetas'})
    files=['calibration'] if smoke else ['calibration','test2','test3','test4','train2','train3','train4','train5','train6']
    all_metrics=[];episodes=[]
    for name in files:
        d=pd.read_csv(out/f'predicciones/{method}_{config}_{name}.csv.gz')
        er=np.column_stack([d['error__'+j].to_numpy() for j in OUTPUTS]);s=score(er);alerta=s>threshold
        a=pd.DataFrame({'archivo':name,'indice':d.indice,'timestamp':d.timestamp,'score':s,'alerta':alerta.astype(int)})
        if 'Attack' in d:a['Attack']=d.Attack
        atomic_csv(a,out/f'deteccion/{method}_{config}_{name}.csv.gz')
        row={'metodo':method,'config':config,'archivo':name,'n':len(a),'n_alerta':int(alerta.sum()),'fraccion_alerta':float(alerta.mean())}
        if 'Attack' in a:
            labels=a.Attack.to_numpy(dtype=int)
            # Es posible que una ventana tenga un inicio de episodio previo. No afirmar que hay detección si ocurre en los primeros L pasos.
            starts=np.flatnonzero((labels==1)&(np.r_[0,labels[:-1]]==0));ends=np.flatnonzero((labels==1)&(np.r_[labels[1:],0]==0))
            for st,en in zip(starts,ends):episodes.append({'metodo':method,'config':config,'archivo':name,'inicio_indice':int(a.indice.iloc[st]),'fin_indice':int(a.indice.iloc[en]),'detectado':int(alerta[st:en+1].any())})
            row['episodios_visibles']=len(starts);row['episodios_detectados']=sum(alerta[st:en+1].any() for st,en in zip(starts,ends));row['alertas_fuera_etiqueta']=int((alerta & (labels==0)).sum())
        all_metrics.append(row)
        plt.figure(figsize=(12,3));idx=np.arange(len(s));down=max(1,len(s)//6000)
        plt.plot(idx[::down],s[::down],lw=.6);plt.axhline(threshold,color='gray',ls='--');plt.title(f'{method} {config} {name}');plt.xlabel('Observaciones');plt.ylabel('Puntuación OSA');plt.tight_layout();plt.savefig(out/f'figuras/score_{method}_{config}_{name}.png',dpi=150);plt.close()
    atomic_csv(pd.DataFrame(all_metrics),out/f'metricas/deteccion_{method}_{config}.csv')
    atomic_csv(pd.DataFrame(episodes),out/f'metricas/episodios_{method}_{config}.csv')

def summarize():
    out=dirs();metrics=list((out/'metricas').glob('deteccion_*.csv'))
    if not metrics:raise ValueError('No existen métricas de detección todavía.')
    d=pd.concat([pd.read_csv(x) for x in metrics],ignore_index=True)
    atomic_csv(d,out/'metricas/comparacion_deteccion.csv')
    fig=d[d.archivo.str.startswith('test')].copy()
    if len(fig):
        fig['etiqueta']=fig.metodo+' '+fig.config
        agg=fig.groupby('etiqueta').agg({'episodios_detectados':'sum','episodios_visibles':'sum'}).dropna().sort_index()
        agg['cobertura']=agg.episodios_detectados/agg.episodios_visibles
        ax=agg.cobertura.plot(kind='bar',figsize=(11,4));ax.set_ylabel('Cobertura de episodios (solo etiquetas de ataque)');ax.set_ylim(0,1);plt.tight_layout();plt.savefig(out/'figuras/comparacion_ablaciones.png',dpi=170);plt.close()
    return d

def persistence(smoke=False):
    """Referencia x_hat[t+1,j] = x[t,j], sin ajuste de parámetros."""
    out=dirs();d,_=get_data(); sets={'calibration':segments(d)['calibration']}
    if smoke:sets={'calibration':sets['calibration'].iloc[:1600]}
    else:
        for file in ['test2.csv','test3.csv','test4.csv','train2.csv','train3.csv','train4.csv','train5.csv','train6.csv']:
            sets[file.replace('.csv','')]=read_csv(file)
    rows=[]
    for name,tab in sets.items():
        indices=np.arange(L,len(tab));actual=tab[OUTPUTS].iloc[indices].to_numpy(dtype=np.float32)
        pred=tab[OUTPUTS].iloc[indices-1].to_numpy(dtype=np.float32)
        frame=pd.DataFrame({'archivo':name,'indice':indices,'timestamp':tab.timestamp.iloc[indices].to_numpy()})
        if name.startswith('test'):frame['Attack']=tab.Attack.iloc[indices].to_numpy(dtype=int)
        for j,target in enumerate(OUTPUTS):
            e=actual[:,j]-pred[:,j]
            frame[f'real__{target}']=actual[:,j];frame[f'pred__{target}']=pred[:,j];frame[f'error__{target}']=e
            rows.append({'metodo':'PERSIST','configuracion':'FULL','archivo':name,'variable':target,'componente':target[:2],'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e)))})
        atomic_csv(frame,out/f'predicciones/PERSIST_FULL_{name}.csv.gz')
    atomic_csv(pd.DataFrame(rows),out/'metricas/prediccion_PERSIST_FULL.csv')
