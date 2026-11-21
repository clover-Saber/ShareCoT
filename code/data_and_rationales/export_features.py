import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier,DMatrix
from protocol import FEATURE_COLS,TARGET,write_json

def main(suite):
    suite=Path(suite);src=suite/'experiments/02_main/data';out=suite/'experiments/01_features';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((src/'shap_metadata.json').read_text())
    if meta['fit_split']!='train_only':raise ValueError('SHAP provenance mismatch')
    train=pd.read_csv(src/'train.csv');x=train[FEATURE_COLS].copy()
    x['dropoff_poi_category']=x['dropoff_poi_category'].astype(str).map(meta['category_mapping']).fillna(-1)
    x['trip_start_timestamp']=pd.to_datetime(x.trip_start_timestamp).astype('int64')/1e9
    x=x.astype(float).fillna(meta['medians'])
    model=XGBClassifier();model.load_model(src/'shap_model.json')
    contrib=model.get_booster().predict(DMatrix(x),pred_contribs=True)
    pd.DataFrame({'feature':FEATURE_COLS,'mean_abs_shap':np.abs(contrib[:,:-1]).mean(0)}).sort_values('mean_abs_shap',ascending=False).to_csv(out/'global_importance.csv',index=False)
    local=pd.DataFrame(contrib[:,:-1],columns=FEATURE_COLS);local.insert(0,'sample_id',train.sample_id);local['base_log_odds']=contrib[:,-1];local['observed_label']=train[TARGET];local['probability']=model.predict_proba(x)[:,1];local.to_csv(out/'train_local_shap.csv',index=False)
    write_json(out/'manifest.json',{'status':'complete','fit_split':'train_only','training_rows':len(train),'features':FEATURE_COLS,'selection_policy':'fixed nine features remaining after deleting tip; no new all-candidate ranking claimed','source_shap_seed':meta['seed'],'student_seed':44,'export_complete':True})
    print('Exported no-tip global and local training SHAP',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);args=parser.parse_args();main(args.suite)
