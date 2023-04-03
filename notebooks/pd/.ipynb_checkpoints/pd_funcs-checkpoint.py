
# A set of functions specific to the PD implementation of IDEARs, possibly combine later on
import numpy as np

date_run='202304'
path_shaps='/Users/michaelallwright/Documents/data/ukb/pd/shaps/'
race_dict=dict({'British':'White',
 'Any other white background':'White',
 'Other ethnic group':'Others',
 'Irish':'White',
 'White':'White',
 'Prefer not to answer':'Others',
 'Any other mixed background':'Mixed',
 'African':'Black',
 'Indian':'South Asian',
 'White and Asian':'Mixed',
 'Pakistani':'South Asian',
 'Caribbean':'Black',
 'Chinese':'Chinese',
 'Any other Asian background':'South Asian',
 'White and Black African':'Mixed',
 'White and Black Caribbean':'Mixed',
 'Black or Black British':'Black',
 'Do not know':'Others',
 'Bangladeshi':'South Asian',
 'Any other Black background':'Black',
 'Mixed':'Mixed',
 'Asian or Asian British':'South Asian'})

def prep_data(df,eids_in,eids_inc_depvar,eids_exc_depvar,depvar):
    mask=(df['eid'].isin(eids_in))&~(df['eid'].isin(eids_exc_depvar))
    df_out=df.loc[mask,]

    df_out[depvar]=0
    df_out.loc[df_out['eid'].isin(eids_inc_depvar),depvar]=1
    
    return df_out

def make_inf_nan(df):
    
    # change infinity values to nan as these should be mean imputed
    for c in df.columns:
        mask=(df[c]==np.inf)
        df.loc[mask,c]=np.nan
    return df

def shap_multy(df,figname='test',depvar='PD',resize=1,resizeratio=10,meanshapmin=0.03,minrecs=3,rank=25,runs=10,barplots=1,
              holdout_ratio=0.5,df_val_use=None,preprocess=False):
    
    feats_full=[]
    aucs_full=[]
    for i in range(runs):
        rets=ds.shapruns_new(df=df,run='test',\
    remwords=ml.wordsremovePD+'|dementia|location|AD',depvar=depvar,resize=resize,resizeratio=resizeratio,perc=True,barplots=barplots,
                            holdout_ratio=holdout_ratio,df_val_use=df_val_use,preprocess=preprocess)

        feats=ml.shapgraphs_tuple(rets['shaps'],max_disp=30,figname='SHAP test',plot=False)
        feats['Variable2']=feats['Variable']
        feats['Variable']=feats['Variable'].map(ds.varmap)
        feats.loc[pd.isnull(feats['Variable']),'Variable']=feats['Variable2']
        feats_full.append(feats)
        aucs_full.append(rets['aucs'])
        
    #ml.boxplot_shap(df=feats_full,meanshapmin=meanshapmin,minrecs=minrecs,lab='all',figprint=True,figname=figname,format_file='.jpg',rank=rank,list=True)
    
    return feats_full,rets,aucs_full

def varmap(c):
    # maps variables to more readable form for output
    if c in ml.varmap:
        if ml.varmap[c]:
            c=ml.varmap[c]
    return c