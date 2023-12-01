#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
statistics
----------
Version 1.4
-------------------
*Added randomlist to tools
*Added fit_transform to onehot encoder
*Corrected description of partial correlation
*Fixed the following bugs:
    -scatter3d
    -class describe.data
    -scatterplot
    -correlation matrix
    -regression
*Fixed variable name bug in regression class
*Fixed object bug in onehot class
    
"""


# In[ ]:


from lib import *


# In[ ]:


'''
Class tools
-----------
'''
class tools:
    def rlist(length,ulimit,llimit=1):
        randomlist = []
        for i in range(0,length):
            n = random.randint(1,ulimit)
            randomlist.append(n)
        return randomlist
    def stars(number):
        if number<0.0001:
            return '****'
        elif 0.0001<=number<0.001:
            return '***'
        elif 0.001<=number<0.01:
            return '**'
        elif 0.01<=number<0.05:
            return '*'
        else:
            return 'not sig'
    def cd_pairs(x,y):
        #preliminaries
        ctab=pd.crosstab(x,y)
        conctab=ctab.copy()
        disctab=ctab.copy()
        col=list(ctab.columns)
        row=list(ctab.index)
        for i in range(ctab.shape[0]):
            for j in range(ctab.shape[1]):
                conctab.loc[row[i],col[j]]=0
                disctab.loc[row[i],col[j]]=0
        #count cell pairs
        def cell_pairs(ctab,n,m,method='conc'):
            col=list(ctab.columns)
            row=list(ctab.index)
            counts=[]
            for i in range(ctab.shape[0]):
                for j in range(ctab.shape[1]):
                    if method=='conc':
                        if (n-i)*(m-j)>0:
                            counts.append(ctab.loc[row[i],col[j]])
                        else:
                            pass
                    else:
                        if (n-i)*(m-j)<0:
                            counts.append(ctab.loc[row[i],col[j]])
                        else:
                            pass
            return pd.Series(counts).sum()#*ctab.loc[row[n],col[m]]

        for i in range(ctab.shape[0]):
            for j in range(ctab.shape[1]):
                conctab.loc[row[i],col[j]]=cell_pairs(ctab,i,j,method='conc')
                disctab.loc[row[i],col[j]]=cell_pairs(ctab,i,j,method='disc')

        conc=conctab*ctab   
        disc=disctab*ctab  
        conc=conc.to_numpy().sum()
        disc=disc.to_numpy().sum()

        return conc/2,disc/2,conctab,disctab
    
    def extreme(x,method,k='auto'):
        if method=='iqr':
            if k=='auto':
                k=1.5
            else:
                pass
            q1, q3 = np.quantile(x,0.25),np.quantile(x,0.75)
            iqr=q3-q1
            median=x.median()
            l,u=median-k*iqr, median+k*iqr
            return l,u
        elif method=='mad':
            if k=='auto':
                k=math.sqrt(stats.chi2.ppf(0.975,df=1))
            elif k=='norm':
                k=1/stats.norm.ppf(0.75)
            else:
                pass
            median=x.median()
            x_mad=(x-x.median()).abs()
            mad=x_mad.median()
            l,u=median-k*mad, median+k*mad
            return l,u
        elif method=='zscore':
            if k=='auto':
                k=3
            else:
                pass
            mean=x.mean()
            std=x.std()
            l,u=mean-k*std, mean+k*std
            return l,u
        else:
            pass
    
    def mahalanobis(X,i):
        col=list(X.columns)
        mat=X.copy()
        for c in col:
            mat[c]=[X[c].mean()]*len(X)
        cov=X.cov()
        inv=np.matrix(cov).I
        a=np.array(X.loc[i])
        b=np.array(mat.loc[i])
        mahala=np.array(spatial.distance.mahalanobis(a,b,inv))[0][0]
        return mahala
    
    def get_table(df,i=0):
        result= df.tables[i].as_html()
        return pd.read_html(result, header=0, index_col=0)[0]
    
    def rep(s):
        if isinstance(s,str) and '=' in s:
            return s
        else:
            return ''


# In[ ]:





# In[ ]:


'''
Class dataprep
--------------
.group_sep
    arguments
    ---------
    - data = name of dataframe (var)
    - groupvar = name of the grouping variable (str)
    returns
    -------
    list of groupwise dataframes (list of pd.DataFrame)
.nan
    arguments
    ---------
    - data = name of dataframe (var)
    - cols = list of column names that should be examined ([str,str,...])
    returns
    -------
    .analysis -> summary table for the analysis of nans (pd.DataFrame)
    .drop -> a dataframe where nans are removed (pd.DataFrame)
.onehot
    arguments
    ---------
    - drop = string that indicates which dummies should be dropped, can be set to None (str) [defaul='first'] 
    returns
    -------
    .fit(X,y)
    .transform(X,y,sparse) -> a dataframe where nans are removed (pd.DataFrame) 
        arguments
        --------- 
        - X = a dataframe (pd.DataFrame)
        - y = target variable (pd.Series) [optional, default = None]
        - sparse = whether output should be returned in sparse format (bool) [default=False]
.encoder
    arguments
    ---------
    - cols = nominal columns to encoder [default=None]
    - order = a dictionary of the format {colname:order of categories} to encode ordinal variables [default=None]
    if cols and order are both none the encoder treats all columns with strings as nominal variables
    returns
    -------
    .transform(data,sparse) -> an encoded dataframe
        arguments
        --------- 
        - data = a dataframe (pd.DataFrame) or a vector (pd.Series)
        - sparse = whether output should be returned in sparse format (bool) [default=False]
'''
class dataprep:
    def group_sep(data,groupvar):
        gb = data.groupby(groupvar)
        return [gb.get_group(x) for x in gb.groups]
    
    class nan:
        def __init__(self,data,cols=[]):

            def drop(data,cols=[]):
                if len(cols)==0:
                    cols=data.columns
                else:
                    pass
                df_new=data[cols]
                df_new=df_new.dropna()
                return df_new

            def analysis(data):
                cols=list(data.columns)
                if len(cols)<3:
                    miss=[data[c].isna().sum() for c in cols]+(3-len(cols))*['']
                    columns=cols+(3-len(cols))*['']
                else:
                    miss=[data[c].isna().sum() for c in cols]
                    columns=cols
                number=['','',sum([True for idx,row in data.iterrows() if any(row.isnull())])]
                number=number+(len(cols)-3)*['']
                results = {'':[*tuple(number)],'Column':[*tuple(columns)],'Missing Values':[*tuple(miss)]}
                idx=['Analysis Missing Values','','Number of Rows with NaNs']+(len(cols)-3)*['']
                table = pd.DataFrame(results, columns = ['','Column', 'Missing Values'],index=idx)
                return table

            self.analysis=analysis(data)
            self.drop=drop(data,cols=cols)
   
    class onehot:
        def __init__(self,cats=[],drop='first'):
            encoder=OneHotEncoder(drop=drop,sparse=True)
            self.preprocessor = ColumnTransformer(
            transformers=[("dummy", encoder, cats)],remainder='passthrough')
        def fit(self, X, y=None):
            return self.preprocessor.fit(X)
        def transform(self, X, y=None,sparse=False):
            X_c=X.copy()
            ar_encoded=self.preprocessor.transform(X_c)
            names=self.preprocessor.get_feature_names_out(list(X_c.columns))
            X_encoded = pd.DataFrame(ar_encoded, columns=names)
            remainders = [s for s in list(X_encoded.columns) if "remainder" in s]
            rename_dic=dict()
            for r in remainders:
                rename_dic[r]=r.split('__')[1]
            X_encoded.rename(columns=rename_dic,inplace=True)
            for c in list(X_encoded.columns):
                X_encoded[c]=pd.to_numeric(X_encoded[c], errors='ignore')

            if sparse is False:
                return X_encoded
            else:
                return ar_encoded
        def fit_transform(self, X, y=None,sparse=False):
            X_c=X.copy()
            ar_encoded=self.preprocessor.fit_transform(X_c)
            names=self.preprocessor.get_feature_names_out(list(X_c.columns))
            X_encoded = pd.DataFrame(ar_encoded, columns=names)
            remainders = [s for s in list(X_encoded.columns) if "remainder" in s]
            rename_dic=dict()
            for r in remainders:
                rename_dic[r]=r.split('__')[1]
            X_encoded.rename(columns=rename_dic,inplace=True)
            for c in list(X_encoded.columns):
                X_encoded[c]=pd.to_numeric(X_encoded[c], errors='ignore')

            if sparse is False:
                return X_encoded
            else:
                return ar_encoded
            
    class encoder:
        def __init__(self,cols=None,order=None):
            if order is not None:
                keys=list(order.keys())
                dics={}
                for k in keys:
                    dics[k]={order[k][i]:i for i in range(len(order[k]))}
                mappings=[{'col':k,'mapping':dics[k]} for k in keys]
            else:
                mappings=None
                keys=None
            self.encoder=ordenc(cols=cols,mapping=mappings,handle_unknown='return_nan',handle_missing='return_nan')
            self.cols=cols
            self.order=order
            self.keys=keys
        def fit_transform(self,data,sparse=False):
            cols=self.cols
            order=self.order
            enc=self.encoder
            keys=self.keys
            encoded=enc.fit_transform(data)
            
            if order!=None:
                if cols!=None:
                    cols=list(set(cols)-set(list(order.keys)))
                else:
                    pass
            else:
                if cols!=None:
                    pass
                else:
                    if isinstance(data,pd.Series):
                        cols=[data.name]
                    else:
                        cols=list(data.select_dtypes(object).columns)
            
            if cols!=None:
                for c in cols:   
                    encoded[c]=encoded[c]-1
            else:
                pass

            if sparse==True:
                if isinstance(data,pd.Series):
                    encoded=encoded[data.name]
                else:
                    encoded=encoded.to_numpy()
            else:
                pass

            return encoded


# In[ ]:





# In[ ]:


'''
Class describe
--------------
.data
.contingency
.corrmat
'''
class describe:
    #cr@FK
    '''
    Class data
    --------------
    arguments
    ---------
    - data = name of dataframe (var)
    - ordinal = list of ordinal variables ([str,str,...])
    - nominal = list of nominal variables ([str,str,...])
    returns
    -------
    .table(show) -> descriptive statistics (pd.DataFrame)
        arguments
        ---------
        - show = statistics for which variables ('numeric','ordinal','nominal') (str) [default='numeric']
    '''
    class data:
        def __init__(self,data,ordinal=[],nominal=[]):
            self.non_num=set(ordinal).union(set(nominal))
            self.num=list(set(list(data.columns))-self.non_num)
            self.data=data
            self.ordinal=ordinal
            self.nominal=nominal
        def table(self,show='numeric'):
            nominal=self.nominal
            ordinal=self.ordinal
            data=self.data
            num=self.num
            if show=='numeric':
                return data[num].describe()
            elif show=='ordinal':
                counts=[len(data[o]) for o in ordinal]
                categories=[len(data[o].unique()) for o in ordinal]
                mins=[min(data[o]) for o in ordinal]
                q1=[np.quantile(data[o], 0.25) for o in ordinal]
                medians=[data[o].median() for o in ordinal]
                q3=[np.quantile(data[o], 0.75) for o in ordinal]
                iqrs=[q3[i]-q1[i] for i in range(0,len(ordinal))]
                maxs=[max(data[o]) for o in ordinal]
                table=pd.DataFrame()
                for i in range(0,len(ordinal)):
                    table[ordinal[i]]=[counts[i],categories[i],iqrs[i],mins[i],q1[i],medians[i],q3[i],maxs[i]]
                table.index=['count','categories','iqr','min','25%','50%','75%','max']
                return table
            elif show=='nominal':
                counts=[len(data[n]) for n in nominal]
                mode=[data[n].mode()[0] for n in nominal]
                categories=[len(data[n].unique()) for n in nominal]
                vcounts=[pd.Series(data[n].value_counts()) for n in nominal]
                least_freq=[v.idxmin() for v in vcounts]
                most_freq=[v.idxmax() for v in vcounts]
                per_least=[vcounts[i][least_freq[i]]/counts[i] for i in range(0,len(nominal))]
                per_most=[vcounts[i][most_freq[i]]/counts[i] for i in range(0,len(nominal))]
                per_least=["{:.2%}".format(p) for p in per_least]
                per_most=["{:.2%}".format(p) for p in per_most]
                least=[str(least_freq[i])+''+'({})'.format(per_least[i]) for i in range(0,len(nominal))]
                most=[str(most_freq[i])+''+'({})'.format(per_most[i]) for i in range(0,len(nominal))]
                table=pd.DataFrame()
                for i in range(0,len(nominal)):
                    table[nominal[i]]=[counts[i],mode[i],categories[i],least[i],most[i]]
                table.index=['count','mode','categories','least freq','most freq']
            else:
                return print('Please state only numeric,ordinal or nominal!')
            return table
    '''
    Method contigency
    -----------------
    arguments
    ---------
    - x,y = vector of the variables (pd.Series)
    - show = what to show ('observed','expected','deviations') (str) [default='observed']
    - decimals = decimal places for percentage deviations (int)
    returns
    -------
    observed or expected frequencies or their deviation
    '''
    def contingency(x,y,show='observed',decimals=0):
        observed=pd.crosstab(x,y)
        expected=pd.DataFrame(stats.chi2_contingency(observed)[3],columns=observed.columns,index=observed.index)
        deviations=(observed-expected)/(expected+1/(10**6))
        deviations=round(deviations * 100,decimals).astype(str) + '%'
        if show=='observed':
            return observed
        elif show=='expected':
            return expected
        else:
            return deviations
    '''
    Class corrmat
    -----------------
    arguments
    ---------
    - data = name of dataframe (var)
    - nominal,ordinal = list of names of nominal,ordinal variables ([str,str,...])
    - ordvsord = correlation coefficient for ordinal vs ordinal/numerical ('spearman','kendall','gk_gamma') (str) [default = 'spearman']
    - nomvsnom = correlation coefficient for nominal vs nominal/ordinal ('cramer') (str) 
    - numvsnom = correlation coeffient for numerical vs nominal ('eta','pbc') (str) [default = 'eta']
    - stars = do you want to flag significant correlations with stars (bool) [default = False]
    - padjust = method to correct for multiple testing ('none','bonf','sidak','holm','fdr_bh','fdr_by') (str) [default='bonf']
    - utri = do you want to show the upper triangle of the matrix (bool) [default = True]
    - ltri = do you want to show the upper triangle of the matrix (bool) [default = True]
    - fill = how to fill the empty spaces if upper/lower triangle is masked (str) [default = '']
    - decimals = how many decimal places to show in case a triangle is masked or stars is True (int) [default = 4]
    - percent = display the correlations as percentages (bool) [default = False]
    - force_biserial = always use rank-biserial resp. point-biserial coefficent when ordinal/numerical vs binary nominal (bool) [default = True]
    returns
    -------
    .table -> correlation matrix (pd.DataFrame)
    .def heatmap(self,cmap='coolwarm',roty=0,rotx=0,lsize=15,tsize=20,annot=True,fsize=70,fig=[12,8],down=0): -> matrix as heatmap
        arguments
        ---------
        - cmap = palette for heatmap (str) [default = 'coolwarm']
        - rotx, roty = rotate x or y labels (int) [default: rotx, roty = 0, 0]
        - lsize = labelsize (int) [default = 15]
        - tsize = ticksize (int) [default = 20]
        - annot = show numbers in cells (bool) [default = True]
        - fsize = fontsize in cells (int) [default = 70]
        - fig = size of figure ([int,int]) [default = [12,8]] 
        - down = shift the caption down only relevant when stars = True (dec) [default = 0]
    '''
    class corrmat:
        def __init__(self,data,ordinal=[],nominal=[],ordvsord='spearman',ordvsnom=None,nomvsnom='cramer',numvsnom='eta',
                     stars=False,padjust='bonf',alternative='two-sided',utri=True,ltri=True,fill='',
                     decimals=4,percent=False,show_nominal=False,force_biserial=True):
            #self arguments
            self.utri=utri
            self.ltri=ltri
            self.dec=decimals
            self.per=percent
            self.nom=show_nominal
            self.stars=stars
            #check for numerics
            ##ordinal
            if len(data[ordinal].select_dtypes(include=np.number).columns.tolist())!=len(data[ordinal].columns):
                return print('Some of your ordinal variable have not been converted to numbers yet!')
            else:
                pass
            #preliminaries
            cm=data.corr(numeric_only=True)
            cols=data.columns
            numerical=list(set(list(cols))-set(ordinal)-set(nominal))
            if ordvsnom is None:
                ordvsnom=nomvsnom
            else:
                pass
            cdic={'cramer':tests.correlation.cramer,'gk_gamma':tests.correlation.gk_gamma,'eta':tests.correlation.eta,
                  'rbc':tests.correlation.rbc,'pbc':tests.correlation.pbc,'spearman':stats.spearmanr,'kendall':stats.kendalltau}
            # p-value matrix and coef matrix
            pm=cm.copy()
            coefm=cm.copy()
            #adjust cmatrix and fill pmatrix
            for j in numerical:
                for i in numerical:
                    pm.loc[i,j]=pm.loc[j,i]=stats.pearsonr(data[i], data[j])[1]
                    coefm.loc[i,j]=coefm.loc[j,i]='pearson'
            for j in ordinal:
                for i in ordinal+numerical:
                    pm.loc[i,j]=pm.loc[j,i] = cdic[ordvsord](data[i], data[j])[1]
                    cm.loc[i, j] = cm.loc[j, i] = cdic[ordvsord](data[i], data[j])[0]
                    coefm.loc[i,j]=coefm.loc[j,i]=ordvsord
            for j in nominal:
                if force_biserial == True:
                    if len(data[j].unique())==2:
                        ordvsnom_f='rbc'
                        numvsnom_f='pbc'
                    else:
                        ordvsnom_f=ordvsnom
                        numvsnom_f=numvsnom
                #else:
                    #pass
                for i in nominal:
                    pm.loc[i,j]=pm.loc[j,i]=cdic[nomvsnom](data[i], data[j])[1]
                    cm.loc[i, j] = cm.loc[j, i] =cdic[nomvsnom](data[i], data[j])[0]
                    coefm.loc[i,j]=coefm.loc[j,i]=nomvsnom
                for i in ordinal:
                    pm.loc[i,j]=pm.loc[j,i]=cdic[ordvsnom_f](data[i], data[j])[1]
                    cm.loc[i, j] = cm.loc[j, i] =cdic[ordvsnom_f](data[i], data[j])[0]
                    coefm.loc[i,j]=coefm.loc[j,i]=ordvsnom_f
                for i in numerical:
                    pm.loc[i,j]=pm.loc[j,i]=cdic[numvsnom_f](data[i], data[j])[1]
                    cm.loc[i, j] = cm.loc[j,i] =cdic[numvsnom_f](data[i], data[j])[0]
                    coefm.loc[i,j]=coefm.loc[j,i]=numvsnom_f
            # adjust names
            def ordnom(x):
                sep = " "
                if x in ordinal:
                    return sep.join([x,'(ord)'])
                elif x in nominal:
                    return sep.join([x,'(nom)'])
                else:
                    return x
            data=data.copy()
            cm_adj=cm.copy()
            for c in list(data.columns):
                data=data.rename(columns={c: ordnom(c)})
            nindex=[ordnom(r) for r in cm.index]
            cm_cols=list(cm.columns)
            cm_adj.index=nindex
            pm.index=nindex
            coefm.index=nindex
            for c in cm_cols:
                cm_adj=cm_adj.rename(columns={c: ordnom(c)})
                pm=pm.rename(columns={c: ordnom(c)})
                coefm=coefm.rename(columns={c: ordnom(c)})
            ordinal=[ordnom(o) for o in ordinal]
            nominal=[ordnom(n) for n in nominal]
            self.nomnames=nominal
            #export adjusted matrix
            self.raw=cm_adj
            #percentages
            if percent==True:
                cm_adj=round(cm_adj * 100,decimals-2).astype(str) + '%'
            else:
                pass
            #remove upper or lower triangle
            if utri==False and ltri==False:
                return print('If utri AND ltri are false, you get an empty table?')
            elif utri==False:
                cm_adj=cm_adj.round(decimals)
                cm_adj=cm_adj.mask(np.triu(np.ones(cm.shape)).astype(bool))
                cm_adj=cm_adj.fillna(fill)
            elif ltri==False:
                cm_adj=cm_adj.round(decimals)
                cm_adj=cm_adj.mask(np.tril(np.ones(cm.shape)).astype(bool))
                cm_adj=cm_adj.fillna(fill)
            else:
                pass
            if stars==True:
                cm_adj=cm_adj.round(decimals)
                # relevant p-values
                p_rel=[]
                for i in list(data.columns):
                    for j in list(data.columns):
                        if i>j:
                            p_rel.append(pm.loc[i,j])
                # adjusted p-values
                p_adj=pg.multicomp(p_rel,method=padjust)[1]
                pdic={}
                for i in range(len(p_rel)):
                    pdic[p_rel[i]]=p_adj[i]
                #adjust p-value matrix
                pm_adjust=pm.copy()
                for i in list(cm_adj.columns):
                    for j in list(cm_adj.columns):
                        if i!=j:
                            pm_adjust.loc[i,j]=pdic[pm.loc[i,j]]
                            if tools.stars(pm_adjust.loc[i,j])!='not sig':
                                cm_adj.loc[i,j]=str(cm_adj.loc[i,j])+tools.stars(pm_adjust.loc[i,j])
                            else:
                                pass
                        else:
                            pass
            else:
                pass

            if show_nominal==False:
                cm_adj=cm_adj.drop(nominal)
                cm_adj=cm_adj.drop(nominal,axis=1)
            else:
                pass
            #returns
            self.table = cm_adj
            coefm=coefm.mask(np.triu(np.ones(cm.shape)).astype(bool))
            coefm=coefm.fillna(fill)
            self.coef = coefm

            
        def heatmap(self,cmap='coolwarm',roty=0,rotx=0,lsize=15,tsize=20,annot=True,hm_font='auto',fig=[12,8],nsize=70,down=0):
            #self objects
            cm=self.raw
            utri=self.utri
            ltri=self.ltri
            decimals=self.dec
            percent=self.per
            show_nominal=self.nom
            nominal=self.nomnames
            stars=self.stars
            #drop nominals
            if show_nominal==False:
                cm=cm.drop(nominal)
                cm=cm.drop(nominal,axis=1)
            else:
                pass
            #columns
            coli=list(cm.columns)
            #mask
            if utri==False:
                mask = np.triu(np.ones_like(cm, dtype=bool))
            elif ltri==False:
                mask = np.tril(np.ones_like(cm, dtype=bool))
            else:
                mask=None
            #font in cells
            if hm_font=='auto':
                fsize=nsize/len(coli)
            else:
                fsize=hm_font
            #annot format and stars
            if stars==False:
                if percent is False:
                    aformat='.{}f'.format(decimals)
                else:
                    aformat='.{}%'.format(decimals-2)
            else:
                table=self.table.astype(str)
                annot=np.array(table)
                aformat=''

            fig, ax = plt.subplots(figsize=(fig[0],fig[1]))  
            hm = sns.heatmap(cm,annot=annot, cmap=cmap,annot_kws={'fontsize':fsize},fmt=aformat,
                             vmin=-1., vmax=1.,ax=ax, xticklabels=coli, yticklabels=coli,mask=mask)
            bottom, top = hm.get_ylim()
            ax.tick_params(labelsize=lsize)
            plt.yticks(rotation=roty)
            plt.xticks(rotation=rotx)
            if stars==True:
                txt="* p<0.05, ** p<0.01, *** p<0.001, **** p<0.0001"
                plt.figtext(0.5, 0.01-down, txt, wrap=True, horizontalalignment='center', fontsize=12)
            else:
                pass


# In[ ]:





# In[ ]:


'''
Class plots
--------------
.dist
.qq
.scatter
.scatter3d
.outlier
'''     
class plots:
    '''
    Method dist
    -----------
    arguments
    ---------
    - data = either name of dataframe (var) or vector (pd.Series)
    - var, groupvar = name of variable and grouping variable (str) -> only relevant when data is name of a dataframe
    - fig = figure size ([int,int]) [default=[12,8]]
    - ticksize = size of ticks (int) [default=15]
    - labelsize = size of labels (int) [default=25]
    - legsize = size of legend (int) [default=15]
    - dark = dark background (bool) [default=False]
    - linewidth = line width (int) [default=3]
    - lineclr = line color (str) [default='blue']
    - xlabel = custom x-axis label (str) [default=None]
    - sharediagram = whether plots should be done in the same diagram (bool) [default = True]
    
    returns
    -------
    Distribution plot
    '''
    def dist(data,var=None,groupvar=None,fig=[12,8],ticksize=15,labelsize=25,legsize=15,dark=False,
             linewidth=3,lineclr='blue',xlabel=None,sharediagram=True):
        #xlable
        def xl(xlabel=None):
            if xlabel is not None:
                return plt.xlabel(xlab,fontsize=labelsize), plt.ylabel('Density',fontsize=labelsize)
            else:
                return ax.yaxis.label.set_size(labelsize), ax.xaxis.label.set_size(labelsize)
        #darkgrid
        if dark is True:
            sns.set_style("darkgrid")
        else:
            sns.set_style("whitegrid")
        #plot
        if isinstance(data, pd.Series):
            plt.figure(figsize=(fig[0],fig[1]))
            ax=sns.distplot(data, hist=False, label=data.name,kde_kws=dict(linewidth=linewidth,color=lineclr))
            plt.tick_params(labelsize=ticksize)
            xl(xlabel=xlabel)
            plt.legend(fontsize=legsize)
        elif isinstance(data, pd.DataFrame):
            if var is None or groupvar is None:
                return print('Please specify: var= and groupvar=!')
            else:
                groups=dataprep.group_sep(data,groupvar)
                if sharediagram==True:
                    fig = plt.figure(figsize=(fig[0],fig[1]))
                    ax = fig.add_subplot(111)
                    for g in groups:
                        sns.distplot(g[var], hist=False, label="Group: {}".format(list(g[groupvar].unique())[0]),kde_kws=dict(linewidth=linewidth))
                        plt.legend(fontsize=legsize)
                        plt.tick_params(labelsize=ticksize)
                        xl(xlabel=xlabel)
                else:
                    for g in groups:
                        plt.figure(figsize=(fig[0],fig[1]))
                        ax=sns.distplot(g[var], hist=False, label="Group: {}".format(list(g[groupvar].unique())[0]),kde_kws=dict(linewidth=linewidth,color=lineclr))
                        plt.legend(fontsize=legsize)
                        plt.tick_params(labelsize=ticksize)
                        xl(xlabel=xlabel)
        else:
            return print('Please provide either a series object or a dataframe and specify var+groupvar!')
    '''
    Method qq
    -----------
    arguments
    ---------
    - data = either name of dataframe (var) or vector (pd.Series)
    - var, groupvar = name of variable and grouping variable (str) -> only relevant when data is name of a dataframe
    - fig = figure size ([int,int]) [default=[12,8]]
    - ticksize = size of ticks (int) [default=15]
    - labelsize = size of labels (int) [default = 18]
    - legsize = size of legend [default = 16]
    - dark = dark background? (bool) [default = False]
    - rotx, roty = rotate x and y (int between 0 and 360) [default: rotx=0, roty=90]
    - dotsize = size of dots (int) [default = 80]
    - confidence = whether to plot a confidence interval (dec between 0 and 1) [default = False]
    
    returns
    -------
    QQ Plot
    '''
    def qq(data,var=None,groupvar=None,fig=[12,8],ticksize=14,labelsize=18,legsize=16,dark=False,rotx=0,
           roty=90,dotsize=80,confidence=False):
            #darkgrid
            if dark is True:
                sns.set_style("darkgrid")
            else:
                sns.set_style("whitegrid")
            #plot
            if isinstance(data, pd.Series):
                plt.figure(figsize=(fig[0],fig[1]))
                pg.qqplot(data, dist='norm',confidence=confidence,s=dotsize)
                plt.xlabel('Theoretical quantiles', fontsize = labelsize,rotation=rotx)
                plt.ylabel('Ordered quantiles', fontsize = labelsize,rotation=roty)
                plt.xticks(fontsize=ticksize)
                plt.yticks(fontsize=ticksize)
            elif isinstance(data, pd.DataFrame):
                if var is None or groupvar is None:
                    return print('Please specify: var= and groupvar= when data = dataframe!')
                else:
                    groups=dataprep.group_sep(data,groupvar)
                    for g in groups:
                        plt.figure(figsize=(fig[0],fig[1]))
                        pg.qqplot(g[var], dist='norm',confidence=confidence,s=dotsize,label='{}'.format(var)+" (Group: {})".format(list(g[groupvar].unique())[0]))
                        plt.xlabel('Theoretical quantiles', fontsize = labelsize,rotation=rotx)
                        plt.ylabel('Ordered quantiles', fontsize = labelsize,rotation=roty)
                        plt.xticks(fontsize=ticksize)
                        plt.yticks(fontsize=ticksize)
                        plt.legend(fontsize=legsize)
            else:
                return print('Please provide either a series object or a dataframe and specify var+groupvar!')
    '''
    Method Scatter
    --------------
    arguments
    ---------
    - data = name of dataframe (var)
    - x,y = names of variables (str)
    - fig = figure size ([int,int]) [default = [8,6]]
    - ticksize = size of ticks (int) [default = 14]
    - labelsize = size of labels (int) [default = 16]
    - dark = dark background? (bool) [default = False]
    - dotsize = size of dots (int) [default = 100]
    - dotclr = color of dots (str) [default = ['blue']]
    - hue = name of third variable whose values are used to color the dots (str) [default = None]
    - hueclr = palette for colors of hue variable (str) [default = 'tab10']
    - namexy = names of variables (list) [default = None]
    - rotx, roty = rotate x and y (int between 0 and 360) [default: rotx=0, roty=90]
    - ordinal = enable bubble plot (bool) [default = False]
    - bubsize = size of dots in bubble plot ((int,int)) [default = (20,2000)]
    - regression = plot a regression ('linear','logistic','poly') (str) [default = False]
    - linewidth = width of regression line (int) [default = 2]
    - lineclr = color of regression line (str) [default = 'red']
    - poly_deg = polynomial degrees only relevant if regression = 'poly' (int) [default = 3]
    - legend = enable legend (bool) [default = False]
    - legendfont = size of legend (int) [default = 12]
    - legendcol = number of legend cols (int) [default = 1]
    - legendspaceh = horizontal space between legend items (int) [default = 3]
    - legendspacev = vertical space between legend items (int) [default = 2]
    - intext = whether to plot the regression description inside plot (bool) [default = False]
    - pos = position of intext [x,y] ([int,int]) [default = [0,0]]
    - txtclr = color of intext (str) [default = 'red']
    - txtsize = size of intext [default = 12]
    returns
    -------
    Scatter Plot
    '''
    def scatter(x,y,data=None,fig=[8,6],ticksize=14,labelsize=16,dark=False,dotsize=100,dotclr=['blue'],
                hue=None,hueclr='tab10',hue_norm=None, namexy=[],rotx=0,roty=90, ordinal=False,bubsize=(20, 2000),
                regression=None,linewidth=2,lineclr='red',poly_deg=3,
                legend=False,legendfont=12, legendcol=1,legendspaceh=3,
                legendspacev=2,intext=False,pos=[0,0],txtclr='red',txtsize=12):
        #preliminaries
        if data is None:
            if isinstance(x,pd.DataFrame):
                x_poly=x[list(x.columns)[0]]
                v1,v2=x,y
            else:
                v1,v2=pd.DataFrame(x),y
                x_poly=x
                if len(namexy)==0:
                    namexy=[x.name,y.name]
                else:
                    pass
        else:
            v1,v2=data[[x]], data[y]
            x_poly=data[x]
            if len(namexy)==0:
                namexy=[x,y]
            else:
                pass
        #define space
        l,u = math.floor(min(v1[list(v1.columns)[0]])),math.ceil(max(v1[list(v1.columns)[0]]))
        Xs = [i for i in range(l,u)]
        #define and fit models
        if regression in ['linear','logistic']:
            def_m={'linear':LinearRegression(),'logistic':LogisticRegression()}
            model=def_m[regression]
            model.fit(v1,v2)
            if regression=='linear':
                param=round(model.coef_[0],2)
                inter=round(model.intercept_,2)
                eq='{} = {} + {}{}'.format(namexy[1],inter,param,namexy[0])
            else:
                param=round(model.coef_[0][0],2)
                inter=round(model.intercept_[0],2)         
                eq='p({}=1) = exp(z)/(1+exp(z)) where z={}+{}{}'.format(namexy[1],inter,param,namexy[0])
        elif regression=='poly':
            model=np.poly1d(np.polyfit(x_poly, v2, poly_deg))
            eq='Polynomial regression with degree {}'.format(poly_deg)
        else:
            pass
        #plot
        if dark is True:
            sns.set_style("darkgrid")
        else:
            sns.set_style("whitegrid")    
        a,b =fig[0],fig[1]
        fig=plt.figure()
        fig.set_size_inches(a, b)
        if ordinal is False:
            g=sns.scatterplot(data=data, x=x, y=y, hue=hue,legend=legend,s=dotsize,c=dotclr,palette=hueclr,hue_norm=hue_norm)
        if ordinal is True:
            pairs=pd.Series([(x_poly[i],v2[i]) for i in list(v2.index)])
            aux=pd.DataFrame(pairs.value_counts())
            aux['v1']=[aux.index[i][0] for i in range(0,len(aux))]
            aux['v2']=[aux.index[i][1] for i in range(0,len(aux))]
            if hue!=None:
                hue='count'
            else:
                pass
            g=sns.scatterplot(data=aux, x='v1', y='v2',size='count', hue=hue,legend=legend,sizes=bubsize,c=dotclr,palette=hueclr,
                             hue_norm=hue_norm)    
        plt.xlabel(namexy[0], fontsize = labelsize,rotation=rotx) # x-axis label with fontsize 15
        plt.ylabel(namexy[1], fontsize = labelsize,rotation=roty) # y-axis label with fontsize 15
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        if legend is not False:
            g.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=legendcol,fontsize=legendfont,handletextpad=legendspaceh,labelspacing = legendspacev)
        else:
            pass
        if regression=='linear':
            Ys = [model.predict([[value]])[0] for value in range(l,u)]
            plt.plot(Xs, Ys, color=lineclr,lw=linewidth)
        elif regression=='logistic':
            Ys = [model.predict_proba([[value]])[0][1] for value in range(l,u)]
            plt.plot(Xs, Ys, color=lineclr,lw=linewidth)
        elif regression=='poly':
            Ys= [model(value) for value in Xs]
            plt.plot(Xs, Ys, color=lineclr,lw=linewidth)
        else:
            pass
        if regression!=None:
            if intext==True:
                xbas=x_poly.mean()
                plt.text(x=xbas+pos[0],y=pos[1],s=eq,color=txtclr,fontsize=txtsize)
            else:
                print(eq)
        else:
            pass
        

    '''
    Method Scatter3d
    ----------------
    arguments
    ---------
    - X = feature matrix with 2 variables (pd.DataFrame)
    - y = dependent variable (pd.Series)
    - fig = figure size ([int,int]) [default = [8,6]]
    - ticksize = size of ticks (int) [default = 14]
    - labelsize = size of labels (int) [default = 16]
    - dotsize = size of dots (int) [default = 60]
    - dotclr = color of dots (str) [default = 'blue']
    - linreg = whether to plot regression plane (bool) [default = True]
    - regpal = palette for regression plane, can be None (obj) [default = plt.cm.RdBu_r]
    - regclr = color of regression plan used only if regpal=None (str) [default = 'red']
    - intext = whether to plot the regression description inside plot (bool) [default = False]
    - pos = position of intext [x,y] ([int,int]) [default = [0,0]]
    - txtclr = color of intext (str) [default = 'red']
    - txtsize = size of intext [default = 12]
    returns
    -------
    3D Scatter Plot 
    '''
    def scatter3d(X,y,fig=[12,8],ticksize=12,labelsize=14,dotsize=60,dotclr='blue',
                  linreg=True,regpal=plt.cm.RdBu_r,regclr='red',
                  intext=False,pos=[0,0,60],txtclr='red',txtsize=10):
        #scatter
        fig = plt.figure(figsize=(fig[0], fig[1])) 
        #ax = Axes3D(fig, azim=-115, elev=15)
        ax=fig.add_subplot(projection='3d')
        ax.scatter(X[X.columns[0]],X[X.columns[1]],y, color='white', alpha=1.0, facecolor=dotclr,s=dotsize) 
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.tick_params(axis='both', which='minor', labelsize=ticksize)
        ax.set_xlabel(X.columns[0],fontsize = labelsize) 
        ax.set_ylabel(X.columns[1],fontsize = labelsize) 
        ax.set_zlabel(y.name,fontsize = labelsize) 
        if linreg==True:
            xx1, xx2 = np.meshgrid(np.linspace(X[X.columns[0]].min(), X[X.columns[0]].max(), 100),  
                           np.linspace(X[X.columns[1]].min(), X[X.columns[1]].max(), 100)) 
            X1 = sm.add_constant(X) 
            model = sm.OLS(y,X1.astype(float)).fit()
            params=model.params
            Z = params[0] + params[1] * xx1 + model.params[2] * xx2
            surf = ax.plot_surface(xx1, xx2, Z, cmap=regpal, alpha=0.6, linewidth=0,color=regclr) 
            eq="{} = {:.2f} + {:.2f}{}+ {:.2f}{}".format(y.name,params[0], params[1],X.columns[0], params[2],X.columns[1])
            if intext==False:
                print(eq)
            else:
                xbas=X[X.columns[0]].min()
                ybas=X[X.columns[1]].min()
                zbas=max(y)
                ax.text(x=xbas+pos[0],y=ybas+pos[1],z=zbas+pos[2],s=eq,color=txtclr,fontsize=txtsize)

        else:
            pass
    
    '''
    Method outlier
    ----------------
    arguments
    ---------
    - x = either feature matrix with variables (pd.DataFrame) or variable (pd.Series)
    - y = dependent variable (pd.Series) or None if univariate
    - k = outlier detection factor either 'auto' (str) or a dictionary containing methods and factors (dic) [default='auto']
    - method = outlier detection method ('iqr','zscore','mad','Cook','Mahalanobis') (str) [default = iqr]
    - limround = decimal places to round tresholds (int) [default = 0]
    - dtype = data type ('univariate','bivariate','multivariate') (str) [default = 'univariate']
    returns
    -------
    uni- and multivariate return a observation vs distance/values plot with tresholds
    bivariate returns a scatterplot where outliers are flagged red
    '''
 
    def outlier(x,y=None,k='auto',method='iqr',limround=0,dtype='univariate'):
        dick={}
        if y is not None:
            add=1
        else:
            add=0
        if k=='auto':
            if dtype!='univariate':
                if isinstance(x,pd.Series):
                    X_new=pd.DataFrame(x)
                    x=pd.DataFrame(x)
                else:
                    X_new=x.copy()
                X_new[y.name]=y
                dick['Cook']='auto'
                dick['Mahalanobis']='auto'
            else:
                pass
            for m in ['zscore','iqr','mad']:
                dick[m]='auto'
        else:
            for m in ['zscore','iqr','mad','Cook','Mahalanobis']:
                dick[m]=k[m]
        if dtype=='univariate':
            #preliminaries
            l,u = tools.extreme(x=x,method=method,k=dick[method])
            l,u = round(l,limround), round(u,limround)
            #plot
            plt.figure(figsize = (10, 5))
            for i in list(x.index):
                plt.vlines(x=i, ymin=min(0,x[i]), ymax=max(0,x[i]), color='blue')
            plt.axhline(y=u, color='red', linestyle='--',lw=1, label="Upper limit = {}".format(u))
            plt.axhline(y=l, color='green', linestyle='--',lw=1, label="Lower limit = {}".format(l))
            plt.title('Outlier detection {} (k = {})'.format(method,dick[method]))
            plt.xlabel('observation')
            plt.ylabel('{}'.format(x.name))
            plt.legend(bbox_to_anchor=(1.25, 1))
        elif dtype=='multivariate':
            if method=='Cook':
                if dick['Cook']=='auto':
                    tresh=4/len(x)
                else:
                    tresh=dick['Cook']
                visualizer = CooksDistance(linefmt='blue')
                Cdist=visualizer.fit(x, y).distance_
                plt.close()
                plt.figure(figsize = (10, 5))
                for i in list(x.index):
                    plt.vlines(x=i, ymin=min(0,Cdist[i]), ymax=max(0,Cdist[i]), color='blue')
                plt.axhline(y=tresh, color='red', linestyle='--',lw=1, label="Treshold = {}".format(tresh))
                plt.xlabel('observation')
                plt.ylabel('Cooks Distance')
                plt.legend(bbox_to_anchor=(1.25, 1))
                plt.title('Outlier detection {} (k = {})'.format(method,tresh))
            elif method=='Mahalanobis':
                #plot
                plt.figure(figsize = (10, 5))
                dist={}
                for i in list(x.index):
                    dist[i]=tools.mahalanobis(X_new,i)
                if dick['Mahalanobis']!='auto':
                    dick['Mahalanobis']=round(dick['Mahalanobis'],limround+2)
                    tresh=dick['Mahalanobis']
                else:
                    tresh=round(stats.chi2.ppf(0.975, df=len(x.columns)+add),limround+2)
                for i in list(x.index):
                    plt.vlines(x=i, ymin=0, ymax=dist[i], color='blue')
                plt.xlabel('observation')
                plt.ylabel('Mahalanobis Distance')
                plt.axhline(y=tresh, color='red', linestyle='--',lw=1, label="Treshold = {}".format(tresh))
                plt.legend(bbox_to_anchor=(1.25, 1))
                plt.title('Outlier detection {} (k = {})'.format(method,tresh))
            else:
                pass
        elif dtype=='bivariate':
            dicko={}
            for m in ['Cook', 'Mahalanobis']:
                dicko[m]=outlier.multivariate(x=x,y=y,k=dick[m]).show(method=m)
            colors=pd.DataFrame()
            colors['out']=['blue']*len(x)
            colors.index=x.index
            for i in dicko[method]:
                colors.loc[i,'out']='red'
            plt.scatter(x=x,y=y,c=colors['out'])
            plt.xlabel(list(x.columns)[0])
            plt.ylabel(y.name)
            plt.title('{} Distance Outlier Detection'.format(method ))
        else:
            pass


# In[ ]:





# In[ ]:


'''
Class tests
------------
.t
.nonparametric
.independence
.correlation
.association
.equal_var
'''
class tests:
    '''
    Class t
    ------------
    .one_sample
        arguments
        ---------
        - data = name of dataframe (var)
        - var = name of the variable in the dataframe (str)
        - nullmean = mean under null hypothesis (int)
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        returns
        -------
        table of one-sample t-test (pd.DataFrame)
    .two_sample
        arguments
        ---------
        - data = name of dataframe (var)
        - var = name of the variable in the dataframe (str)
        - groupvar = name of the grouping variable (str)
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        returns
        -------
        table of two-sample t-test (pd.DataFrame)
    .paired
        arguments
        ---------
        - data = name of dataframe (var) [optional]
        - var1, var2 = names of the variables in the dataframe (str) or vectors (pd.Series) if data is None
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        returns
        -------
        table of paired t-test (pd.DataFrame)

    Remarks:
    -------
    - Missing values are automatically removed from the data.
    '''
    class t:
        def one_sample(data,var=None,nullmean=None,alternative='two-sided'):
                if var is None:
                    if isinstance(data, pd.Series):
                        v=data
                        namevar=v.name
                    elif isinstance(data, pd.DataFrame):
                        return print('Please specifiy the variable (var=) in your dataframe!')
                    else:
                        return print('data must be a series or a dataframe object!')
                else:
                    v=data[var]
                    namevar=var
                if nullmean is None:
                    return print('Please specifiy the mean under the null hypothesis (nullmean=)!')
                else:
                    pass

                #statistics
                mean=v.mean()

                #table
                results=pg.ttest(v, nullmean,alternative=alternative)
                results.rename(columns={"T": "t"},inplace=True)
                results.insert(0, "null mean", [nullmean])
                results.insert(0, "mean", [mean])
                results.insert(0, "var", [namevar])
                results.index=['One-Sample t-Test']

                return results

        def two_sample(data,var=None,groupvar=None,alternative='two-sided'):
            #preliminaries
            l=len(data[groupvar].unique())
            if l!=2:
                return print('There are more or less than 2 samples!')
            else:
                groups=dataprep.group_sep(data,groupvar)
            #statistics
            means=[g[var].mean() for g in groups]
            gnames=[g[groupvar].unique()[0] for g in groups]
            #Table
            results_t=pg.ttest(groups[0][var], groups[1][var],alternative=alternative,correction=False)
            results_welch=pg.ttest(groups[0][var], groups[1][var],alternative=alternative,correction=True)
            results_t.index=[0]
            results_welch.index=[1]
            results = pd.concat([results_t,results_welch])
            results.rename(columns={"T": "t"},inplace=True)
            results['alternative']=[alternative,'']
            results.insert(0, "variances", ['equal','unequal'])
            results.insert(0, "mean", [*tuple(means)])
            results.insert(0, "group", [*tuple(gnames)])
            results.insert(0, "var", [var,''])
            results.index=['Two-Sample', 't-Test']

            return results

        def paired(var1,var2,data=None,alternative='two-sided'):
            #preliminaries
            if data is None:
                if isinstance(var1, pd.Series):
                    v1=var1
                    if isinstance(var2, pd.Series):
                        v2=var2
                        if len(var1)!=len(var2):
                            return print('You selected a paired test but the variables have different length')
                        else:
                            name1=var1.name
                            name2=var2.name
                    else:
                        return print('You did not specify a dataframe and var2 is not a series object')
                else:
                    return print('You did not specify a dataframe and var1 is not a series object')
            else:
                v1=data[var1]
                v2=data[var2]
                name1=var1
                name2=var2
            #statistics
            mean_diff=v1.mean()-v2.mean()
            corr=stats.pearsonr(v1, v2)[0]
            name_group_diff=name1+'-'+name2
            #table
            results=pg.ttest(v1, v2,paired=True,alternative=alternative)
            results.rename(columns={"T": "t"},inplace=True)
            results.insert(0, "correlation", [corr])
            results.insert(0, "mean diff", [mean_diff])
            results.insert(0, "var", [name_group_diff])
            results.index=['Paired Sample t-Test']

            return results

    '''
    Class nonparametric
    -------------------
    .sign
        arguments
        ---------
        - data = name of dataframe (var or pd.Series)
        - var = name of the variable in the dataframe (str) or None if data = pd.Series
        - nullmedian = median under null hypothesis (int) [if nullmedian=None -> nullmedian=median of variable]
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        returns
        -------
        table of one-sample sign test (pd.DataFrame)
    .mwu
        arguments
        ---------
        - data = name of dataframe (var)
        - var = name of the variable in the dataframe (str)
        - groupvar = name of the grouping variable (str)
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        returns
        -------
        table of Mann-Whitney U test (pd.DataFrame)
    .wilcoxon
        arguments
        ---------
        - data = name of dataframe (var) [optional]
        - var1, var2 = names of the variables in the dataframe (str) or vectors (pd.Series) if data is None
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        returns
        -------
        table of Wilcoxon signed-rank test (pd.DataFrame)
    '''
    class nonparametric:
        def sign(data,var=None,nullmedian=None,alternative='two-sided',p=0.5):
            #preliminaries
            if var is None:
                if isinstance(data, pd.Series):
                    v=data
                else:
                    return print('"data" must be a series object if "var" is not specified!')
            else:
                v=data[var]

            if nullmedian is None:
                nullmedian=v.median()
            else:
                pass
            name=v.name
            n=len(v)
            #statistics
            plus=len([x for x in v if x > nullmedian])
            minus=len([x for x in v if x < nullmedian])
            #minpm=min(plus,minus)
            p=stats.binom_test(plus, n=plus+minus, p=p, alternative=alternative)  
            M=(plus-minus)/2
            #table
            results=pd.DataFrame()
            results['var']=[name]
            results['median']=[v.median()]
            results['null median']=[nullmedian]
            results['n(-)']=[minus]
            results['n(+)']=[plus]
            results['n']=[n]
            results['M']=[M]
            results['alternative']=[alternative]
            results['p-val']=[p]
            results.index=['Sign Test']

            return results

        def mwu(data,var,groupvar,alternative='two-sided'):
            #preliminaries
            l=len(data[groupvar].unique())
            if l!=2:
                return print('There are more or less than 2 samples!')
            else:
                groups=dataprep.group_sep(data,groupvar)
                namegroups=[list(groups[0][groupvar].unique())[0],list(groups[1][groupvar].unique())[0]]
            #statistics
            ns=[len(g) for g in groups]
            medians=[g[var].median() for g in groups]
            U1=stats.mannwhitneyu(groups[0][var],groups[1][var]).statistic
            U2=ns[0]*ns[1]-U1
            #table
            results=pg.mwu(groups[0][var],groups[1][var],alternative=alternative)
            results.insert(0, "median", ['{}'.format(medians[0])+'/'+'{}'.format(medians[1])])
            results.insert(0, "n", ['{}'.format(ns[0])+'/'+'{}'.format(ns[1])])
            results.insert(0, "groups", ['{}'.format(namegroups[0])+'/'+'{}'.format(namegroups[1])])
            results.insert(0, "var", [var])
            results.index=['Mann–Whitney U test']

            return results

        def wilcoxon(var1,var2,data=None,alternative='two-sided'):
            #preliminaries
            if data is None:
                if isinstance(var1, pd.Series):
                    v1=var1
                    if isinstance(var2, pd.Series):
                        v2=var2
                        if len(var1)!=len(var2):
                            return print('You selected a paired test but the variables have different length')
                        else:
                            name1=var1.name
                            name2=var2.name
                    else:
                        return print('You did not specify a dataframe and var2 is not a series object')
                else:
                    return print('You did not specify a dataframe and var1 is not a series object')
            else:
                v1=data[var1]
                v2=data[var2]
                name1=var1
                name2=var2
            #statistics
            ns=[len(v1),len(v2)]
            medians=[v1.median(),v2.median()]
            name_group_diff=name1+'-'+name2
            #Ranksums
            X=v1-v2
            X=[x for x in X if x!=0]
            S=pd.Series([np.sign(x) for x in X])
            index_pos=S.index.where(S>0).dropna()
            index_neg=S.index.where(S<0).dropna()
            R=pd.Series([abs(x) for x in X]).rank()
            pos=sum(pd.Series([R[i] for i in index_pos]))
            neg=sum(pd.Series([R[i] for i in index_neg]))
            #table
            results=pg.wilcoxon(v1,v2,alternative=alternative)
            results.insert(0, "ranksum (-)", ['{}'.format(neg)])
            results.insert(0, "ranksum (+)", ['{}'.format(pos)])
            results.insert(0, "median", ['{}'.format(medians[0])+'/'+'{}'.format(medians[1])])
            results.insert(0, "n", [ns[0]])
            results.insert(0, "var", [name_group_diff])
            results.index=['Wilcoxon signed-rank Test']

            return results
        
    '''
    Class independence
    -------------------
    .chi2
        arguments
        ---------
        - data = name of dataframe (var) [optional]
        - var1, var2 = names of the variables in the dataframe (str) or vectors (pd.Series) if data is None
        - yates = whether to apply the yates correction (bool) [default = False]
        returns
        -------
        table of chi2 independence test (pd.DataFrame)
    .exact
        arguments
        ---------
        - data = name of dataframe (var)
        - var = name of the variable in the dataframe (str)
        returns
        -------
        table of containing Fisher, Barnard, and Boschloo Exact Test (pd.DataFrame)
    '''
    class independence:
        def chi2(var1,var2,data=None,yates=False):
            #Preliminaries
            if data is None:
                if isinstance(var1, pd.Series):
                    v1=var1
                    if isinstance(var2, pd.Series):
                        v2=var2
                        name1=var1.name
                        name2=var2.name
                        data=pd.DataFrame()
                        data[name1]=var1
                        data[name2]=var2
                    else:
                        return print('You did not specify a dataframe and var2 is not a series object')
                else:
                    return print('You did not specify a dataframe and var1 is not a series object')
            else:
                v1=data[var1]
                v2=data[var2]
                name1=var1
                name2=var2

            #Length
            catv1=len(v1.unique())
            catv2=len(v2.unique())

            #table
            t=pg.chi2_independence(data=data, x=name1,y=name2,correction=yates)
            results=t[2]
            results.drop('lambda',axis=1,inplace=True)
            results.rename(columns={"pval": "p-val"},inplace=True)
            results['test'].loc[2]='G(log-likelihood)'
            results.insert(0, "no. categories", [catv1,catv2,'','','',''])
            results.insert(0, "vars", [name1,name2,'','','',''])   
            results.index=['Chi2 Tests', 'of Independence','','','', '']

            return results

        def exact(var1,var2,data=None):
            #Preliminaries
            if data is None:
                if isinstance(var1, pd.Series):
                    v1=var1
                    if isinstance(var2, pd.Series):
                        v2=var2
                        name1=var1.name
                        name2=var2.name
                    else:
                        return print('You did not specify a dataframe and var2 is not a series object')
                else:
                    return print('You did not specify a dataframe and var1 is not a series object')
            else:
                v1=data[var1]
                v2=data[var2]
                name1=var1
                name2=var2
            X=pd.crosstab(v1, v2, dropna=False)
            #Statistics
            fisher=stats.fisher_exact(X)
            barnard=stats.barnard_exact(X)
            boschloo=stats.boschloo_exact(X)
            #table
            results=pd.DataFrame()
            results['vars']=[name1,name2,'']
            results['test']=['fisher','barnard','boschloo']
            results['statistic']=[fisher[0],barnard.statistic,boschloo.statistic]
            results['p-val']=[fisher[1],barnard.pvalue,boschloo.pvalue]
            results.index=['Exact Tests', 'of Independence','']

            return results
        
    '''
    Class correlation
    -----------------
    .simple
        arguments
        ---------
        - data = name of dataframe (var) [optional]
        - var1, var2 = names of the variables in the dataframe (str) or vectors (pd.Series) if data is None
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        - method = correlation coefficient ('pearson','spearman','kendall') (str) [default = 'pearson']
        returns
        -------
        table of simple correlation test (pd.DataFrame)
    .partial
        arguments
        ---------
        - data = name of dataframe (var)
        - var1,var2 = name of the variable in the dataframe (str)
        - covar = list of names of the covariates ([str,str,...])
        - groupvar = name of the grouping variable (str)
        - method = correlation coefficient ('pearson','spearman','kendall') (str) [default = 'pearson']
        returns
        -------
        table of partial correlation test (pd.DataFrame)
    .pbc
        arguments
        ---------
        - x = vector of the numerical variable (pd.Series)
        - nom = vector of the nominal variable (pd.Series)
        returns
        -------
        (point-biserial correlation coefficient, p-value) (tup)
    .rbc
        - x = vector of the ordinal variable (pd.Series)
        - nom = vector of the nominal variable (pd.Series)
        returns
        -------
        (rank-biserial correlation coefficient, p-value) (tup)

    .eta
        - x = vector of the numerical variable (pd.Series)
        - nom = vector of the nominal variable (pd.Series)
        returns
        -------
        (eta, p-value) (tup)

    .gk_gamma
        arguments
        ---------
        - x,y = vectors of the variables (pd.Series)
        - alternative = direction of the test ('two-sided', 'left', 'right') (str)
        returns
        -------
        (Goodman and Kruskal's gamma, asymptotic pvalue under null, asymptotic pvalue under alternative,pvalue standard) (tup)
    .cramer
        arguments
        ---------
        - x,y = vectors of the variables (pd.Series)
        returns
        -------
        (Cramer's V, p-value) (tup)
    Remarks:
    -------
    - For .simple and .partial missing values are automatically removed from the data.
    '''
    class correlation:
        def simple(var1,var2,data=None,alternative='two-sided',method='pearson'):
            #Preliminaries
            if data is None:
                if isinstance(var1, pd.Series):
                    v1=var1
                    if isinstance(var2, pd.Series):
                        v2=var2
                        name1=var1.name
                        name2=var2.name
                    else:
                        return print('You did not specify a dataframe and var2 is not a series object')
                else:
                    return print('You did not specify a dataframe and var1 is not a series object')
            else:
                v1=data[var1]
                v2=data[var2]
                name1=var1
                name2=var2

            #table
            results=pg.corr(v1,v2,alternative=alternative,method=method)
            results.insert(3, "alternative", [alternative])
            results.rename(columns={"r": "r ({})".format(method)},inplace=True)
            results.insert(0, "var2", [name2])
            results.insert(0, "var1", [name1])
            results.index=['{} Test of Correlation'.format(method)]

            return results

        def partial(data,var1,var2,covar=[],alternative='two-sided',method='pearson'):
            #table
            results=pg.partial_corr(data=data,x=var1,y=var2,covar=covar,alternative=alternative,method=method)
            results.insert(3, "alternative", [alternative])
            results.rename(columns={"r": "r ({})".format(method)},inplace=True)
            results.insert(0, "covar", [covar])
            results.insert(0, "var2", [var2])
            results.insert(0, "var1", [var1])
            results.index=['{} Partial Correation Test'.format(method)]

            return results

        def pbc(x,nom):
            if len(nom.unique())!=2:
                return print('The nominal variable is not binary!')
            else:
                pass
            enc=dataprep.encoder()
            nom=enc.fit_transform(nom,sparse=True)
            return stats.pointbiserialr(nom, x)

        def eta(x,nom):
            data=pd.DataFrame()
            data['num'],data['nom']=x,nom
            an=pg.anova(dv='num', between='nom', data=data,detailed=True,effsize='n2')
            eta=math.sqrt(an['n2'][0])
            p=an['p-unc'][0]
            return eta,p

        def rbc(x,nom):
            if len(nom.unique())!=2:
                return print('The nominal variable is not binary!')
            else:
                pass
            data=pd.DataFrame()
            data['num'],data['nom']=x,nom
            data['num']=data['num'].rank()
            groups=dataprep.group_sep(data=data,groupvar='nom')
            g1=groups[0]['num']
            g2=groups[1]['num']
            r=pg.mwu(g2,g1)
            return r['RBC'][0], r['p-val'][0]

        def gk_gamma(x,y,alternative='two-sided'):
            ctab=pd.crosstab(x,y)
            cd=tools.cd_pairs(x=x,y=y)
            P,Q,co,dc,=2*cd[0],2*cd[1],cd[2],cd[3]
            pairs=P+Q
            gamma=(P-Q)/pairs
            agg0=(co-dc)**2*ctab
            agg1=(co*Q-dc*P)**2*ctab
            ase0=2*math.sqrt(agg0.to_numpy().sum()-(P-Q)**2/len(x))/pairs
            ase1=4*math.sqrt(agg1.to_numpy().sum())/pairs**2
            z0=gamma/ase0
            z1=gamma/ase1
            zs=gamma*math.sqrt(pairs/(len(x)*(1-gamma**2)))
            def pval(z,alternative,gamma):
                if alternative=='two-sided':
                    return 2*stats.norm.sf(z)
                elif (gamma<=0 and alternative=='left') or (gamma>=0 and alternative=='right'):
                    return stats.norm.sf(abs(z))
                elif (gamma>0 and alternative=='left') or (gamma<0 and alternative=='right'):
                    return stats.norm.cdf(abs(z))
                else:
                    pass      
            p0,p1,ps=pval(z0,alternative=alternative,gamma=gamma),pval(z1,alternative=alternative,gamma=gamma),pval(zs,alternative=alternative,gamma=gamma)

            return gamma,p0,p1,ps

        def cramer (x,y):
            ctab=stats.contingency.crosstab(x, y)[1]
            pvalue=stats.chi2_contingency(ctab)[1]
            cramer=stats.contingency.association(ctab)
            return cramer,pvalue
        
    '''
    Class equal_var
    -----------------
    .levene
        arguments
        ---------
        - data = name of dataframe (var)
        - var = name of the variable in the dataframe (str)
        - groupvar = name of the grouping variable in the dataframe (str)
        - rem = whether or not to show explanatory remarks (bool) [defaul = False]
        returns
        -------
        table of Levene's test (pd.DataFrame)
    .bartlett
        arguments
        ---------
        - data = name of dataframe (var)
        - var = name of the variable in the dataframe (str)
        - groupvar = name of the grouping variable in the dataframe (str)
        returns
        -------
        table of Bartlett test (pd.DataFrame)
    '''
    class equal_var:
        def levene(data,var,groupvar,rem=False):
            s=dataprep.group_sep(data,groupvar)
            k=len(list(data[groupvar].unique()))

            if rem==True:
                rem=['var','group','','f','dof1','dof2','p-val','remark']
            else:
                rem=['var','group','','f','dof1','dof2', 'p-val']

            #values
            plevene=stats.levene(*tuple([s[i][var] for i in range(0,k)]),center="mean").pvalue
            slevene=stats.levene(*tuple([s[i][var] for i in range(0,k)]),center="mean").statistic
            plevenem=stats.levene(*tuple([s[i][var] for i in range(0,k)]),center="median").pvalue
            slevenem=stats.levene(*tuple([s[i][var] for i in range(0,k)]),center="median").statistic
            plevenet=stats.levene(*tuple([s[i][var] for i in range(0,k)]),center="trimmed").pvalue
            slevenet=stats.levene(*tuple([s[i][var] for i in range(0,k)]),center="trimmed").statistic

            #dof
            df1=k-1
            df2=data[var].count()-k

            # results
            results = {'var':[var,'',''],'group':[groupvar,'',''],'':['Mean', 'Median', 'Trimmed'],'dof1':[df1,df1,df1],'dof2':[df2,df2,df2], 'f':[slevene,slevenem,slevenet], 'p-val':[plevene,plevenem,plevenet],'remark':['for symmetric, moderate-tailed distributions','for skewed (non-normal) distributions','for heavy-tailed distributions']}

            table_levene = pd.DataFrame(results, columns = rem, index=['Levenes Test', 'of Equal Variances',''])

            return table_levene

        def bartlett(data,var,groupvar):
            #dof
            k=len(list(data[groupvar].unique()))
            df1=k-1
            df2=data[var].count()-k
            #table
            results=pg.homoscedasticity(data=data,dv=var,group=groupvar,method='bartlett')
            results.insert(1, "dof2", [df2])
            results.insert(1, "dof1", [df1])
            results.insert(0, "group", [groupvar])
            results.insert(0, "var", [var])
            results.drop('equal_var',axis=1,inplace=True)
            results.index=['Bartletts Test of Equal Variances']
            return results


# In[ ]:





# In[ ]:


'''
Class regression
-----------------
arguments
---------
- X = matrix of independent variables (pd.DataFrame)
- y = dependent variable (pd.Series)
- regression = type of regression ('linear','logistic','multinomial','ordinal') (str) [default = 'linear']
- var = name of the variable in the dataframe (str)
returns
-------
.resid -> (pseudo) residuals of the regression
.pred -> predictions of the regression
.datafit -> general information and goodness of fit of the regression
.coef -> table with coefficients
.vif -> variance inflation factors
.asscheck -> tests to check assumptions
.summary -> summary of regression analysis
'''

class regression:
   def __init__(self,X,y,regression='linear'):
       # method dictionary
       mdic={'linear':sm.OLS,'logistic':sm.Logit,'multinomial':sm.MNLogit,'ordinal':OrderedModel}
       namey=y.name
       encoder=dataprep.encoder()
       #Remove NaN
       df=X.copy()
       df['y']=y
       df=df.dropna()
       y=df['y']
       X=df.drop('y',axis=1)
       #Add constant
       if regression in ['linear','logistic','multinomial']:
           X=sm.add_constant(X)
           X=X.rename(columns={'const': 'intercept'})
       else:
           pass
       #Format Dependent Logistic
       if regression=='logistic':
           y=encoder.fit_transform(y,sparse=True)
       else:
           pass
       #Fit the Model
       model=mdic[regression](y,X).fit()
       #summary
       summary=model.summary()
       #get residuals
       ypred=model.predict(X)
       if regression in ['linear','logistic']:
           residuals=y-ypred
       else:
           y_str=y.astype(str)
           y_dum=pd.get_dummies(pd.DataFrame(y_str))
           coldum=list(y_dum.columns)
           for i in range(len(y.unique())):
               y_dum=y_dum.rename(columns={coldum[i]: i})
           residuals=y_dum-ypred
           ydic={}
           for i in range(len(y.unique())):
               ydic[i]=sorted(list(y.unique()))[i]
           residuals=residuals.rename(columns=ydic)
       #get tables
       table1=tools.get_table(summary,i=0)
       table2=tools.get_table(summary,i=1)
       table2=table2.reset_index()
       #datafit
       dfn, dfk= model.df_resid, model.df_model
       fit=pd.DataFrame()
       fit['dv']=[namey]
       fit['dof resid']=[dfn]
       fit['dof model']=[dfk]
       if regression=='linear':
           fit['R2']=[model.rsquared]
           fit['adj. R2']=[model.rsquared_adj]
           fit['omnibus (F)']=[model.fvalue]
           fit['omnibus (p-val)']=[model.f_pvalue]
           fit['LL']=[model.llf]
       else:
           fit['pseudo R2']=[model.prsquared]
           fit['LL']=[model.llf]
           fit['LLR']=[model.llr]
           fit['LLR (p-val)']=[model.llr_pvalue]
       fit.index=[regression+' reg.'+' fit']

       #results (coefficients)
       results=table2.copy()
       if regression in ['linear','logistic','ordinal']:
           results.index=[regression+' reg.']+['coefficients']+['']*(len(results)-2)
           results=results.rename(columns={"index": ""})
           if regression=='linear':
               scaler=StandardScaler()
               coef=results[['coef']]
               coef=list(scaler.fit_transform(coef).reshape(1,-1)[0])
               results.insert(2, 'stand. coef', coef)
           elif regression=='logistic':
               exp=[math.exp(x) for x in results['coef']]
               results.insert(2, 'exp(coef)', exp)
           else:
               pass
       elif regression=='multinomial':
           name0=list(results.columns)[0]
           names=[name0]+list(results[name0])
           names_ninter=[i for i in names if i!='intercept']
           ibool=pd.Series(names_ninter).str.contains('=')
           category=[names_ninter[i] if ibool[i]==True else '' for i in range(len(names_ninter))]
           results=results[~results[name0].str.contains('=')] 
           results=results.rename(columns={name0: ""})
           results.insert(0,'category',category)
           results.index=[regression+' reg.']+['coefficients']+['']*(len(results)-2)
       else:
           pass
       #assumptions
       l=len(list(X.columns))
       if regression!='ordinal':
           l1=l-1
       else:
           l1=l
       if l1>1:
           vif=pd.DataFrame()
           vifs=[variance_inflation_factor(X,i) for i in range(l)]
           vif['var']=list(X.columns)
           vif['vif']=vifs
           vif.index=['variance inflation','factors']+['']*(l-2)
       else:
           vif = 'There is only 1 independent variable. I cannot compute vifs.'
       ass=pd.DataFrame()
       if regression=='linear':
           jb,bp,dw,rr=stats.jarque_bera(residuals),sm.stats.diagnostic.het_breuschpagan(residuals,X), sm.stats.stattools.durbin_watson(residuals), statsmodels.stats.outliers_influence.reset_ramsey(model)
           ass['test']=['Jarque-Bera','Breusch-Pagan','Durbin-Watson','Ramsey RESET']
           ass['statistic']=[jb[0],bp[0],dw,rr.statistic]
           ass['p-val']=[jb[1],bp[1],1,rr.pvalue]
           ass=ass.round(4)
           ass=ass.replace({'p-val':{1:''}})
           ass.index=[regression+' reg.']+['assumptions']+['','']
       else:
           ass = 'No assumption check implemented for this regression type yet'
           
       #self methods
       self.resid = residuals
       self.pred = ypred
       self.datafit = fit
       self.coef = results
       self.vif = vif
       self.asstest =ass
       self.summary = summary


# In[ ]:





# In[ ]:


'''
Class outlier
-----------------
.univariate
   arguments
   ---------
   - x = vector (pd.Series)
   - k = detection factor either 'auto' (str) or [(dec,dec,dec)] for zscore,iqr,mad [default = 'auto']
   returns
   -------
   .analysis -> outlier analysis summary
   .show -> indices of outliers detected by the specified method
       arguments
       ---------
       - method = outlier detection method ('zscore','iqr','mad') (str) [default = 'iqr']
.multivariate
   arguments
   ---------
   - x = matrix of independent variables (pd.DataFrame)
   - y = dependent variable (pd.Series)
   - k = detection factor either 'auto' (str) or [(dec,dec)] for Cook,Mahalanobis [default = 'auto']
   returns
   -------
   .analysis -> outlier analysis summary
   .show -> indices of outliers detected by the specified method
       arguments
       ---------
       - method = outlier detection method ('Cook','Mahalanobis') (str) [default = 'Cook']
'''

class outlier:
   class univariate:
       def __init__(self,x,k='auto'):
           #preliminaries
           k_auto=[3,1.5,math.sqrt(stats.chi2.ppf(0.975, df=1))]
           ana=pd.DataFrame()
           ana['method']=['zscore','iqr','mad']
           dick={}
           dicl={}
           dicu={}
           dico={}
           if k=='auto':
               for i in range(3):
                   dick[ana['method'][i]]=k_auto[i]
           else:
               for i in range(len(ana['method'])):
                   dick[ana['method'][i]]=k[i]
           for m in ana['method']:
               dicl[m]=tools.extreme(x=x,method=m,k=dick[m])[0]
               dicu[m]=tools.extreme(x=x,method=m,k=dick[m])[1]
           for m in ana['method']:
               dico[m]=x.where((x<dicl[m]) | (x>dicu[m])).dropna().index.to_list()
           ana['pot. outlier']=[len(dico[m]) for m in ana['method']]
           ana['proportion']=[str(round(ana['pot. outlier'][i]/len(x)*100,2))+'%' for i in range(len(ana['method']))]
           #under Normal
           norm_prob=stats.norm.sf(dick['zscore'])*2
           norm_per=str(round(norm_prob*100,2))+'%'
           norm_abs=math.floor(norm_prob*len(x))
           ana.loc[3]=['E[ND] (>{} std from mean)'.format(dick['zscore']),norm_abs,norm_per]
           ana.index=['extreme value','analysis','','']
           self.analysis=ana
           self.dico=dico
       def show(self, method='iqr'):
           return self.dico[method]
   class multivariate:
       def __init__(self,x,y,k='auto'):
           #reset_indices
           #preliminaries
           ana=pd.DataFrame()
           ana['method']=['Cook','Mahalanobis']
           cook=CooksDistance()
           ddist={}
           dtresh={}
           dico={}
           #distances
           ddist['Cook']=cook.fit(x,y).distance_
           plt.close()
           X_new=x.copy()
           X_new[y.name]=y
           ddist['Mahalanobis']=pd.Series([tools.mahalanobis(X_new,i) for i in list(x.index)])
           ddist['Mahalanobis']=ddist['Mahalanobis'].set_axis(list(x.index))
           #tresholds
           if k=='auto':
               dtresh['Cook']=4/len(y)
               dtresh['Mahalanobis']=stats.chi2.ppf(0.975, df=len(X_new.columns))
           else:
               for i in range(len(ana['method'])):
                   dick[ana['method'][i]]=k[i]
           #outliers
           for m in ana['method']:
               dico[m]=ddist[m].where(ddist[m]>dtresh[m]).dropna().index.to_list()
           #table
           ana['pot. outlier']=[len(dico[m]) for m in ana['method']]
           ana['proportion']=[str(round(ana['pot. outlier'][i]/len(x)*100,2))+'%' for i in range(len(ana['method']))]
           ana.index=['extreme value','analysis']
           self.analysis=ana
           self.dico=dico
           self.ddist=ddist
       def show(self, method='Cook'):
           return self.dico[method]
       


# In[ ]:




