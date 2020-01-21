## SETUP THE ENVIRONMENT 

## Import general system modules
from google.colab import drive
from google_drive_downloader import GoogleDriveDownloader
from itertools import product
from itertools import chain
import random 
import pickle
import gzip
import sys
import os


## Core modules
import pandas as pd
import numpy as np

## Import plotting libraries
from matplotlib import pyplot as plt
import matplotlib 
import seaborn as sns

## ML modules
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from sklearn.cluster import DBSCAN
import umap

## UTILITY FUNCTIONS ## 


## Define simple function that will truncate numeric variable within trunc_range
## This truncation is based on quantile function 

def truncate_by_quantile(series, trunc_range = [0, 1]):
    wk_series = series.copy()
    q_values = series.quantile(trunc_range).values
    q_min = np.min(q_values)
    q_max = np.max(q_values)
    wk_series = np.where(np.logical_and(wk_series < q_min,~np.isnan(wk_series)) , q_min, wk_series)
    wk_series = np.where(np.logical_and(wk_series > q_max,~np.isnan(wk_series)) , q_max, wk_series)
    return(wk_series)

## Take single vm and plot it 
def plot_single_vm(__perf_data, __vmid = None, transformation = None):

    ## Calculate dims 
    __N = len(__perf_data)
    __VMs = list(__perf_data.keys())
    __T , __D = __perf_data[__VMs[0]].shape
    __F = list(__perf_data[__VMs[0]].columns)

    ## 
    CHECK_VMS = any([i == __vmid for i in __VMs])

    ## Check vms
    if not CHECK_VMS:

        if __vmid is not None:
            print(__vmid, 'is not valid ID!\nRandom vm has been picked! ')

        vmIx = random.randint(0,(__N-1))
        __vmid = __VMs[vmIx] 

    ### 
    if transformation is None or transformation =='None':
        transformation  = return_transformer(None)

    ## Create instance of the data for this vm: 
    __vmid, data  = sample_observation(__perf_data, vmid = __vmid)

    ## Create VMId + VM index title: 
    plotTitle = 'VM ID:'+str(__vmid)

    ##  Create the plot: 
    ts, axes =  plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(30,8))
    ts.suptitle(plotTitle, fontsize=14)


    for i, ax in enumerate(axes.reshape(-1)):

        metric = __F[i]
        data_serices = data[metric]
        
        data_serices = transformation(data_serices)


        # For each metric find indexes with missing values 
        missing_ix = data[data[metric].isna()].index
        minVal = data_serices.min()
        dummy_x = [minVal for i in range(len(missing_ix))]

        ax.set_ylabel(metric)
        ax.yaxis.set_label_position('right')
        ax.plot(data.index,data_serices)
        ax.scatter(missing_ix,dummy_x , c = 'red', s = 18, marker = 'x')
    plt.show()

## This is a utility function that samples the information for single virtual machine 
## It can sample random as well as specific virtual machine and metric (metrics)
## It also can remove rows with missing values

def sample_observation(data, metrics = 'All',vmid = None):
    N = len(data)
    VMs = list(data.keys())
    if vmid == None:
        vmIx = random.randint(0,(N-1))
        vmid = VMs[vmIx]
    
    ## Selec the data
    data_sample = data[vmid].copy()

    if metrics != 'All':
        data_sample = data_sample.loc[: , data_sample.columns.intersection(metrics)]

    return(vmid, data_sample)


## These function setup some groundwork for easier training
## This function takes python dict where keys represent algorithm parameters
## the element of the list represent lists with parameter values 
## eg for t-sne param_dict = {'perplexity':[5,10], 'n_components':[2]}
## the function creates generator with all the combinations of the parameter values  

def model_params_product(pram_dict):
    keys = pram_dict.keys()
    vals = pram_dict.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

## This is a function for model tunning 
## it takes dataset, model function, hyperparameters, specific to this function 
## and whether or not to apply PCA and how many pc to include 
## other parameters are model parameters that are constant across all ins

def model_train(data, model , param_dict , apply_pca = False, n_pc = None, other_params = {}):
    
    ## Get the data shape 
    Nrows, Ncols = data.shape
    model = str(model).upper()
    ## Init the arrays that we will use for modeling
    if isinstance(data, pd.DataFrame):
        data_instance = data.values
    else:
        data_instance = data

    n_features = [Ncols]

    ## Check if there should be pca
    ## and if all n_components are integer values 

    if apply_pca:
        if n_pc == None:
            raise Exception('Provide int or list with number of components')
        
        if all([isinstance(x, int) for x in n_pc]):
            max_n_pc = max(n_pc)
            if max_n_pc > Ncols:
                print('The max number of PC({}) is greater than the number of features in the dataset ({}).\nThe max number of PC has been truncated to {}!\n'.format(max_n_pc,Ncols, Ncols))

            n_features = list(sorted(set([x if x <= Ncols else Ncols for x in n_pc ])))
            data_instance = PCA().fit_transform(data)

        else:
            raise Exception('Values', ', '.join([str(x) for x in  n_pc if not(isinstance(x, int))]), ' are not valid integers')
    

    models_ph = []

    i = 0

    for nfeat in n_features:
        ## Create generator with the hyperparams:
        model_params = model_params_product(param_dict)

        for model_param in model_params:
            params_for_tunning = model_param.copy()
            model_param.update(other_params)

            if model == 'DBSCAN':
                running_model = DBSCAN(**model_param).fit_predict(data_instance[:, :nfeat])
            elif model == 'TSNE':
                running_model = TSNE(**model_param).fit_transform(data_instance[:, :nfeat])
            elif model == 'UMAP':
                running_model = umap.UMAP(**model_param).fit_transform(data_instance[:, :nfeat])

            ## Add the number of PCA Components as additional argument
            if apply_pca:
                model_param['n_pc'] = nfeat
                params_for_tunning['n_pc'] = nfeat
            
            ## Log the progress
            print('Finished iteration ',i,' with parameters:', ', '.join([ k+':'+str(v) for k, v in model_param.items()]))
            i += 1

            ## Append the tuple of the parameters and the fitted model to list 
            models_ph.append((params_for_tunning, running_model, other_params))
    
    print('Training has finished with {} tested combinations.'.format(len(models_ph)))
    
    return(models_ph)   

## This is a plotting function that will help us visualize the results of our embedding. 
## It takes a result tuple (as generated in the function model_train) and plots the first two dimensions of the embedding. 


def plot_results(results, leading_param = None, color_array = None, force_categorical = False, point_size = 3):
    """
    This is simple plotting fuction that take tuning results object 
    and yields plot grid. The grid can be colored by specific variable
    
   results: model results as generated from model_train function 
   leading_param: the parameter that will serve for aligning the plots. leading_param is plotted row wise
   color_array: data, by which the plots will be colored. Can be single numpy vector or list of vectors with the same size to the results list 
   force_categorical: Should coloring be represented on a categorical scale  
    """

    ## Obtaining all distinct hyperparameters used for tuning:
    tuned_params = list(set(chain.from_iterable([list(r[0].keys()) for r in results])))
    
    ## Check if leading param is in the tuned_params
    lp_in_tuned_params = any([p  == leading_param for p in tuned_params])

    if leading_param == None or not(lp_in_tuned_params):
        leading_param = tuned_params[0]
        print('No valid leading parameter is given!')
        print('Valid tuned params are:', ', '.join(tuned_params))
        print(leading_param, 'is picked as leading param!')
    
    ## Check if the color variable
    if isinstance(color_array, list):
        if len(color_array) != 1 and len(color_array) != len(results):
            print(len(color_array))
            raise Exception('The length of the color array must be either 1 or equal to the length of the tuning results!')

    ## Obtain the distinct values of the leading param  
    leading_param_values = [meta[leading_param] for meta, array, _ in results]
    leading_param_values_length = len(leading_param_values)
    ## 
    leading_param_distinct_values = sorted(set(leading_param_values))
    leading_param_distinct_values_length = len(leading_param_distinct_values)

    ## Set the order of the subplots
    subplots_order = [results[k] for _, k in sorted(zip(leading_param_values, range(leading_param_values_length)))]
    orig_index = [k for _, k in sorted(zip(leading_param_values, range(leading_param_values_length)))]


    ## Define grid dims
    nrow = int(leading_param_distinct_values_length)
    ncol = int(leading_param_values_length/leading_param_distinct_values_length)

    ## If there is only 1 variable plot the results on a single row
    if ncol ==1:
        ncol = nrow
        nrow = 1
    
    ## Define plot:
    figsize_width =  (ncol  + 0.5 )* 6
    figsize_height = nrow * 6

    fig, ax = plt.subplots(nrows = nrow, ncols = ncol, figsize = (figsize_width,figsize_height))
    
    
    ## Init some variables
    color_var = None

    ## Plot
    for i, result in enumerate(subplots_order):
        meta_info = result[0]
        ylab = leading_param +':'+ str(meta_info[leading_param])
        col_ix = orig_index[i]
        title = ',   '.join([ k+':'+str(v) for k, v in meta_info.items() if k != leading_param])+'('+str(col_ix)+')'
        working_ax = ax.reshape(-1)[i]
        working_ax.set_title(title)
        working_ax.set_ylabel(ylab)

        ## Check if we should color 
        ## Define the color variable 
        
        if color_array != None:
            if len(color_array) == len(results):
                color_var  = color_array[col_ix]
            else:
                 color_var = color_array[0]

            ## Check if the color var is numeric or categorical: 
            col_var_numeric = np.issubdtype(np.array(color_var).dtype, np.number)
            if col_var_numeric and not(force_categorical):
                myplt = working_ax.scatter(x = result[1][:,0],y = result[1][:,1], c = color_var,cmap = plt.cm.get_cmap('Spectral'), s = point_size)
                plt.colorbar(myplt, ax = working_ax)
            else:
                ## Define custom color palette:
                color_set_values = list(sorted(set(color_var)))
                color_set_length = len(color_set_values)
                color_set_values_indeces = list(range(color_set_length))

                color_set_index = dict([(str(v), i) for i, v in enumerate(color_set_values)])
                value_set_index = dict([(str(i), v) for i, v in enumerate(color_set_values)])
                color_var = [color_set_index[str(i)] for i in color_var]

                ## Apply label formatting:
                formatter = plt.FuncFormatter(lambda x ,loc: value_set_index[x])

                myplt = working_ax.scatter(x = result[1][:,0] ,y = result[1][:,1], c = color_var,cmap = plt.cm.get_cmap('Spectral', color_set_length), s = point_size)
                plt.colorbar(myplt, ax = working_ax,ticks = color_set_values_indeces) #, format = formatter
        else:
            myplt = working_ax.scatter(x = result[1][:,0] ,y = result[1][:,1], s = point_size)
    fig.show()


##########
def plot_embedding(result, fig_scale = 1, color_var_list = None, force_categorical = False, plot_centers = False, dont_plot_minus_one=True, point_size = 4):
    meta, emb, _ = result
    _, Ncomp = emb.shape

    plot_title = ', '.join([k+':'+str(v) for k, v in meta.items()])
    w_size, h_size =  (Ncomp-0.4)*6*fig_scale, (Ncomp-1)*6*fig_scale
    subtitle_y = round((Ncomp-1)/Ncomp,3) + (0.12/fig_scale)/Ncomp


    fig, ax = plt.subplots(nrows = Ncomp, ncols = Ncomp, figsize = (w_size, h_size), sharex=True, sharey=True)
    
    for i in range(Ncomp):
        for j in range(Ncomp):
            if i<=j:
                fig.delaxes(ax[i][j])
                continue
            
            working_ax = ax[i][j]

            if color_var_list is not None:
                color_var = color_var_list[0]
                col_var_numeric = np.issubdtype(np.array(color_var).dtype, np.number)

                if col_var_numeric and not(force_categorical):
                    myplt = working_ax.scatter(x = emb[:,j], y = emb[:,i], c = color_var,cmap = plt.cm.get_cmap('Spectral'), s = point_size)
                    plt.colorbar(myplt, ax = working_ax)

                else:
                    ## Define custom color palette:
                    color_set_values = list(sorted(set(color_var)))
                    color_set_length = len(color_set_values)
                    color_set_values_indeces = list(range(color_set_length))

                    color_set_index = dict([(str(v), i) for i, v in enumerate(color_set_values)])
                    value_set_index = dict([(str(i), v) for i, v in enumerate(color_set_values)])
                    color_var = [color_set_index[str(i)] for i in color_var]
                    myplt = working_ax.scatter(x = emb[:,j], y = emb[:,i], c = color_var,cmap = plt.cm.get_cmap('Spectral', color_set_length), s = point_size)
        
                    ## Apply label formatting:
                    formatter = plt.FuncFormatter(lambda x ,loc: value_set_index[str(x)])


                    plt.colorbar(myplt, ax = working_ax, ticks = color_set_values_indeces, format = formatter) #

                    ## Plot cluster centers: 
                    if plot_centers:
                        ## Create ph for the calculated centers
                        calc_centers = []
                        for ctr in np.sort(np.unique(color_var)):
                            arr_ix = np.where(color_var == ctr)                      
                            str_label = value_set_index[str(ctr)]
                            if dont_plot_minus_one and str(str_label) == '-1':
                                continue 

                            running_mean = np.mean(emb[arr_ix, :], axis  = 1, keepdims=False)[0]
                            
                            calc_centers.append(running_mean)
                            working_ax.text(running_mean[j], running_mean[i], str_label
                                        ,   bbox=dict(boxstyle="square",
                                            ec=(1., 0.5, 0.5),
                                            fc=(1., 0.8, 0.8),
                                            ))
                            
            else:
                working_ax.scatter(emb[:,j], emb[:,i], s = point_size)

            working_ax.set_xlabel('Component '+str(i+1))
            working_ax.set_ylabel('Component '+str(j+1))
    fig.tight_layout()
    #fig.suptitle()
    fig.suptitle(plot_title, y = subtitle_y, x = 0.05,horizontalalignment = 'left')
    plt.show()
    