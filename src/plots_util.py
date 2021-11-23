# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pred_scatter_plot(real_values, pred_values, title, xlabel, ylabel, savefig, figure_name):
    fig, ax = plt.subplots()
    ax.scatter(real_values, pred_values, c='orangered',
               alpha=0.6, edgecolors='black')
    # ax.plot([real_values.min(),real_values.max()],[real_values.min(),real_values.max()],'k--',lw = 4)
    ax.plot(real_values, real_values, 'k--', lw=4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.title(title)
    plt.xlim([4, 11])
    plt.ylim([4, 11])
    plt.show()
    if savefig:
        fig.savefig(figure_name, dpi=1200, format='eps')


def feature_rel_density_map(feature_rel, thresholds, window_values, title, xlabel, ylabel, savefig, figure_name):

    x_axis, y_axis = np.meshgrid(window_values, thresholds)
    z_axis = np.array([i.iloc[:, 0] for i in feature_rel])

    fig = plt.figure()
    contourf_plot = plt.contourf(x_axis, y_axis, z_axis, levels=np.linspace(
        0, 100, 11), cmap='Reds', linestyles='solid')

    plt.title(title)
    plt.xticks(window_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    colorbar_plot = plt.colorbar(contourf_plot)
    plt.show()
    if savefig:
        fig.savefig(figure_name, dpi=1200, format='eps')


def pssm_window_heatmap(pssm_window_match, window_values, pssm_thresholds, title, xlabel, ylabel, savefig, figure_name):
    fig = plt.figure()

    ax = sns.heatmap(np.array([i.iloc[:, 0] for i in pssm_window_match]), vmin=0, vmax=100, annot=True,
                     fmt='.2f', linewidths=.5, cmap='Reds', xticklabels=window_values, yticklabels=pssm_thresholds)

    ax.invert_yaxis()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()
    if savefig:
        fig.savefig(figure_name, dpi=1200, format='eps')


def pssm_feature_relevance_heatmap(pssm_feature_relevance,
                                   relevance_threshold, pssm_thresholds,
                                   title, xlabel, ylabel, savefig, figure_name):
    fig = plt.figure()
    ax = sns.heatmap(pssm_feature_relevance, vmin=0, vmax=100, annot=True,
                     fmt='.2f', linewidths=.5, cmap='Reds', xticklabels=relevance_threshold, yticklabels=pssm_thresholds)

    ax.invert_yaxis()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()
    if savefig:
        fig.savefig(figure_name, dpi=1200, format='eps')



def min_max_scale(data):
    data_scaled = (data - np.min(data)) / ((np.max(data) - np.min(data)) + 1e-05)

    return data_scaled



def gradram_plot(data_input, ram_input, binding_sites, labels, legend_colors,legend_text, legend_marker, xlabel,
                 savefig=False, figure_name=None, offset=50):
    
    max_seq_len = max([len(i[1]) for i in data_input])

    ram_norm_list = [min_max_scale(ram_input[i])[:len(
            data_input[i][1])] for i in range(len(ram_input))]

    ram_norm_pos = [i[i > 0] for i in ram_norm_list]
    
  
    indices_list = [np.where(i > 0)[0] for i in ram_norm_list]
    


    fig = plt.figure(figsize=(7, 4), dpi=1200)
    ax = fig.add_subplot()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tick_params(top=False, bottom=True, left=False, right=False,
                        labelleft=False, labelbottom=True)


    plt.xticks([i for i in range(offset, max_seq_len+offset, 2*offset)],
                   [j-offset for j in [i for i in range(offset, max_seq_len+offset, 2*offset)]],fontsize=7)
    plt.xlim([0, max_seq_len+offset])


        
    labels = labels
        
    unique_data = list(sorted(set([i[0] for i in data_input])))

    for i in range(len(data_input)):

        plt.plot([offset+j for j in range(len(data_input[i][1]))],
                     [i]*len(data_input[i][1]), 'black', lw=0.5)
        
        color = legend_colors[unique_data.index(data_input[i][0])]

        ax.text(10, i, s=labels[i], va="center", ha='center',fontsize=5)



        plt.scatter([k+offset for k in binding_sites[i]], [i-0.1]
                        * len(binding_sites[i]), s=5, color=color)

        for j in range(len(indices_list[i])):
            plt.vlines(indices_list[i][j]+offset, ymin=i,
                           ymax=i+ram_norm_list[i][indices_list[i][j]]/2,
                           colors='black', ls='-', lw=1.5)
                
                
    colors = legend_colors  
    texts = legend_text
        
    markers = legend_marker               
                
    patches = [ plt.plot([],[], marker=markers[i], ms=6, ls="", color=colors[i], 
            label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
    plt.legend(handles=patches, 
           loc='center right', ncol=1, numpoints=1,fontsize=8, facecolor='white', edgecolor='black', framealpha=1)
        
    plt.xlabel(xlabel,fontsize=8)

    if savefig:
        plt.savefig(figure_name, dpi=1200, format='png',bbox_inches='tight')
