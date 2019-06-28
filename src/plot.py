"""
plot heatmap of matrix and learning curve
"""
import argparse
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(X,title):
    #canvas setup
    sns.set(font_scale = 2)
    fig,ax = plt.subplots()
    fig.set_size_inches(10, 10)

    #plot
    sns.heatmap(X,cmap='YlGnBu')
    plt.title(title)

    #save file
    file_name = '_'.join(title.split(' '))
    #plt.show()
    plt.savefig('{}.png'.format(file_name), bbox_inches = 'tight', pad_inches =0.5)

def plot_learning_curve(X,title):
   #canvas setup
    sns.set(style="whitegrid", color_codes=True,font_scale=2)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    #plot
    plt.plot(X)
    plt.title(title)

    #save file
    file_name = '_'.join(title.split(' '))
    #plt.show()
    plt.savefig('{}.png'.format(file_name), bbox_inches = 'tight', pad_inches =0.5)




