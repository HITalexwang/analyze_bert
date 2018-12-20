# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import argparse
from collections import Counter

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    #print data, label, n_samples, n_features
    #exit()
    return data, label, n_samples, n_features

def load_map(file):
    map = {}
    with open(file, 'r') as fi:
        for line in fi.read().strip().split('\n'):
            items = line.split()
            token = items[0]
            if token not in map:
                map[token] = {items[1]:{'sense':items[2], 'pos':items[3], 'suffix':items[4]}}
            else:
                map[token][items[1]] = {'sense':items[2], 'pos':items[3], 'suffix':items[4]}
    return map

def load_label(file, n_token=0, map_file=None, key=None):
    color_map = {}
    colors = []
    with open(file, 'r') as fi:
        label = fi.read().strip().split('\n')
    if n_token > 0:
        label = label[:n_token]
    if map_file:
        map = load_map(map_file)
        #print map["giant"]
        #exit()
        if key:
            assert key in map
            for i in range(len(label)):
                if label[i] in map[key]:
                    label[i] = map[key][label[i]]["pos"]
                else:
                    print "{} not in {} map!".format(label[i], key)
                    #exit()
    for l in label:
        if l not in color_map:
            color_map[l] = len(color_map)
        colors.append(color_map[l])
    #print (zip(label, colors))
    print color_map
    print Counter(colors)
    #exit()
    return label, colors, color_map

def load_bert(file, n_token=0, map_file=None):
    sents = []
    data = []
    label = []
    lems = []
    tags = []
    n = 0
    with open(file, 'r') as fi:
        line = fi.readline()
        while line:
            bert = json.loads(line)
            tok = bert["tokens"][bert["position"]]
            data.append(bert["values"])
            tags.append(bert["label"])
            lems.append(bert["lemma"])
            label.append(tok+'-'+str(bert["index"]))
            sents.append(bert["tokens"])
            n += 1
            if n_token > 0 and n >= n_token: break
            line = fi.readline()
    data = np.array(data)
    n_samples, n_features = data.shape
    #print data, ' '.join(label), n_samples, n_features
    #exit()

    color_map = {}
    colors = []
    if map_file:
        key = lems[0]
        map = load_map(map_file)
        #print map["giant"]
        #exit()
        if key:
            print key
            assert key in map
            for i in range(len(tags)):
                if tags[i] in map[key]:
                    tags[i] = map[key][tags[i]]["pos"]
                else:
                    print "{} not in {} map!".format(tags[i], key)
    for l in tags:
        if l not in color_map:
            color_map[l] = len(color_map)
        colors.append(color_map[l])

    print color_map
    print Counter(colors)

    return data, label, n_samples, n_features, sents, colors, color_map, tags, lems

def plot_embedding(data, label, title, colors, color_map=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    if color_map:
        for n,key in enumerate(color_map):
            plt.text(0,1.05-0.03*n,key, color=plt.cm.Set1(color_map[key]))
    for i in range(data.shape[0]):
        #print color.index(label[i].split('-')[0]) / 10.
        #plt.text(data[i, 0], data[i, 1], str(label[i]),
        #         color=plt.cm.Set1(color.index(label[i].split('-')[0])),
        #         fontdict={'weight': 'bold', 'size': 9})
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(colors[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input bert for tokens')
    parser.add_argument('--n_token', type=int, default=0, help='max tokens(0 to disable)')
    parser.add_argument('--output', type=str, default=None, help='output sentences')
    parser.add_argument('--label', action='store_true', default=False, help='use labels')
    parser.add_argument('--key', type=str, default=None, help='key word')
    parser.add_argument('--map', type=str, default=None, help='input map from word sense to POS')
    args = parser.parse_args()
    data, label, n_samples, n_features, sents, colors, color_map, tags, lems = load_bert(args.input, args.n_token, args.map)
    #if args.tag:
        #tag, colors, color_map = load_label(args.tag, args.n_token, map_file=args.map, key=args.key)
    if args.output:
        with open(args.output, 'w') as fo:
            for l, sent, tag in zip(label, sents, tags):
                fo.write("{}: tag:{}, {}\n".format(l, tag, ' '.join(sent).encode('utf-8')))
    #print (zip(label, tag, colors))
    #exit()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    title = 't-SNE embedding of word {} (time {:.2f}s)'.format(lems[0], time() - t0)
                        
    fig = plot_embedding(result, label, title, colors, color_map)
    plt.show(fig)


if __name__ == '__main__':
    main()