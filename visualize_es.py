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
import matplotlib.lines as mlines
import sys

def load_bert(file, n_token=0, map_file=None):
    sents = []
    data = []
    tags = []
    ids = []
    n = 0
    with open(file, 'r') as fi:
        line = fi.readline()
        while line:
            bert = json.loads(line)
            tok = bert["tokens"][bert["position"]]
            data.append(bert["values"])
            id, labs = bert["label"].split('\t')
            ids.append(id)
            tags.append(labs)
            #tags.append(labs.strip('::').split(':',1)[1])
            #lems.append(bert["lemma"])
            #label.append(tok+'-'+str(bert["index"]))
            #lab = str(bert["index"])+'-'+bert["key"]
            #label.append(lab.encode('utf-8'))
            sents.append(bert["tokens"])
            n += 1
            if n_token > 0 and n >= n_token: break
            line = fi.readline()
    data = np.array(data)
    n_samples, n_features = data.shape
    #print data, ' '.join(label), n_samples, n_features
    #exit()

    return data, n_samples, n_features, sents, tags, ids

def plot_text(data, tags, ids):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    col_set = {'nature-1':'blue',
                'nature-2':'red',
                'naturaleza-1':'darkblue',
                'naturaleza-2':'darkred'}

    fig = plt.figure()
    ax = plt.subplot()
    reload(sys)
    sys.setdefaultencoding('utf-8')
    for i in range(data.shape[0]):
        #print (tags[i])
        if tags[i] not in col_set: continue
        color = col_set[tags[i]]
        #print (color)
        plt.text(data[i, 0], data[i, 1], str(ids[i]),
                color=color,
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show(fig)

def plot(data2, tags):
    #lexicon = {line.strip().split()[0]: line.strip().split()[1] for line in open('outputs/visualization/candidates.txt', 'r')}

    fig, ax = plt.subplots()
    patches = []
    flag = True
    if flag:
        col_set = (('nature-1', 'x', 'blue'),
                ('nature-2', '<', 'red'),
                ('naturaleza-1', '+', 'darkblue'),
                ('naturaleza-2', '>', 'darkred'))
    else:
        col_set = (('naturaleza-1 (before)', 'x', 'blue'),
                    ('naturaleza-2 (before)', '<', 'red'),
                    ('naturaleza-1 (after)', '+', 'darkblue'),
                    ('naturaleza-2 (after)', '>', 'darkred'))

    for target, marker, color in col_set:
        X = [x for tag, (x, y) in zip(tags, data2) if tag == target] 
        Y = [y for tag, (x, y) in zip(tags, data2) if tag == target]
        ax.scatter(X, Y, marker=marker, color=color)
        patches.append(mlines.Line2D([], [], color=color, marker=marker, label=target))
        
    #proto_words = ('Bush', 'Iraq', 'people', 'food', 'other', 'good', 'get', 'know')
    #for proto_word in proto_words:
    #    for word, (x, y) in zip(words, data2):
    #        if word == proto_word:
    #            proto_x, proto_y = x, y
    #            break
    #    plt.text(x, y, proto_word, bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc="cyan", alpha=0.6))

    plt.legend(handles=patches, loc = 'lower right')
    #fig.savefig("visualization.pdf")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input bert for tokens')
    parser.add_argument('--n_token', type=int, default=0, help='max tokens(0 to disable)')
    parser.add_argument('--output', type=str, default=None, help='output sentences')
    parser.add_argument('--label', action='store_true', default=False, help='use labels')
    parser.add_argument('--key', type=str, default=None, help='key word')
    parser.add_argument('--map', type=str, default=None, help='input map from word sense to POS')
    args = parser.parse_args()

    data, n_samples, n_features, sents, tags, ids = load_bert(args.input, args.n_token, args.map)
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
    title = 't-SNE embedding (time {:.2f}s)'.format(time() - t0)
    

    plot(result, tags)                  
    #plot_text(result, tags, ids)


if __name__ == '__main__':
    main()