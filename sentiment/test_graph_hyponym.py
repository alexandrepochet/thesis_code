from nltk.corpus import wordnet as wn
import networkx as nx
import sentiment.wordnet as w
import matplotlib.pyplot as plt


def main():
    """
    Execute matching action for testing
    """

    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
    run = wn.synset('run.v.01')
    G = closure_graph(run, lambda s: s.hyp0nyms())
    index = nx.betweenness_centrality(G)
    plt.rc('figure', figsize=(12, 7))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, node_size=300, edge_color='b', alpha=1, width=1.5)
    plt.savefig(location + 'wordnet_graph.jpg', bbox_inches='tight')


if __name__ == '__main__':
    main()