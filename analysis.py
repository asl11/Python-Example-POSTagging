import math
import random
import numpy
from collections import *
import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy
#################   BASIC PLOTTING FUNCTIONS BELOW   #################
#################  SKIP TO LINE 288 FOR FORMAL CODE  #################
def show():
    """
	Do not use this function unless you have trouble with figures.

	It may be necessary to call this function after drawing/plotting
	all figures.  If so, it should only be called once at the end.

	Arguments:
	None

	Returns:
	None
	"""
    plt.show()


def plot_dist_linear(data, title, xlabel, ylabel, filename=None):
    """
	Plot the distribution provided in data as a bar plot on a linear
	scale.

	Arguments:
	data     -- dictionary which will be plotted with the keys
				on the x axis and the values on the y axis
	title    -- title label for the plot
	xlabel   -- x axis label for the plot
	ylabel   -- y axis label for the plot
	filename -- optional name of file to which plot will be
				saved (in png format)

	Returns:
	None
	"""
    _plot_dist(data, title, xlabel, ylabel, False, filename)


def plot_dist_loglog(data, title, xlabel, ylabel, filename=None):
    """
	Plot the distribution provided in data as a scatter plot on a
	loglog scale.

	Arguments:
	data     -- dictionary which will be plotted with the keys
				on the x axis and the values on the y axis
	title    -- title label for the plot
	xlabel   -- x axis label for the plot
	ylabel   -- y axis label for the plot
	filename -- optional name of file to which plot will be
				saved (in png format)

	Returns:
	None
	"""
    _plot_dist(data, title, xlabel, ylabel, True, filename)


def _pow_10_round(n, up=True):
    """
	Round n to the nearest power of 10.

	Arguments:
	n  -- number to round
	up -- round up if True, down if False

	Returns:
	rounded number
	"""
    if up:
        return 10 ** math.ceil(math.log(n, 10))
    else:
        return 10 ** math.floor(math.log(n, 10))


def _plot_dist(data, title, xlabel, ylabel, scatter, filename=None):
    """
	Plot the distribution provided in data.

	Arguments:
	data     -- dictionary which will be plotted with the keys
				on the x axis and the values on the y axis
	title    -- title label for the plot
	xlabel   -- x axis label for the plot
	ylabel   -- y axis label for the plot
	scatter  -- True for loglog scatter plot, False for linear bar plot
	filename -- optional name of file to which plot will be
				saved (in png format)

	Returns:
	None
	"""
    ### Check that the data is a dictionary
    if not isinstance(data, types.DictType):
        msg = "data must be a dictionary, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if scatter:
        _plot_dict_scatter(data)
    else:
        _plot_dict_bar(data, 0)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid
    gca = pylab.gca()
    gca.yaxis.grid(True)
    gca.xaxis.grid(False)

    if scatter:
        ### Use loglog scale
        gca.set_xscale('log')
        gca.set_yscale('log')
        gca.set_xlim([_pow_10_round(min([x for x in data.keys() if x > 0]), False),
                      _pow_10_round(max(data.keys()))])
        gca.set_ylim([_pow_10_round(min([x for x in data.values() if x > 0]), False),
                      _pow_10_round(max(data.values()))])

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)


def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
	Plot a line graph with the provided data.

	Arguments:
	data     -- a list of dictionaries, each of which will be plotted
				as a line with the keys on the x axis and the values on
				the y axis.
	title    -- title label for the plot
	xlabel   -- x axis label for the plot
	ylabel   -- y axis label for the plot
	labels   -- optional list of strings that will be used for a legend
				this list must correspond to the data list
	filename -- optional name of file to which plot will be
				saved (in png format)

	Returns:
	None
	"""
    ### Check that the data is a list
    if not isinstance(data, types.ListType):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for i in range(len(data) - len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)


def _dict2lists(data):
    """
	Convert a dictionary into a list of keys and values, sorted by
	key.

	Arguments:
	data -- dictionary

	Returns:
	A tuple of two lists: the first is the keys, the second is the values
	"""
    xvals = data.keys()
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals


def _plot_dict_line(d, label=None):
    """
	Plot data in the dictionary d on the current plot as a line.

	Arguments:
	d     -- dictionary
	label -- optional legend label

	Returns:
	None
	"""
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)


def _plot_dict_bar(d, xmin=None, label=None):
    """
	Plot data in the dictionary d on the current plot as bars.

	Arguments:
	d     -- dictionary
	xmin  -- optional minimum value for x axis
	label -- optional legend label

	Returns:
	None
	"""
    xvals, yvals = _dict2lists(d)
    if xmin == None:
        xmin = min(xvals) - 1
    else:
        xmin = min(xmin, min(xvals) - 1)
    if label:
        pylab.bar(xvals, yvals, align='center', label=label)
        pylab.xlim([xmin, max(xvals) + 1])
    else:
        pylab.bar(xvals, yvals, align='center')
        pylab.xlim([xmin, max(xvals) + 1])


def _plot_dict_scatter(d):
    """
	Plot data in the dictionary d on the current plot as points.

	Arguments:
	d     -- dictionary

	Returns:
	None
	"""
    xvals, yvals = _dict2lists(d)
    pylab.scatter(xvals, yvals)


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d
#####################  FORMAL CODE BEGINS  #####################
class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

def read_file(filename):
    #file_representation = []
    """
    Separate read_file function for test_data
    :param filename: the filename
    :return: list of words that have spaces in between
    Justparsed through one line because there only is one line in the file
    """


    f = open(str(filename), "r")
    for line in f:

        word = line.split(" ")
        return word
    #return file_representation

def update_hmm(hmm, u_words, u_tags, testdata):
    """

    :param hmm: HMM
    :param u_words: list of unique words
    :param u_tags: list of unique tags
    :param testdata: test data is a list of STRINGS, not pair(string, tag)
    :return: an updated hmm
    """
    ep = 0.00001
    wordstotal = set(u_words)
    e_mat = hmm.emission_matrix

    for word in testdata:
        wordstotal.add(word)
    for tag in u_tags:
        for word in wordstotal:
            e_mat[tag][word] += ep


    #normalizes now
    for tag in u_tags:
        sum = 0
        for value in e_mat[tag].values():
            sum += value
        for w, val in e_mat[tag].items():
            e_mat[tag][w] = 1.0 * val / sum
    hmm.emission_matrix = e_mat
    return hmm

def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags
def read_pos_file_percent(filename, percent, numtotal):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    stop_line = int(percent * numtotal) # NEW CODE HERE , WAY TO ONLY READ A CERTAIN PERCENT OF TOTAL FILE
    count = 0 #count stops the line once it reaches stop_line
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        if count > stop_line: #checks if the line has reached a certain total
            f.close()
            return file_representation, unique_words, unique_tags

        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
        count += 1
    f.close()
    return file_representation, unique_words, unique_tags

def compute_counts(training_data, order):
    """

    :param training_data: a return of the read_pos_file - a list of tuples in the form (word, POS)
    :param order: the order of the HMM model
    :return: returns a tuple in the form (numtokens, dict of words, dict of POS): words counts the the number of words per type per unique word
    dict of POS changes nests depending on the order
    """
    t_words = defaultdict(lambda:defaultdict(int))
    c1 = defaultdict(int)
    c2 = defaultdict(lambda:defaultdict(int))
    c3 = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))

    for index, (word,tag) in enumerate(training_data):
        t_words[tag][word] += 1
        c1[tag] += 1

        if index > 0 and training_data[index-1][1] != ".":
            c2[training_data[index-1][1]][training_data[index][1]] += 1

        if index > 1 and (training_data[index-2][1] != "." and training_data[index-1][1] != "."):
            c3[training_data[index-2][1]][training_data[index-1][1]][training_data[index][1]] += 1

    if order == 2:
        return len(training_data), t_words, c1, c2

    return len(training_data), t_words, c1, c2, c3



training_data = [('hw7','N'), ('is','V'),('difficult','A'),('.','.')]
unique_tags = ['N', 'V', 'A', '.']
num_tokens = 4
_C1 = {'A': 1, '.': 1, 'V': 1, 'N': 1}
_C2 = {'A': {'.': 1}, 'V': {'A': 1}, 'N': {'V': 1}}
_C3 = {'V': {'A': {'.': 1}}, 'N': {'V': {'A': 1}}}

#print compute_counts(training_data,3)

def compute_initial_distribution(training_data, order):
    """

    :param training_data: result of read_pos_file
    :param order: order of HMM
    :return: returns the initial probability distribution given the order
    """
    result = defaultdict(int)

    if order == 2:

        result[training_data[0][1]] += 1
        for index in range(len(training_data)):
            if (training_data[index][1] == '.' or training_data[index][1] == '?' or training_data[index][1] == '!') and (index < len(training_data)-1):
                result[training_data[index+1][1]] += 1

        total = 0
        for num in result.values():
            total += num

        for word, count in result.items():
            result[word] = 1.0 * count / total

    if order == 3:
        result = defaultdict(lambda: defaultdict(int))
        result[training_data[0][1]][training_data[1][1]] += 1
        for index in range(len(training_data)):
            if (training_data[index][1] == '.') and index < (len(training_data) - 2):
                result[training_data[index+1][1]][training_data[index+2][1]] += 1

        total = 0
        for num1, dict2 in result.items():
            for num2 in dict2.values():
                total += num2
        for num, dict1 in result.items():
            for word, count in dict1.items():
                result[num][word] = 1.0 * count / total



    return result

#print compute_initial_distribution(training_data, 2)
def compute_emission_probabilities(unique_words, unique_tags, W, C):
    """

    :param unique_words: list of unique words
    :param unique_tags: list of unique tags
    :param W: returned by compute counts
    :param C: returned by compute counts
    :return: counts the emission probabilities
    """

    e_matrix = defaultdict(lambda: defaultdict(int))
    for type, words in W.items():
        for word, count in words.items():
            e_matrix[type][word] = 1.0 * count / C[type]
    return e_matrix
def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
    """
    Compute the lambdas for smoothing following the equation in the explanation document
    :param unique_tags: the list of unique tags in the training tags
    :param num_tokens: an int number of tokens in the training data
    :param C1: Counts for each tag
    :param C2: Counts for each tag -> tag chain, represented as nested dictionary {tag:{tag:count}}
    :param C3: Counts for each tag -> tag -> chain, represented as nested dictionary {tag:{tag:{tag:count}}}
    :param order: order of the markov model
    :return: returns a list [a,b,c] of the lambdas 1,2, and 3, respectively
    """
    lams = [0.0,0.0,0.0]
    #_C2 = {'A': {'.': 1}, 'V': {'A': 1}, 'N': {'V': 1}}
    # _C3 = {'V': {'A': {'.': 1}}, 'N': {'V': {'A': 1}}}
    if order == 2:
        for tag1 in unique_tags:
            for tag2 in unique_tags:
                if C2[tag1][tag2] > 0:
                    a0 = (C1[tag2] - 1.0)/ num_tokens

                    if (C1[tag1] - 1.0) > 0:
                        a1 = (C2[tag1][tag2] - 1.0) / (C1[tag1] - 1.0)
                    else:
                        a1 = 0

                    if max(a0,a1) == a0:
                        lams[0] += C2[tag1][tag2]
                    else:
                        lams[1] += C2[tag1][tag2]

    if order == 3:
        for tag1, dict1 in C3.items():
            for tag2, dict2 in dict1.items():
                for tag3, count in dict2.items():
                    a0 = (C1[tag3] - 1.0) / num_tokens

                    if (C1[tag2] - 1.0) > 0:  # checking for division by 0
                        a1 = (C2[tag2][tag3] - 1.0) / (C1[tag2] - 1.0)
                    else:
                        a1 = 0

                    if (C2[tag3][tag2] - 1.0) > 0:
                        a2 = (C3[tag1][tag2][tag3] - 1.0) / (C2[tag3][tag2] - 1.0)
                    else:
                        a2 = 0

                    if max(a0, a1, a2) == a0:
                        lams[0] += C3[tag1][tag2][tag3]
                    elif max(a0, a1, a2) == a1:
                        lams[1] += C3[tag1][tag2][tag3]
                    else:
                        lams[2] += C3[tag1][tag2][tag3]


    sum_lam = sum(lams)
    result = []
    for lam in lams:
        lam = lam / sum_lam
        result.append(lam)

    return result



#print compute_lambdas(unique_tags, num_tokens, _C1, _C2, _C3, 2)
def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
    """

    :param training_data: the training data
    :param unique_tags: the unique tags in the traning data
    :param unique_words: the unique words in the training data
    :param order: the order of the hmm
    :param use_smoothing: boolean, represents whether or not it takes smoothing
    :return:
    """
    if order == 2:
        num_tokens, tag_word, C1, C2 = compute_counts(training_data,2)
        initial_dist = compute_initial_distribution(training_data,2)
        emission_mat = compute_emission_probabilities(unique_words,unique_tags,tag_word,C1)
        transition_mat = defaultdict(lambda:defaultdict(int))
        if use_smoothing:
            lams = compute_lambdas(unique_tags,num_tokens,C1,C2,{},order)
        else:
            lams = [0.0,1.0,0.0]

        for tag1 in unique_tags:
            for tag2 in unique_tags:
                transition_mat[tag1][tag2] = lams[1] * C2[tag1][tag2] / C1[tag1] + lams[0] * C1[tag2] / num_tokens

    if order == 3:
        num_tokens, tag_word, C1, C2, C3 = compute_counts(training_data,3)

        initial_dist = compute_initial_distribution(training_data, 3)
        emission_mat = compute_emission_probabilities(unique_words, unique_tags, tag_word, C1)
        transition_mat = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
        if use_smoothing:
            lams = compute_lambdas(unique_tags, num_tokens, C1,C2,C3,order)
            #print lams, "lams"
        else:
            lams = [0.0,0.0,1.0]
        num_tokens = float(num_tokens)

        for tag1, dict1 in C3.items():
            for tag2, dict2 in dict1.items():
                for tag3, count in dict2.items():
                    part_1 = 0
                    part_2 = 0
                    part_3 = 0
                    if C2[tag1][tag2] != 0:
                        part_1 = lams[2] * count / C2[tag1][tag2]
                    if C1[tag2] != 0:
                        part_2 = lams[1] * C2[tag2][tag3] / C1[tag2]
                    if num_tokens != 0 :
                        part_3 = lams[0] * C1[tag3] / num_tokens
                    transition_mat[tag1][tag2][tag3] = (part_1
                                                        + (part_2)
                                                        + (part_3))


    return HMM(order, initial_dist, emission_mat, transition_mat)

"""
_trigram_initial_distribution = {'Coin1': {'Coin1': .25, 'Coin2': .25}, 'Coin2': {'Coin1': .25, 'Coin2': .25}}
_trigram_emission_probabilities = {'Coin1': {'Heads': .9, 'Tails': .1}, 'Coin2': {'Heads': .5, 'Tails': .5}}
_trigram_transition_matrix = {'Coin1': {'Coin1': {'Coin1': .5, 'Coin2': .5}, 'Coin2': {'Coin1': .5, 'Coin2': .5}}, 'Coin2': {'Coin1': {'Coin1': .5, 'Coin2': .5}, 'Coin2': {'Coin1': .5, 'Coin2': .5}}}

hmm = HMM(3, _trigram_initial_distribution, _trigram_emission_probabilities, _trigram_transition_matrix)
sentence = ['Heads', 'Heads', 'Tails']
"""

def trigram_viterbi(hmm, sentence):
    """
        Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
        Inputs:
          hmm --- the HMM to use to predict the POS of the words in the sentence.
          sentence ---  a list of words.
        Returns:
          A list of tuples where each tuple contains a word in the
          sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))


    for tag1 in unique_tags:
        for tag2 in unique_tags:

            if (hmm.initial_distribution[tag1][tag2] != 0) and (hmm.emission_matrix[tag1][sentence[0]] != 0) and (hmm.emission_matrix[tag2][sentence[1]] != 0):
                viterbi[tag1][tag2][1] = math.log(hmm.initial_distribution[tag1][tag2]) + math.log(hmm.emission_matrix[tag1][sentence[0]]) + math.log(hmm.emission_matrix[tag2][sentence[1]])
            else:
                viterbi[tag1][tag2][1] = -1 * float('inf')



    # Dynamic programming.
    for t in range(2, len(sentence)):
        backpointer["No_Path"]["No_Path"][t] = "No_Path"
        for s in unique_tags:
            backpointer[s]["No_Path"][t] = "No_Path"
            backpointer["No_Path"][s][t] = "No_Path"
            for s2 in unique_tags:
                max_value = -1 * float('inf')
                max_state = None
                for s_prime in unique_tags:
                    val1 = viterbi[s_prime][s][t - 1]
                    val2 = -1 * float('inf')
                    if hmm.transition_matrix[s_prime][s][s2] != 0:
                        val2 = math.log(hmm.transition_matrix[s_prime][s][s2])
                    curr_value = val1 + val2
                    if curr_value > max_value:
                        max_value = curr_value
                        max_state = s_prime
                val3 = -1 * float('inf')
                if hmm.emission_matrix[s2][sentence[t]] != 0:
                    val3 = math.log(hmm.emission_matrix[s2][sentence[t]])
                viterbi[s][s2][t] = max_value + val3
                if max_state == None:

                    backpointer[s][s2][t] = "No_Path"
                else:
                    backpointer[s][s2][t] = max_state


    """
    for ut in unique_tags:
        for ut2 in unique_tags:
            string = ""
            for i in range(0, len(sentence)):
                if (viterbi[ut][ut2][i] != float("-inf")):
                    string += str(int(viterbi[ut][ut2][i])) + "\t"
                else:
                    string += str(viterbi[ut][ut2][i]) + "\t"
            print string
    """
    # Termination
    max_value = -1 * float('inf')
    last_state = None
    second_last = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        for s_final in unique_tags:
            if viterbi[s_prime][s_final][final_time] > max_value:
                max_value = viterbi[s_prime][s_final][final_time]
                last_state = s_final
                second_last = s_prime
    if last_state == None:
        last_state = "No_Path"
    if second_last == None:
        second_last = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence) - 1], last_state))
    tagged_sentence.append((sentence[len(sentence)-2], second_last))

    for i in range(len(sentence) - 3, -1, -1):
        next_tag = tagged_sentence[-1][1]
        second_last_tag = tagged_sentence[-2][1]
        curr_tag = backpointer[next_tag][second_last_tag][i + 2]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence


#print trigram_viterbi(hmm, sentence)

def checkacc(test, actual):
    """
    Checks the accuracy of the test
    :param test: The output of the test_data function in the form of [(word,tag),...]
    :param actual: The actual tags of the test data
    :return: percent of the correctly tagged words in test versus actual
    """
    num_correct = 0
    for index in range(min(len(test),len(actual))):
        if (test[index][1] == actual[index][1]) and (test[index][0] == actual[index][0]):
            num_correct += 1

    return num_correct * 1.0 / len(test)

def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])

        else:
            viterbi[tag][0] = -1 * float('inf')


    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2

                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])

            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"
    #print string
    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"
    #print viterbi
    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence


"""
Important information for testing
"""
num_lines = sum(1 for line in open('training.txt'))
sentence = read_file('testdata_untagged.txt')
actual, act_uqw, act_uqt = read_pos_file('testdata_tagged.txt')
#sentence = ['The', 'New','Deal', 'was', 'a', 'series', 'of', 'domestic', 'programs','enacted', 'in', 'the', 'United', 'States', 'between']



def test_data(percent, num_lines, sentence, actual, training_file, order, use_smooth):
    """

    :param percent: percent of the data to use for training
    :param num_lines: number of total lines in the training data
    :param sentence: the sentence to be tagged
    :param actual: the actual tagged sentence to compare
    :param training_file: training data file name
    :param order: order of the model to use
    :param use_smooth: boolean value whether or not smoothing
    :return: returns a
    """
    training, uqw, uqt = read_pos_file_percent(training_file, percent, num_lines)
    hmm = build_hmm(training, uqt, uqw, order, use_smooth )

    hmm = update_hmm(hmm, uqw, uqt, sentence)

    tagged = []
    period = -1
    for index in range(len(sentence)):
        if sentence[index] == ".":
            if order == 2:
                for pair in bigram_viterbi(hmm, sentence[period+1:index+1]):

                    tagged.append(pair)


            if order == 3:
                for pair in trigram_viterbi(hmm, sentence[period+1:index+1]):
                    tagged.append(pair)
            period = index

    return checkacc(tagged, actual)

#print test_data(0.01,num_lines,sentence,actual,'training.txt',3,False)

def experiment(percents, order, smoothing, sentence, actual, training_file, num_lines):
    """
    Uses the test_data function and runs it many times
    :param percents: list of percents of the total data we want to run on
    :param order: order of the model
    :param smoothing: boolean for smoothing or no
    :param sentence: the input "sentence" is actually the whole input, we parse through sentence by sentence
    :param actual: the actual result, ordered [(word, tag)]
    :param training_file: training file name
    :param num_lines: number of lines in the training file so that we can control percentages
    :return: returns a dictionary of {percent:accuracy} so that it can be plotted
    """
    results = {}
    for percent in percents:
        print "=======TESTING ON ", percent, "with smoothing" if smoothing else ", NO SMOOTHING,", "TRIGRAM" if order == 3 else "BIGRAM", "==========="
        data = test_data(percent, num_lines, sentence, actual, training_file, order, smoothing)
        results[percent] = data
        print data

    print "=======FINISHED, TOTAL RESULTS FOR", "TRIGRAM" if order == 3 else "BIGRAM", "========="

    return results

"""
TESTS BELOW 
"""
percents = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
data = [experiment(percents, 2, False, sentence, actual, 'training.txt', num_lines),
        experiment(percents, 2, True, sentence, actual, 'training.txt', num_lines),
        experiment(percents, 3, False, sentence, actual, 'training.txt', num_lines),
        experiment(percents, 3, True, sentence, actual, 'training.txt', num_lines),]

plot_lines(data,"Accuracy of HMM models at percents of training corpus", "Percent of Training Corpus", "Accuracy Percent",
           ["Bigram No Smoothing", "Bigram with Smoothing","Trigram no Smoothing", "Trigram with Smoothing"], "graph.png")

show()
