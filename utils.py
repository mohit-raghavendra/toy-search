import csv
import dash
import os
import rank_bm25 

import numpy as np

from collections import OrderedDict
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def get_ie_data():

	articles, links, titles = [], [], []

	with open('data_ie/indian_express_dataset_short.csv', 'r') as read_obj:
		csv_reader = csv.reader(read_obj)
		for row in csv_reader:
			articles.append(row[2])
			links.append(row[1])
			titles.append(row[0])

	tokens, avg_doc_len = [], 0
	with open('data_ie/processed_articles.csv', 'r') as read_obj:
		csv_reader = csv.reader(read_obj)
		for row in csv_reader:
			tokens.append(row)
			avg_doc_len += len(row)


	summaries = []
	with open('data_ie/summaries.csv', 'r') as read_obj:
		csv_reader = csv.reader(read_obj)
		for row in csv_reader:
			summaries.append(row[0])

	N = len(tokens)
	avg_doc_len /= N

	return articles, titles, tokens, links, summaries, N, avg_doc_len


def preprocess_query(query):
	query_terms = query.split(" ")  #Split the query -> tokenisation

	#Query normalisation
	query_terms = [term for term in query_terms if term.isalpha()]
	query_terms = [ps.stem(term) for term in query_terms]
	query_terms = [word for word in query_terms if not word in stopwords.words("english")]

	return query_terms

def calculate_BM25_package(query, tokens, bm25):

	query_terms = preprocess_query(query)
	scores = bm25.get_scores(query_terms)
	ranked_list = np.argsort(scores).tolist()
	ranked_list.reverse()
	return ranked_list

ps = PorterStemmer()
