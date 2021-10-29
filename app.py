import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import json
import os
import rank_bm25 
import math

import csv 
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import OrderedDict
from nltk.stem import PorterStemmer
import numpy as np


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

articles, links, titles = [], [], []

with open('data/articles.csv', 'r') as read_obj:
	csv_reader = csv.reader(read_obj)
	for row in csv_reader:
		articles.append(row[5])
		links.append(row[3])
		titles.append(row[4])

summarised_articles = []
with open('data/summarised_articles.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    for row in csv_reader:
        summarised_articles.append(row)

tokens, avg_doc_len = [], 0
with open('data/processed_articles.csv', 'r') as read_obj:
	csv_reader = csv.reader(read_obj)
	for row in csv_reader:
		tokens.append(row)
		avg_doc_len += len(row)

N = len(tokens)
avg_doc_len /= N


unique_docs = set()
u_articles, u_titles, u_links, u_tokens, u_summarised = [], [], [], [], []
for idx, title in enumerate(titles):
    if title not in unique_docs:
        unique_docs.add(title)
        u_articles.append(articles[idx])
        u_titles.append(titles[idx])
        u_links.append(links[idx])
        u_tokens.append(tokens[idx])
        u_summarised.append(summarised_articles[idx])
articles, titles, tokens, links, summaries = u_articles, u_titles, u_tokens, u_links, u_summarised


ps = PorterStemmer()

def preprocess_query(query):
	query_terms = query.split(" ")  #Split the query -> tokenisation

	#Query normalisation
	query_terms = [term for term in query_terms if term.isalpha()]
	query_terms = [ps.stem(term) for term in query_terms]
	query_terms = [word for word in query_terms if not word in stopwords.words("english")]

	return query_terms

print(preprocess_query("Machine Learning"))

def calculate_BM25(query):
	query_terms = preprocess_query(query)

	K1, b = 1, 0.75
	combination_factor, document_frequency = {}, {}
	for word in query_terms:
		combination_factor[word] = [0] * N
		document_frequency[word] = 0 
		for idx, article in enumerate(tokens):
			if word in article:
				document_frequency[word] += 1
			freq = article.count(word)
			num = (K1 + 1) * freq
			den = (K1 * ((1-b) + (b * (len(article) / avg_doc_len)))) + freq
			combination_factor[word][idx] = num / den


	scores = []
	for doc_id, document in enumerate(tokens):
		k = set(query_terms) & set(document)

		this_doc__score = 0
		for k_i in k:
			n_i = document_frequency[k_i]
			log_value = math.log(((N - n_i + 0.5) / (n_i + 0.5)), 2)
			this_doc__score += combination_factor[k_i][doc_id] * log_value
		scores.append(this_doc__score)

	ranked_list = np.argsort(scores).tolist()
	ranked_list.reverse()
	return ranked_list


bm25 = rank_bm25.BM25Okapi(tokens)
def calculate_BM25_package(query):

	query_terms = preprocess_query(query)
	scores = bm25.get_scores(query_terms)
	ranked_list = np.argsort(scores).tolist()
	ranked_list.reverse()
	return ranked_list

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([

	html.Div(
		children = [
					html.Div(children = "", style={'text-align': 'center'}, className='center'),
					], className = 'banner'),

	dcc.Tabs([
		dcc.Tab(label="Search Engine for ML articles on Medium", 
			children=[

					html.Div([

						html.Br(), html.Br(), html.Br(),

						html.Div(dcc.Input(id="input_query", type='text', placeholder="Enter query", style={'width': '50%'})), html.Br(),
						html.Button('Submit', id='submit_query', n_clicks=0), html.Br(), html.Br(), html.Br(),
						# html.Div(id="show_title", children=''),


						html.A('', id='show_link', target="_blank"), 
						html.Div(id='show_results', children=[]),
						
						], style={'align-content': 'center'}, className = "main_content")
					]),
			])
])

@app.callback(
	Output("show_results", "children"), 
	[Input("submit_query", "n_clicks")], 
	[State('input_query','value')]
)
def render_articles(n_clicks, query):
	if query:
		docs = calculate_BM25_package(query)[:15]

		return_divs = []
		for doc in docs:
			div = html.Div([
					html.A(titles[doc], id='show_link', target="_blank", href=links[doc]), html.Br(),
					html.P(summaries[doc][:200]), html.Br(),
					])

			return_divs.append(div)
		return return_divs

	
if __name__ == '__main__':
	app.run_server()
