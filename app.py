import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State

from utils import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

articles_ie, titles_ie, tokens_ie, links_ie, summaries_ie, N_ie, avg_doc_len_ie = get_ie_data()
bm25_ie = rank_bm25.BM25Okapi(tokens_ie)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([

	html.Div(
		children = [
					html.Div(children = "", style={'text-align': 'center'}, className='center'),
					], className = 'banner'),

	dcc.Tabs([
		dcc.Tab(label="News search - Search engine for Indian Express news articles", 
			children=[

					html.Div([

						html.Br(), html.Br(), html.Br(),

						html.Div(dcc.Input(id="input_query_ie", type='text', placeholder="Enter query Ex: 'Car accident', 'New Delhi'", style={'width': '50%'})), html.Br(),
						html.Button('Submit', id='submit_query_ie', n_clicks=0), html.Br(), html.Br(), html.Br(),

						html.A('', id='show_link_ie', target="_blank"), 
						html.Div(id='show_results_ie', children=[]),
						
						], style={'align-content': 'center', 'width': '50%', 'padding-left':'7%'}, className = "main_content")
					]),
			])
])

@app.callback(
	Output("show_results_ie", "children"), 
	[Input("submit_query_ie", "n_clicks")], 
	[State('input_query_ie','value')]
)
def render_articles(n_clicks, query):
	if query:
		docs = calculate_BM25_package(query, tokens_ie, bm25_ie)[:15]

		return_divs = []
		for doc in docs:
			div = html.Div([
					html.A(titles_ie[doc], id='show_link_ie', target="_blank", href=links_ie[doc]), html.Br(),
					html.P(summaries_ie[doc], style={'text-align': 'justify', 'text-justify': 'inter-word'}), html.Br(),
					])

			return_divs.append(div)
		return return_divs

	
if __name__ == '__main__':
	app.run_server()
