# toy-search-engine

Toy Search Engine for 2500+ News Articles, sampled from [here](https://www.kaggle.com/pulkitkomal/news-article-data-set-from-indian-express). 

Try it out - [https://toy-search-engine.herokuapp.com/](https://toy-search-engine.herokuapp.com/)


* BM25 Okapi model is used to rank the articles. 

* The news summaries are obtained by generating summaries of the full stories using a DistilBART model, from [🤗 Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.SummarizationPipeline).  
