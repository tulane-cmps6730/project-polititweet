##
## OpenAI Interface Layer
## 	Author: Leonardo Matone
## 
## This code contains functions used to query OpenAI's ChatGPT API with a list of categories and a categorization prompt.


import os
import re
import json
import difflib
import numpy as np
import pandas as pd
from textwrap import wrap
from openai import OpenAI
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

# Create new `pandas` methods which use `tqdm` progress
tqdm.pandas()

client = OpenAI()

def match_column(column, column_names):
	closest_matches = difflib.get_close_matches(column, column_names, n=1, cutoff=0)
	# If we have at least one match, return the first one. Otherwise, return None.
	if closest_matches:
		return closest_matches[0]
	return None


def clean(s):
	"""Remove alphanumeric characters"""
	pattern = re.compile('[\W_]+')
	return pattern.sub('', s)


def get_gpt_pred(categories, text, model="gpt-3.5-turbo", query=None):
	"""
	Calls ChatGPT API with given model and text, returns series with predictons.
	Different models interpret the formatting aspect of the prompt differently, which can lead to unexpected errors.
	Raises error when formatting of response is not interpretable.
	"""

	# formatting = 'Follow these instructions precisely. When provided with this input:\n {"content":content, "categories":categories} \
	#               determine which of the categories are discussed in the content, and always respond with a list of the following format\n ["category: 0 or 1" ...]'

	# query = f'"content":{text},"content":{str(categories)}'
	if query is None:
		query = "Which of these topics: {}\n\nAre discussed in this review: \"{}\". For all topics, respond in the format: \"topic\": 0 or 1.".format(categories, text)
	else:
		query = query.format(categories, text)

	completion = client.chat.completions.create(
		model=model,
		messages=[
			# {"role":"user", "content":formatting},
			{"role": "user", "content": query}
		],
		temperature=0.1,  # creativity/randomness (lower is more deterministic, less random)
		top_p=0.1  # top token propbability mass (lower is restrictive)
	)
	response = completion.choices[0].message.content

	model_designation = model.split('-')[1] + "_"

	if "\n" in response:
		predictions = re.sub("[{}\"-]", "", response).split("\n")
	elif ", " in response:
		predictions = re.sub("[{}\"-]", "", response).split(", ")
	else:
		print(f"ERROR: could not find delimiter using model: {model}, here's the text:\n {response}")
		return pd.Series({model_designation + "output": response})

	try:
		# create series from dict of model+category and the predicted value, initialize fields to zero for when model selectively outputs.
		series = {model_designation + category: 0 for category in categories}
		for prediction in predictions:
			if ': ' not in prediction:
				print("ERROR: response formatted incorrectly, no ':' in prediction (could be fluke)")
				continue
			column = model_designation + match_column(prediction.split(": ")[0], categories)
			series[column] = clean(prediction.split(": ")[1])
		series[model_designation + "output"] = predictions
	except Exception as e:
		print(
			f"ERROR: parsing failed. here's the exception:\n {e}\n\nhere's the text:\n {response}\n\nand here's the parsed text:\n {predictions}")
		return pd.Series({model_designation + "output": response})
	return pd.Series(series)


def predict_table(categories, text_series, model="gpt-3.5-turbo", query=None, log=None):
	"""
	Given a list of categories and a series of text, predict which categories are present in each document.
	:param categories: List of strings consisting of categories
	:param text_series: pd.Series containing text to be categorized
	:param model: (optional) specify which OpenAI model to use for predictions
	:param query: (optional) prompt given to model for each prediction
	:return: Dataframe of predictions
	"""
	print("Querying ChatGPT for categorization predictions", end="")
	if log:
		print(" |", log)
	else:
		print()

	predictions = text_series.progress_apply(lambda x: get_gpt_pred(categories, x, model, query))
	# TODO: automatically remove failures
	# TODO: automatically replace NaNs with 0s for non-fail entries
	return predictions
