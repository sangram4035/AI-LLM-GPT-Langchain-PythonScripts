
from datetime import datetime
from flask import render_template
from FlaskWebProject1 import app
from flask import Flask,jsonify
import openai
from num2words import num2words
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
import time
import tiktoken
import os
import requests
import json
from azure.storage.blob import ContainerClient
import io
import operator
import fitz
import pandas as pd
from operator import itemgetter
import re
import openpyxl 
from reportlab.lib.styles import getSampleStyleSheet
# s is input text
def normalize_text(s, sep_token = " \n "):
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace("|","")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.replace("#","")
    s = s.strip()

    return s

#This method is used to create embeddings for all the string records present in the dataframe
def get_embedding(text, engine = 'text-embedding-ada-002'):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], engine=engine)['data'][0]['embedding']

#this method is used to form embedding for the user query and then compare the vectors using cosine similarity
def search_docs(df, user_query, top_n):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002"
    )
    df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res


#storing azure OpenAI keys and secrets
openai.api_type = "azure"
openai.api_key = "api key"
openai.api_base = "https://openaipocwestu.openai.azure.com/"
openai.api_version = "2022-12-01"

#reading the pdf and storing the data in JSON format into azure blob
df = pd.read_csv("Data/Manual_Intents.csv")




df_qnaPairs=df.copy()
df_qnaPairs['UTTERANCES'] = df["UTTERANCES"].apply(lambda x : normalize_text(x))
df_qnaPairs['UTTERANCES'] = df_qnaPairs['UTTERANCES'].apply(lambda x: ' '.join(x.split(maxsplit=2800)[:2800]))
tokenizer = tiktoken.get_encoding("cl100k_base")
df_qnaPairs['n_tokens'] = df_qnaPairs["UTTERANCES"].apply(lambda x: len(tokenizer.encode(x)))
df_qnaPairs['curie_search'] = df_qnaPairs["UTTERANCES"].apply(lambda x: get_embedding(x, engine = 'text-embedding-ada-002'))
    
output = df_qnaPairs.to_json(orient = 'records')
containerName = "dataframe"
blobName = "dataframe.json"

blob_block = ContainerClient.from_connection_string(
    conn_str="blob storage connection string",
    container_name=containerName
    )

blob_block.upload_blob(blobName, output, overwrite=True, encoding='cp1252')
#https://botdtatedata.blob.core.windows.net/dataframe/dataframe.csv

@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/getPreciseAnswer/<string:input>')
def getPreciseAnswer(input):
    url = "https://botdtatedata.blob.core.windows.net/dataframe/dataframe.json"
    df_qnaPairs = pd.read_json(url)
    res = search_docs(df_qnaPairs, input, top_n=7)
    context = '.'.join(res['Contents'])
    print(context)
    #combined_prompt = "Rephrase the below content in a better way:\n\n"+str(context)
    combined_prompt = 'You are a Intenet classification mode.Classify the Intent and entities from the data for user query. Output format: "{"intent":"Intent_name","entities":"entity_value","Score":confidence score}".If you dont know the answer return "{"intent":"None","Score":1.0}". User query :'+str(context)
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=combined_prompt,
    temperature=1,
    max_tokens=600,
    top_p=0.3,
    frequency_penalty=0,
    presence_penalty=0,
    best_of=1,
    stop=None)

    return jsonify(response)

if __name__  =="__main__":
    app.run(debug=True)



