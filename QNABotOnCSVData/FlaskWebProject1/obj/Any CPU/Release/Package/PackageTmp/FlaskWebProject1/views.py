"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from FlaskWebProject1 import app
from flask import Flask,jsonify
import openai
import requests
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


# s is input text
def normalize_text(s, sep_token = " \n "):
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.replace("#","")
    s = s.strip()

    return s

def get_answers(inputtext):
  url = "https://hrva-bot-languageservice-poc.cognitiveservices.azure.com/language/:query-knowledgebases?projectName=SampleProject-1&api-version=2021-10-01&deploymentName=production"

  payload = json.dumps({
  "top": 3,
  "question": inputtext,
  "includeUnstructuredSources": True
  })
  headers = {
  'Ocp-Apim-Subscription-Key': '576b5449278a4262973f741fe99d48fe',
  'Content-Type': 'application/json'
  }

  response = requests.request("POST", url, headers=headers, data=payload)

  return response.content


def get_embedding(text, engine = 'text-embedding-ada-002'):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], engine=engine)['data'][0]['embedding']


def search_docs(df, user_query, top_n=3):
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





openai.api_type = "azure"
openai.api_key = "56bc67a33a4642f1b1752235f1260f12"
openai.api_base = "https://openaipocwestu.openai.azure.com/"
openai.api_version = "2022-12-01"

df = pd.read_csv("Data/sampledataset.csv",encoding='cp1252')
df_qnaPairs=df.copy()
df_qnaPairs['Answer'] = df["Answer"].apply(lambda x : normalize_text(x))
df_qnaPairs['Answer'] = df_qnaPairs['Answer'].apply(lambda x: ' '.join(x.split(maxsplit=2800)[:2800]))
tokenizer = tiktoken.get_encoding("cl100k_base")
df_qnaPairs['n_tokens'] = df_qnaPairs["Answer"].apply(lambda x: len(tokenizer.encode(x)))
df_qnaPairs['curie_search'] = df_qnaPairs["Answer"].apply(lambda x: get_embedding(x, engine = 'text-embedding-ada-002'))
    
output = df_qnaPairs.to_json(orient = 'records')
containerName = "dataframe"
blobName = "dataframe.json"

blob_block = ContainerClient.from_connection_string(
    conn_str="DefaultEndpointsProtocol=https;AccountName=botdtatedata;AccountKey=iO9cK2WUd+vRqHewoPtV+6+xpVaAiVQ7SBTzfAx/hi8M0Ad86LXAqfMxjcXqWJamv6ascWdtKR1i+AStkbA7+g==;EndpointSuffix=core.windows.net",
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
    #df_qnaPairs = pd.read_csv(io.StringIO(dataframecontent.decode('cp1252')))
    res = search_docs(df_qnaPairs, input, top_n=1)
    #context= get_answers(input)
    context = res.Answer.values
    combined_prompt = "Find the answer from below data: "+"\nContext :"+str(context) +  "\n  Q :" +input +"A: "
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=combined_prompt,
    temperature=1,
    max_tokens=600,
    top_p=0.5,
    frequency_penalty=0,
    presence_penalty=0,
    best_of=1,
    stop=None)

    return jsonify(response)

if __name__  =="__main__":
    app.run(debug=True)



