

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
