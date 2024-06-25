import os, time, sys

import pandas as pd

from sentence_transformers import SentenceTransformer

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

from pinecone import Pinecone, ServerlessSpec

print("WARNING***************")
print("This includes API Keys!!!")
print("DO PIP INSTALL: pip install langchain_community")

print("This lives in the following workspace...")
print("""
source /home/workspace/mambaforge/bin/activate llm-rag

""")

req_keys = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
for req_key in req_keys:
    if req_key not in os.environ:
	    raise KeyError(f"Missing env var: {req_key}")
		


pinecone_api_key = os.environ["PINECONE_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

print("NOTE: Your ChatGPT paid account is NOT THE SAME as an OpenAI account!")

pc = Pinecone(api_key=pinecone_api_key)

metric = "cosine"

class AbstractEmbedder:
    def __init__(self, **kwargs):
       pass

class OpenAIEmbedder(AbstractEmbedder):

    def __init__(self):
        super().__init__()

    def embed(self, sentences):
        pass


class SentenceTransformerEmbedder(AbstractEmbedder):
    def __init__(self, model_path, device="cpu"):
        super().__init__()

        self.model = SentenceTransformer(model_path)
        self.model.to(device)

    def embed(self, sentences):
        return self.model.encode(sentences, show_progress_bar=True)




text_sample = [
    "Hello Gen AI friends!",
    "Metaflow helps you build production machine learning workflows",
    "Lots of people recognize machine learning systems require robust workflows",
]

# https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
embedding_model = "paraphrase-MiniLM-L6-v2"

encoder = SentenceTransformerEmbedder(embedding_model, device="cpu")
embedding = encoder.embed(text_sample)

dimension = embedding.shape[1]


df_csv = "jasmin_docs.csv"


if os.path.isfile(df_csv):
    print(f"Loading: {df_csv}")
    df = pd.read_csv(df_csv)
else:
    print("Need to add in code here to parse JASMIN repo...later.")
    df.to_csv(df_csv, index=False)
    print(f"Wrote: {df_csv}")

print("Downloaded repo:", df.head())

word_count_threshold = 10
char_count_threshold = 25

# Filter out rows with less than N words or  M chars.
df = df[df.word_count > word_count_threshold]
df = df[df.char_count > char_count_threshold]

df


# Instantiate an encoder
encoder = SentenceTransformerEmbedder(embedding_model, device="cpu")

# Fetch docs from dataframe
docs = df.contents.tolist()

# Encode documents
embeddings = encoder.embed(docs)  # takes ~30-45 seconds on average in sandbox instance
dimension = len(embeddings[0])
print("Dimension is %s" % dimension)

pc = Pinecone(api_key=pinecone_api_key)

index_name = "jasmin-documentation"
metric = "cosine"  # https://docs.pinecone.io/docs/indexes#distance-metrics

try:
    pc.delete_index(    
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Deleted: {index_name}")
except Exception as exc:
    print("No index found - so didn't delete.")


if index_name not in pc.list_indexes().names():
    # https://docs.pinecone.io/reference/create_index
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index {index_name} already exists")

ids = df.index.values

# connect to the index
index = pc.Index(index_name)

vectors = [
    {
        "id": str(idx),
        "values": emb.tolist(),
        "metadata": {"text": txt},
    }
    for idx, (txt, emb) in enumerate(zip(docs, embeddings))
]
upsert_response = index.upsert(vectors=vectors)

print("search with a query")


args = sys.argv
if len(args) > 1:
    query = args[1]
else:
    query = "How do I specify the memory when submitting a Slurm job to LOTUS?"

print(f"\n\nTesting this query:\n\t'{query}'.")

# ### But first... a benchmark with the vanilla LLM

# %%

# %%
human_template = "{user_query}"
chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
chat = ChatOpenAI(openai_api_key=openai_api_key)
response = chat(chat_prompt.format_messages(user_query=query))
print("response:", response.content)

# embed with sentence transformer
k = 5

vector = encoder.embed([query])[0]
matches = index.query(vector=vector.tolist(), top_k=k, include_metadata=True)
matches = matches.to_dict()["matches"]

row_idxs = []
for m in matches:
    row_idxs.append(int(m["id"]))

row_idxs

retrieved_results = df.iloc[row_idxs, :]
retrieved_results
print("\n\n\nHERE IS THE RETRIEVED RESULTS:\n\n", retrieved_results)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

system_message = """
You are a helpful assistant that helps scientists to use the JASMIN platform for their data workflows.

Here is some relevant context you can use, each with links to a page in the JASMIN documentation where the context is retrieved from:
"""

context_template = """
{system_message}

{context}

Use the above pieces of context to condition the response.
"""

_context = ""
for _, row in retrieved_results.iterrows():
    _context += "\n### context: {}\n### url: {} \n".format(row.contents, row.page_url)

human_template = "{user_query}"

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_template),
        ("human", human_template),
    ]
)

chat = ChatOpenAI(openai_api_key=openai_api_key)

response = chat(
    chat_prompt.format_messages(
        user_query=query, context=_context, system_message=system_message
    )
)

print("\n\n\nHERE IS THE response.content:\n\n", response.content, "\n\n\n")

