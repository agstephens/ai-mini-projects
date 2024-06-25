# Using Retrieval Augmented Generation (RAG) to improve the integrity of LLM responses

Retrieval Augmented Generation (RAG) is an approach (***REF***) that allows you to exploit the capabilities of a Large Language Model (LLM) whilst focussing the subject area on a collection of your own (i.e. trusted) documents. 

In this example, OpenAI's (***which***) LLM is used to respond to queries about JASMIN. However, by using RAG, we can commit the JASMIN documentation (from GitHub) to a cloud-based Vector Database, and tell ChatGPT to use that information as its source when responding to queries.

This means that RAG allows you to:
 1. index and search your own data
 2. returning the most relevant results
 3. condition the LLM response with this relevant info

## The RAG Lifecycle

This example was adapted from the OuterBounds notebook at: (***DO I ASK THEM?***)

You can provide your own unstructured data from a range of sources. 

The RAG lifecycle works as follows:
 1. Chunk the unstructured data
 2. Compute embeddings on the chunks using a model
 3. Index the embeddings 
 4. Based on user queries, run vector similarity searches against the embeddings
 5. Return the top K most similar vectors
 6. Decode the vectors into the original data format
 7. Use the "similar" data in prompts (to the LLM)

## Chunking documents and pushing them to Vector Databases

There are many cloud-based Vector DB providers, and many allow you a minimum usage quota for free. In this example we use "pincone.ai". In order to run this example, you will need to:
 1. Create an account with pinecone.
 2. Get an API Key (for use in the code below).

The content of your own documentation is recorded into *chunks* which are uploaded into vector storage space.

## Using an OpenAI  Large Language Model

Any commercial LLM will require you to set up an account, probably pay something, and then get an API Key. In this instance, we are using OpenAI. You could also download your tools such as `ollama` and run your own instance of an LLM. It all depends on the cost vs effort considerations.

## Create embeddings

We want to find relevant data in our Pinecone index to condition the LLM on.

query = "How do I specify the memory when submitting a Slurm job to LOTUS?"

But first... a benchmark with the vanilla LLM

Embed and match our search vector

embed with sentence transformer
k = 5


