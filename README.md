# RAG-using-Llama-2-Langchain-and-ChromaDB

Project Overview:
The project implements a Retrieval-Augmented Generation (RAG) model that combines LLaMA-2 for large language model-based generation with LangChain for building applications around language models and ChromaDB for efficient vector storage and similarity search. This setup enables the system to handle multi-modal knowledge retrieval effectively, particularly suited for applications that require natural language understanding and information retrieval from structured and unstructured sources.

RAG Framework: The RAG approach augments a language model's responses by retrieving relevant documents from an external knowledge base, thereby enhancing the quality of generated responses with specific, up-to-date information.
LLaMA-2 Integration: LLaMA-2, a high-performing large language model, powers the generation component, providing advanced natural language responses based on the context retrieved by ChromaDB.
LangChain: This framework allows chaining of LLM tasks and provides an efficient pipeline to manage input/output data flow, model chaining, and retrieval steps.
ChromaDB: Utilized as a vector database, ChromaDB stores document embeddings, allowing fast similarity searches to retrieve contextually relevant information, which is passed to LLaMA-2 for response generation.

Objective¶
Use Llama 2.0, Langchain and ChromaDB to create a Retrieval Augmented Generation (RAG) system. This will allow us to ask questions about our documents (that were not included in the training data), without fine-tunning the Large Language Model (LLM). When using RAG, if you are given a question, you first do a retrieval step to fetch any relevant documents from a special database, a vector database where these documents were indexed.

Definitions
LLM - Large Language Model
Llama 2.0 - LLM from Meta
Langchain - a framework designed to simplify the creation of applications using LLMs
Vector database - a database that organizes data through high-dimmensional vectors
ChromaDB - vector database
RAG - Retrieval Augmented Generation (see below more details about RAGs)
Model details
Model: Llama 2
Variation: 7b-chat-hf (7b: 7B dimm. hf: HuggingFace build)
Version: V1
Framework: PyTorch
LlaMA 2 model is pretrained and fine-tuned with 2 Trillion tokens and 7 to 70 Billion parameters which makes it one of the powerful open source models. It is a highly improvement over LlaMA 1 model.

What is a Retrieval Augmented Generation (RAG) system?
Large Language Models (LLMs) has proven their ability to understand context and provide accurate answers to various NLP tasks, including summarization, Q&A, when prompted. While being able to provide very good answers to questions about information that they were trained with, they tend to hallucinate when the topic is about information that they do "not know", i.e. was not included in their training data. Retrieval Augmented Generation combines external resources with LLMs. The main two components of a RAG are therefore a retriever and a generator.

The retriever part can be described as a system that is able to encode our data so that can be easily retrieved the relevant parts of it upon queriying it. The encoding is done using text embeddings, i.e. a model trained to create a vector representation of the information. The best option for implementing a retriever is a vector database. As vector database, there are multiple options, both open source or commercial products. Few examples are ChromaDB, Mevius, FAISS, Pinecone, Weaviate. Our option in this Notebook will be a local instance of ChromaDB (persistent).

For the generator part, the obvious option is a LLM. In this Notebook we will use a quantized LLaMA v2 model, from the Kaggle Models collection.

The orchestration of the retriever and generator will be done using Langchain. A specialized function from Langchain allows us to create the receiver-generator in one line of code.

Conclusions:
We used Langchain, ChromaDB and Llama 2 as a LLM to build a Retrieval Augmented Generation solution. For testing, we were using the latest State of the Union address from Jan 2023.
