from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from retrieval import retrieve

# laoding sQuad dataset - validation split has answerable and unanswerable questions
dataset = load_dataset("squad_v2",split="validation")



# document collection - context of the questions 
contexts=list(set(dataset['context']))
# set dtype used to dedupliacte contects to build our reterival corpus
print(f"Number of unique contexts: {len(contexts)}")

model=SentenceTransformer('all-MiniLM-L6-v2',device='cpu')

# embed all passages in the context collection
embeddings=model.encode(
    contexts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True     #makes inner product equivalent to cosine similarity.
)

embeddings=np.array(embeddings).astype("float32")

# building FAISS index for retrieval
dimension=embeddings.shape[1]   
index=faiss.IndexFlatIP(dimension)  # inner product index for cosine similarity
index.add(embeddings)  


print(f"Number of passages in the index: {index.ntotal} vectors of dimension {dimension}")

# saving everything so later to  not recompute 
faiss.write_index(index,"corpus_index.faiss")
with open("contexts.pkl", "wb") as f:
    pickle.dump(contexts, f)

# Pull a real question + confirm its context is in our corpus
sample = dataset[0]  # dataset from build_index.py — reuse it, don't reload
print("Sample question:", sample["question"])
print("Is its context in our corpus?", sample["context"] in contexts)