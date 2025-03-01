import numpy as np
import faiss
import argparse
import sys

if len(sys.argv) < 3:
    print("Usage: python embed_compare_similitary.py <embedding_file1> <embedding_file2> [k]")
    sys.exit(1)

# Get file paths from command line arguments
file1_path = sys.argv[1]
file2_path = sys.argv[2]

# Get k (number of nearest neighbors) if provided, otherwise default to 5
k = 1
if len(sys.argv) > 3:
    k = int(sys.argv[3])

# Load the embeddings from the files
print(f"Loading embeddings from {file1_path} and {file2_path}")
file1_embeddings = np.load(file1_path)
file2_embeddings = np.load(file2_path)

# Ensure embeddings are float32 (FAISS requirement)
file1_embeddings = file1_embeddings.astype(np.float32)
file2_embeddings = file2_embeddings.astype(np.float32)

# Vector dimension
d = file1_embeddings.shape[1]

# Create FAISS index (L2 search)
index = faiss.IndexFlatL2(d)

# Add first file embeddings to the index
index.add(file1_embeddings)

# Search using second file embeddings
distances, indices = index.search(file2_embeddings, k)

# Print results
for i in range(len(file2_embeddings)):
    print(f"File2 Embedding {i} -> Closest File1 Embeddings: {indices[i]} with distances {distances[i]}")
