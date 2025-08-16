# Reformer---Efficient-Text-Generation---Pushing-the-Limits-of-Language-Modeling

This project demonstrates the Reformer model (Kitaev, Kaiser, Levskaya) — a Transformer variant optimized for long-sequence modeling with reduced memory consumption.

We train Reformer on Crime and Punishment (a novel with over 500,000 tokens) to highlight its efficiency and scalability compared to standard Transformers.

📌 Problem

Traditional Transformers suffer from quadratic time and memory complexity with sequence length, making them impractical for extremely long texts (books, documents, etc.).

🚀 Methodology

Reformer introduces several innovations to address these limitations:

⚡ LSH Attention – Approximates attention using locality-sensitive hashing, reducing complexity from O(L²) → O(L log L).

♻ Reversible Residual Layers – Saves GPU memory by recomputing activations during backpropagation.

🧩 Axial Positional Embeddings – Efficiently encodes positions for very long sequences.

🔄 Mixed Attention (Local + LSH) – Balances local dependencies with global scalability.

📊 Key Claims

Processes sequences >500K tokens on a single GPU.

Achieves comparable accuracy to full self-attention models.

Provides up to 40× speedup on long-sequence benchmarks.

📚 Learning Highlights

✅ Memory savings via reversible layers and chunked feed-forward activations.

✅ Performance parity with full attention models on character-level language modeling benchmarks.

✅ Demonstrated efficiency on Crime and Punishment (~800 pages).
