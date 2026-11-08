from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class SemanticEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        return self.model.encode(list(input), normalize_embeddings=True).tolist()


def make_embedding_function() -> SemanticEmbeddingFunction:
    return SemanticEmbeddingFunction()
