import unittest

from langchain_openai import OpenAIEmbeddings


class RecordingEmbeddingClient:
    def __init__(
        self,
    ):
        self.inputs = []

    def create(
        self,
        input,
        **kwargs,
    ):
        self.inputs.append(
            input
        )

        return {
            "data": [
                {
                    "embedding": [
                        float(index)
                    ]
                }
                for index, _ in enumerate(
                    input
                )
            ]
        }


class EmbeddingServiceTests(
    unittest.TestCase
):
    def test_compatible_api_receives_strings(self):
        client = RecordingEmbeddingClient()

        embeddings = OpenAIEmbeddings(
            model="test-embedding-model",
            api_key="test-api-key",
            client=client,
            check_embedding_ctx_length=False,
        )

        result = embeddings.embed_documents(
            [
                "第一段文本",
                "第二段文本",
            ]
        )

        self.assertEqual(
            client.inputs,
            [
                [
                    "第一段文本",
                    "第二段文本",
                ]
            ],
        )
        self.assertEqual(
            len(result),
            2,
        )


if __name__ == "__main__":
    unittest.main()
