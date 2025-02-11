DEFAULT_CHUNK_PROMPT = """OCR the following page into Markdown. Tables should be formatted as HTML.
Do not surround your output with triple backticks.

Chunk the document into sections of roughly 250 - 1500 words. Our goal is
to identify parts of the page with same semantic theme. These chunks will
be embedded and used in a RAG pipeline.

Images in the document should be properly described in details such that an LLM can understand the
image and answer questions about the image without seeing the image.
The image description should be returned as a chunk too.
"""

DEFAULT_PAGE_CHUNK_PROMPT = """OCR the following page into Markdown. Tables should be formatted as HTML.
Do not surround your output with triple backticks. The contents of the page should be returned as a single chunk.
Also return the semantic theme of the page.

Images in the document should be properly discribed in details such that an LLM can understand the image and answer
questions about the image without seeing the image.
The description should be returned as a part of the page content.
"""