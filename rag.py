import tempfile

import chromadb

from pypdf import PdfReader

from sentence_transformers import (
    SentenceTransformer
)


class KnowledgeBase:

    def __init__(self):

        self.client = (
            chromadb.PersistentClient(
                path="chroma_db"
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="course_materials"
            )
        )

        self.embedder = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )


    # -------------------------
    # ADD PDF
    # -------------------------

    def add_pdf(self, uploaded_file):

        pdf_data = (
            uploaded_file.getvalue()
        )


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            temp.write(pdf_data)

            pdf_path = temp.name


        reader = PdfReader(
            pdf_path
        )


        text = ""

        for page in reader.pages:

            page_text = (
                page.extract_text()
                or ""
            )

            text += page_text + "\n"


        chunks = self.create_chunks(
            text
        )


        if not chunks:

            return (
                "❌ No readable text found "
                "in the PDF."
            )


        embeddings = (
            self.embedder
            .encode(chunks)
            .tolist()
        )


        current_count = (
            self.collection.count()
        )


        ids = [

            f"document_{current_count}_{i}"

            for i in range(
                len(chunks)
            )

        ]


        self.collection.add(

            ids=ids,

            documents=chunks,

            embeddings=embeddings,

            metadatas=[

                {
                    "source":
                    uploaded_file.name
                }

                for _ in chunks

            ]

        )


        return (
            f"✅ {uploaded_file.name} "
            f"added successfully."
        )


    # -------------------------
    # CREATE CHUNKS
    # -------------------------

    def create_chunks(
        self,
        text,
        chunk_size=500
    ):

        words = text.split()

        chunks = []


        for i in range(
            0,
            len(words),
            chunk_size
        ):

            chunk = " ".join(

                words[
                    i:i + chunk_size
                ]

            )

            if chunk:

                chunks.append(
                    chunk
                )


        return chunks


    # -------------------------
    # SEARCH
    # -------------------------

    def search(
        self,
        query,
        k=4
    ):

        if self.collection.count() == 0:

            return []


        query_embedding = (

            self.embedder
            .encode([query])
            .tolist()

        )


        results = (
            self.collection.query(

                query_embeddings=
                query_embedding,

                n_results=min(
                    k,
                    self.collection.count()
                )

            )
        )


        return results.get(
            "documents",
            [[]]
        )[0]