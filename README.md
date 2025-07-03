# RAG and Chat QA Tool

## Overview

This tool is a Retrieval-Augmented Generation (RAG) and Chat-based Question Answering (QA) system. It allows users to ask questions and receive answers based on a provided knowledge base. The system combines the power of information retrieval with advanced language models to provide accurate and contextually relevant responses.

## Features

*   **Knowledge Base Ingestion:** Supports ingesting various data formats (e.g., text files, PDFs, websites) into a structured knowledge base.
*   **Efficient Retrieval:** Implements efficient retrieval mechanisms to find relevant documents or passages from the knowledge base based on the user's query.
*   **Contextualized Generation:** Leverages pre-trained language models to generate answers that are grounded in the retrieved context.
*   **Chat Interface:** Provides a user-friendly chat interface for users to interact with the QA system.
*   **Customizable:** Offers options to customize the retrieval and generation processes to optimize performance for specific domains.

## Create .env File and place below variables in .enf file
```bash
GOOGLE_API_KEY="your gemini api key"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```