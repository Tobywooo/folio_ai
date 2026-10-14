---
title: Folio AI
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "{{GRADIO_VERSION}}"
python_version: "3.13"
app_file: app.py
pinned: false
short_description: Notebook-style RAG study assistant with citations
---

# Folio AI

Gradio front end for [folio_ai](https://github.com/Tobywooo/folio_ai): create notebooks, add PDF/PPTX/TXT files and
web pages, ask questions answered with citations, and generate reports and quizzes.

This Space is deployed automatically by GitHub Actions from the `ui/` folder of the repository; edit the code there,
not here. It calls the project's Azure Function App backend, configured through the Space secrets
`BACKEND_URL` and `BACKEND_FUNCTION_KEY`.
