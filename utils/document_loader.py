from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def extract_document_text(uploaded_file):

    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    file_bytes = uploaded_file.getvalue()

    if extension == ".pdf":

        reader = PdfReader(BytesIO(file_bytes))

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        extracted_text = "\n".join(pages)

    elif extension == ".docx":

        document = Document(BytesIO(file_bytes))

        parts = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                parts.append(paragraph.text)

        # Extract text from tables as well.
        for table in document.tables:

            for row in table.rows:

                row_text = " | ".join(
                    cell.text for cell in row.cells
                )

                if row_text.strip():
                    parts.append(row_text)

        extracted_text = "\n".join(parts)

    elif extension == ".txt":

        extracted_text = file_bytes.decode(
            "utf-8",
            errors="replace"
        )

    else:

        raise ValueError(
            "Unsupported file format. Use PDF, DOCX, or TXT."
        )

    # Basic cleaning without changing the document's meaning.
    cleaned_lines = []

    for line in extracted_text.splitlines():

        cleaned_line = " ".join(line.split())

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    result = "\n".join(cleaned_lines).strip()

    if not result:
        raise ValueError(
            "No readable text was extracted from the document."
        )

    return result