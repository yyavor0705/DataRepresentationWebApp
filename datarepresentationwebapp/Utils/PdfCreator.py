import io

from django.conf import settings
from fpdf import FPDF


class PDFDoc:
    def __init__(self):
        self.__pdf_doc = FPDF(orientation='P')
        self.__pdf_doc.add_page("P")
        self.__pdf_doc.set_auto_page_break(True)
        self.__pdf_doc.set_font('Arial', 'B', 16)

    def add_image(self, image_path):
        self.__pdf_doc.image(image_path, w=200)

    def add_text(self, text=""):
        self.__pdf_doc.multi_cell(0, 10, text, ln=1)

    def save_to_file(self, file_name="report.pdf"):
        self.__pdf_doc.output(file_name, "F")

    def save_as_byte_string(self):
        return self.__pdf_doc.output(dest='S')

    def add_page(self):
        self.__pdf_doc.add_page("P")
