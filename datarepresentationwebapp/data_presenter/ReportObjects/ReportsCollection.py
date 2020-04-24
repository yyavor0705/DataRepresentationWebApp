import os
import base64
import tempfile

from Utils.PdfCreator import PDFDoc


class ReportsCollection:

    @classmethod
    def prepare_report(cls, image_buf):
        pdf_doc = PDFDoc()
        with tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False) as fp:
            fp.write(image_buf.getvalue())
            fp.close()
            file_path = fp.name
        pdf_doc.add_image(file_path)
        os.remove(file_path)

        return pdf_doc.save_as_byte_string()
