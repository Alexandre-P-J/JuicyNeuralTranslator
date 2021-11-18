from reportlab.pdfgen.canvas import Canvas
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl


class CanvasFromPDF:
    def __init__(self, in_path, out_path) -> None:
        self.in_path = in_path
        self.out_path = out_path

    def __enter__(self):
        self.canvas = Canvas(self.out_path)
        return self

    def pages(self, from_page=None, to_page=None):
        pages = PdfReader(self.in_path).pages
        to_page = len(pages) if to_page is None else to_page
        from_page = 1 if from_page is None else from_page
        self.pages = [pagexobj(x) for x in pages[from_page - 1:to_page]]
        for page in self.pages:
            self.canvas.setPageSize((page.BBox[2], page.BBox[3]))
            self.canvas.doForm(makerl(self.canvas, page))
            yield self.canvas
            self.canvas.showPage()

    def __exit__(self, exc_type, exc_value, traceback):
        self.canvas.save()
