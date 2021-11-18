# import fitz

# pdf = fitz.open("/home/pyro/Projects/TFG/Experimental/PDFedit/nice.pdf")
# #pdf = fitz.open("/home/pyro/Downloads/hello_alexander_fondo_azul.pdf")
# for page in pdf:
#     page.wrap_contents()
#     text = page.get_text("text").split('\n')
    
#     for data in text:
#         areas = page.searchFor(data)
#         if areas is not None:
#             print(type(areas[0]))
#             [page.addRedactAnnot(area, fill = False) for area in areas]
#     page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

# pdf.save("out.pdf")


import fitz
from fitz import Rect

pdf = fitz.open("/home/pyro/Projects/TFG/Experimental/PDFedit/nice.pdf")
#pdf = fitz.open("/home/pyro/Downloads/hello_alexander_fondo_azul.pdf")
for page in pdf:
    page.wrap_contents()
    blocks = page.get_text("blocks")
    for block in blocks:
        r = Rect(block[0], block[1], block[2], block[3])
        page.addRedactAnnot(r, fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

pdf.save("out.pdf")