from PyPDF2 import PdfFileReader, PdfFileWriter, PdfFileMerger


path = "/home/pyro/Projects/TFG/Docs/GEP02/Entrega.pdf"
#path = "/home/pyro/Downloads/AD-tema2-21-22.pdf"




inputStream = open(path, "rb")
outputStream = open("out.pdf", "wb")

src = PdfFileReader(inputStream)
output = PdfFileWriter()

[output.addPage(src.getPage(i)) for i in range(src.getNumPages())]
output.removeText(ignoreByteStringObject=True)
output.removeLinks()

output.write(outputStream)