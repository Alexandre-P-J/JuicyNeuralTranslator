
def unfold(doc):
    texts = []
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run.text:
                texts.append(run.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                texts.extend(unfold(cell))
    return texts


def fold(doc, replacements):
    def substitution(ref, replacements, offset):
        for paragraph in ref.paragraphs:
            for run in paragraph.runs:
                if run.text:
                    run.text = replacements[offset]
                    offset += 1
        for table in ref.tables:
            for row in table.rows:
                for cell in row.cells:
                    _, offset = substitution(cell, replacements, offset)
        return ref, offset
    return substitution(doc, replacements, 0)[0]
