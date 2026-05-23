from fpdf import FPDF
pdf=FPDF()
pdf.add_page()
pdf.set_font('Arial','B',16)
pdf.cell(0,10,'Test PDF',ln=True)
b=pdf.output(dest='S').encode('latin1')
print('length', len(b))
with open('models/test_report.pdf','wb') as f:
    f.write(b)
print('written test_report.pdf')
