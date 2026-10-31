from pathlib import Path
import zipfile
D=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000\diagrams")
out=D/"Thanmayi_Final_System_Design_Diagrams_Package.zip"
items=["Thanmayi_Final_System_Design_Diagrams.drawio","DIAGRAM_VERIFICATION_REPORT.json","DIAGRAM_VERIFICATION_REPORT.txt","README_DIAGRAMS.txt","verify_drawio.py"]
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for n in items: z.write(D/n,arcname=n)
with zipfile.ZipFile(out) as z:
    names=z.namelist()
    bad=z.testzip()
print("ZIP",out)
print("CONTENTS",names)
print("TESTZIP",bad)
