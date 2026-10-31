import xml.etree.ElementTree as ET, itertools, json, html, re
from pathlib import Path
p=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000\diagrams\Thanmayi_Final_System_Design_Diagrams.drawio")
root=ET.parse(p).getroot()
required=[
"1 Architecture - UML Component","2 Flow Chart","3 Logical Diagram",
"4 Behavioural - Activity Diagram","5 ER Diagram","6 DFD Level 1"
]
report={"file":str(p),"page_count":0,"pages":[],"required_pages":{},"obsolete_terms":{}}
all_text=[]
for d in root.findall("diagram"):
    name=d.get("name")
    report["page_count"]+=1
    model=d.find("mxGraphModel"); page_w=float(model.get("pageWidth","1600")); page_h=float(model.get("pageHeight","900"))
    rr=model.find("root")
    cells={c.get("id"):c for c in rr.findall("mxCell")}
    verts=[]; edges=[]; texts=[]
    for c in rr.findall("mxCell"):
        val=html.unescape(c.get("value",""))
        val=re.sub(r"<[^>]+>"," ",val)
        val=re.sub(r"\s+"," ",val).strip()
        if val: texts.append(val); all_text.append(val)
        if c.get("vertex")=="1":
            g=c.find("mxGeometry")
            if g is not None:
                x=float(g.get("x",0)); y=float(g.get("y",0)); w=float(g.get("width",0)); h=float(g.get("height",0))
                verts.append((c.get("id"),x,y,w,h,val,c.get("style","")))
        if c.get("edge")=="1":
            edges.append((c.get("id"),c.get("source"),c.get("target"),val))
    broken=[e for e in edges if e[1] not in cells or e[2] not in cells]
    outbounds=[v[0] for v in verts if v[1]<0 or v[2]<0 or v[1]+v[3]>page_w or v[2]+v[4]>page_h]
    # Check box-to-box overlaps; ignore pure text labels/footer/title boxes and intentional line-like elements
    boxes=[v for v in verts if "text;" not in v[6] and v[3]>5 and v[4]>5]
    overlaps=[]
    for a,b in itertools.combinations(boxes,2):
        ow=min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1])
        oh=min(a[2]+a[4],b[2]+b[4])-max(a[2],b[2])
        if ow>1 and oh>1:
            overlaps.append({"a":a[0],"b":b[0],"overlap_w":ow,"overlap_h":oh})
    report["pages"].append({
      "name":name,"vertices":len(verts),"edges":len(edges),"broken_edges":broken,
      "out_of_bounds":outbounds,"box_overlaps":overlaps,"text_preview":texts[:8]
    })
for x in required: report["required_pages"][x]=any(d.get("name")==x for d in root.findall("diagram"))
joined=" ".join(all_text).lower()
obsolete=["stopword","noise removal","cnn / lstm","feature fusion","clpsych","speechwellness","distress analysis interview corpus"]
for t in obsolete: report["obsolete_terms"][t]=t in joined
report["pass"]=(
    report["page_count"]==6 and all(report["required_pages"].values())
    and all(not p["broken_edges"] and not p["out_of_bounds"] and not p["box_overlaps"] for p in report["pages"])
    and not any(report["obsolete_terms"].values())
)
out=p.parent/"DIAGRAM_VERIFICATION_REPORT.json"
out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
txt=[]
txt.append("THANMAYI PROJECT DIAGRAM VERIFICATION")
txt.append("="*60)
txt.append(f"File: {p}")
txt.append(f"Pages: {report['page_count']}")
for pg in report["pages"]:
    txt.append(f"\n{pg['name']}: vertices={pg['vertices']}, edges={pg['edges']}, broken_edges={len(pg['broken_edges'])}, overlaps={len(pg['box_overlaps'])}, out_of_bounds={len(pg['out_of_bounds'])}")
txt.append("\nRequired pages:")
for k,v in report["required_pages"].items(): txt.append(f"- {k}: {'PASS' if v else 'MISSING'}")
txt.append("\nObsolete-term scan:")
for k,v in report["obsolete_terms"].items(): txt.append(f"- {k}: {'FOUND' if v else 'not found'}")
txt.append(f"\nOVERALL: {'PASS' if report['pass'] else 'REVIEW REQUIRED'}")
(p.parent/"DIAGRAM_VERIFICATION_REPORT.txt").write_text("\n".join(txt),encoding="utf-8")
print("\n".join(txt))
