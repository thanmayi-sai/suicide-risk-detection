from pathlib import Path
import sys, json, textwrap
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
sys.path.insert(0,str(ROOT/"src"))
from inference_pipeline import analyze

OUT=ROOT/"reports"
audio=ROOT/"Datasets"/"validation"/"10"/"negative_out.wav"
sample_text="I am feeling very stressed and overwhelmed today."

results={
 "TEXT ONLY": analyze(text=sample_text, audio_path=None),
 "SPEECH ONLY": analyze(text=None, audio_path=audio),
 "TEXT + SPEECH": analyze(text=sample_text, audio_path=audio),
}

def font(size,bold=False):
    cands=[
      "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
      "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for p in cands:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def box(draw,xy,fill,outline=None,width=1,r=20):
    draw.rounded_rectangle(xy,radius=r,fill=fill,outline=outline,width=width)

def wrap(draw,text,f,maxw):
    words=str(text).split()
    lines=[]; cur=""
    for w in words:
        test=(cur+" "+w).strip()
        if draw.textbbox((0,0),test,font=f)[2] <= maxw:
            cur=test
        else:
            if cur: lines.append(cur)
            cur=w
    if cur: lines.append(cur)
    return lines

for idx,(mode,res) in enumerate(results.items(),1):
    W,H=1500,1000
    im=Image.new("RGB",(W,H),(15,18,24))
    d=ImageDraw.Draw(im)
    white=(245,245,247); muted=(170,175,185); card=(36,40,50); accent=(255,75,75); green=(45,170,95)
    d.text((100,70),"Suicide-Risk Detection Using NLP-Based Text and Speech Analysis",font=font(42,True),fill=white)
    d.text((100,130),"Research screening prototype — verified model output",font=font(23),fill=muted)
    box(d,(100,190,1400,275),card)
    d.text((130,212),"INPUT MODE",font=font(21,True),fill=accent)
    d.text((310,208),mode,font=font(30,True),fill=white)
    y=315
    if res.get("text"):
        box(d,(100,y,1400,y+130),card)
        d.text((130,y+22),"Text Input",font=font(20,True),fill=muted)
        for j,line in enumerate(wrap(d,sample_text,font(25),1200)):
            d.text((130,y+55+j*30),line,font=font(25),fill=white)
        y+=155
    if res.get("speech"):
        box(d,(100,y,1400,y+115),card)
        d.text((130,y+20),"Speech Input",font=font(20,True),fill=muted)
        d.text((130,y+54),str(audio),font=font(22),fill=white)
        y+=140

    dec=res["decision"]
    box(d,(100,y,1400,y+230),card)
    d.text((130,y+20),"OUTPUT",font=font(20,True),fill=accent)
    d.text((130,y+62),"Screening Level",font=font(24),fill=muted)
    d.text((370,y+56),dec["screening_level"],font=font(34,True),fill=green if dec["screening_level"]=="Low" else accent)
    d.text((130,y+115),"Combined Evidence Score",font=font(24),fill=muted)
    d.text((500,y+109),f"{dec['combined_evidence_score']:.3f}",font=font(34,True),fill=white)
    if res.get("text"):
        d.text((760,y+62),"P_text",font=font(24),fill=muted)
        d.text((870,y+56),f"{res['text']['text_probability']:.3f}",font=font(34,True),fill=white)
    if res.get("speech"):
        d.text((760,y+115),"P_speech",font=font(24),fill=muted)
        d.text((900,y+109),f"{res['speech']['speech_distress_probability']:.3f}",font=font(34,True),fill=white)
    d.text((130,y+175),"Mode: "+dec["mode"]+"   |   Weights: "+json.dumps(dec["weights"]),font=font(20),fill=muted)
    y+=260

    box(d,(100,y,1400,min(y+155,H-60)),(25,29,37),outline=(70,75,85),width=1,r=15)
    disclaimer=res["disclaimer"]
    lines=wrap(d,disclaimer,font(18),1210)
    d.text((130,y+18),"Research Disclaimer",font=font(19,True),fill=muted)
    for j,line in enumerate(lines[:4]):
        d.text((130,y+50+j*25),line,font=font(18),fill=muted)

    name={1:"OUTPUT_01_TEXT_ONLY.png",2:"OUTPUT_02_SPEECH_ONLY.png",3:"OUTPUT_03_TEXT_AND_SPEECH.png"}[idx]
    im.save(OUT/name)
    tprob = res["text"]["text_probability"] if res.get("text") else None
    sprob = res["speech"]["speech_distress_probability"] if res.get("speech") else None
    print(name,dec["screening_level"],dec["combined_evidence_score"],tprob,sprob)

(OUT/"THREE_MODE_OUTPUT_VALUES.json").write_text(json.dumps(results,indent=2,default=float),encoding="utf-8")
