import os,sys,struct,csv
root,out=sys.argv[1:3]
TAGS={0x829A:'exposure',0x829D:'fnumber',0x8827:'iso',0x9203:'brightness',0xA405:'focal35',0x9003:'dto',0xA403:'wb',0x920A:'focal',0xA404:'dzoom'}
def parse(b):
    i=b.find(b'MM\x00\x2a'); e='>'
    if i<0:
        i=b.find(b'II\x2a\x00'); e='<'
    if i<0: return {}
    t=b[i:]
    def ifd(off):
        r={}; n=struct.unpack(e+'H',t[off:off+2])[0]
        for k in range(n):
            ent=t[off+2+12*k:off+14+12*k]
            tag,typ,cnt=struct.unpack(e+'HHI',ent[:8]); vo=ent[8:]
            if typ in (5,10):
                o=struct.unpack(e+'I',vo)[0]; a,bb=struct.unpack(e+('II' if typ==5 else 'ii'),t[o:o+8]); v=a/bb if bb else None
            elif typ==3: v=struct.unpack(e+'H',vo[:2])[0]
            elif typ==4: v=struct.unpack(e+'I',vo)[0]
            elif typ==2:
                o=struct.unpack(e+'I',vo)[0] if cnt>4 else None; v=(t[o:o+cnt] if o else vo[:cnt]).rstrip(b'\x00').decode(errors='ignore')
            else: v=None
            r[tag]=v
        return r
    try:
        i0=ifd(struct.unpack(e+'I',t[4:8])[0]); ex=ifd(i0[0x8769]) if 0x8769 in i0 else {}
    except Exception: return {}
    return {TAGS[k]:v for k,v in ex.items() if k in TAGS}
rows=[]
for c in sorted(os.listdir(root)):
    p=os.path.join(root,c)
    if not os.path.isdir(p): continue
    for sub in sorted(os.listdir(p)):
        q=os.path.join(p,sub)
        for f in sorted(os.listdir(q)):
            fp=os.path.join(q,f)
            if os.path.isdir(fp): continue
            with open(fp,'rb') as fh: b=fh.read(300000)
            r=parse(b); r.update(cond=c,folder=sub,file=f); rows.append(r)
keys=['cond','folder','file']+list(TAGS.values())
with open(out,'w',newline='') as fh:
    w=csv.DictWriter(fh,keys); w.writeheader(); w.writerows(rows)
