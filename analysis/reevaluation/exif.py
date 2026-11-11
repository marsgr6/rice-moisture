import os,re,sys,csv
root=sys.argv[1]; out=sys.argv[2]
pat=re.compile(rb'(20\d\d):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)')
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
            ts=[m.group(0).decode() for m in pat.finditer(b)]
            mk=b'iPhone' in b
            lens=re.search(rb'iPhone 13 Pro [^\x00]*camera[^\x00]*',b)
            rows.append([c,sub,f,ts[0] if ts else '',len(set(ts)),os.path.getsize(fp),lens.group(0).decode(errors='ignore') if lens else ''])
with open(out,'w',newline='') as fh: csv.writer(fh).writerows([['cond','folder','file','time','nts','size','lens']]+rows)
print(len(rows))
