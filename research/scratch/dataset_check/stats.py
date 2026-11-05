import sys, openpyxl, collections, re
wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
rows = [r[:4] for r in wb.worksheets[0].iter_rows(values_only=True)][1:]
rows = [r for r in rows if r[0] is not None]
ar = lambda s: any(0x600<=ord(c)<=0x6FF for c in s)
lat = lambda s: sum(c.isascii() and c.isalpha() for c in s)
by = collections.defaultdict(list)
for p,l,a,g in rows: by[g.lower()].append((str(p),str(l),a))
for g,v in by.items():
    ps=[p for p,_,_ in v]
    print(f"[{g}] n={len(v)} unique={len(set(ps))} labels={dict(collections.Counter(l for _,l,_ in v))} types={dict(collections.Counter(a for *_,a in v))}")
    print(f"    arabic-script={sum(ar(p) for p in ps)}  latin-only(no arabic, >=60% ascii letters)={sum((not ar(p)) and lat(p)>=0.6*len(p.replace(' ','')) for p in ps)}  non-ascii-latin-homoglyph={sum(any(0x370<=ord(c)<=0x4FF for c in p) for p in ps)}")
    print(f"    per type unique: {{{', '.join(f'{t}: {len(set(p for p,_,a in v if a==t))}' for t in set(a for *_,a in v))}}}")
# label/type consistency
print("label x type:", dict(collections.Counter((l,a) for _,l,a,_ in [(r[0],str(r[1]),r[2],r[3]) for r in rows])))
# Roman Urdu marker words in latin-only rows
mk=re.compile(r"\b(hai|hain|karo|karein|kya|nahi|nahin|aap|mujhe|ke|ki|ko|se|aur)\b",re.I)
lat_rows=[(p,g) for p,l,a,g in rows if not ar(str(p))]
print("latin rows with >=2 Roman-Urdu function words:", sum(len(mk.findall(str(p)))>=2 for p,g in lat_rows))
# cross-lingual overlap: English prompts that are exact dupes across sets
en=set(p for p,_,_ in by['eng']); 
print("exact dup rows overall:", len(rows)-len(set((str(r[0]),r[3].lower()) for r in rows)))
