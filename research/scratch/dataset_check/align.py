import sys, openpyxl, collections, random
wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
rows = [r[:4] for r in wb.worksheets[0].iter_rows(values_only=True)][1:]
idx = [(i,r) for i,r in enumerate(rows) if r[0] is not None]
# block structure: runs of Lang
runs=[]; 
for i,r in idx:
    g=r[3].lower()
    if runs and runs[-1][0]==g and runs[-1][2]==i-1: runs[-1][2]=i; runs[-1][3]+=1
    else: runs.append([g,i,i,1])
print("lang runs (lang,first,last,count):", runs[:10], "... total runs", len(runs))
ur=[r for _,r in idx if r[3].lower()=='urdu']; en=[r for _,r in idx if r[3].lower()=='eng']
same=sum(a[2]==b[2] and str(a[1])==str(b[1]) for a,b in zip(ur,en))
print("position-aligned type+label agreement ur vs en:", same, "/", len(ur))
random.seed(0)
for t in ['edge_case','jailbreak','adversarial','benign']:
    ex=[str(r[0])[:110] for r in en if r[2]==t]
    print(f"-- eng {t} examples:"); [print("   ", e) for e in random.sample(sorted(set(ex)),4)]
blank=[i for i,r in enumerate(rows) if r[0] is None]
print("blank rows:", len(blank), "first/last:", blank[:2], blank[-2:])
