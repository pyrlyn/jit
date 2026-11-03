#!/usr/bin/env python3
"""Summary of mutate.py runs. usage: report.py <work dir>... [--survivors]"""
import json, sys, os, collections
dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
show = "--survivors" in sys.argv
tot = collections.Counter()
for d in dirs:
    f = os.path.join(d, "results.jsonl")
    if not os.path.exists(f): continue
    rs = {}
    for l in open(f):
        r = json.loads(l)
        if r["status"] == "killed" and ": error:" in r.get("note", "") and os.path.basename(r["file"]) + ":" in r["note"]: r["status"] = "invalid"
        rs[r["id"]] = r
    rs = list(rs.values()); inv = sum(r["status"] == "invalid" for r in rs); rs = [r for r in rs if r["status"] != "invalid"]
    c = collections.Counter(r["status"] for r in rs); tot.update(c)
    det = c["killed"] + c["timeout"]
    name = rs[0]["file"] if rs else d
    print(f"{name}: {len(rs)} mutants, killed {c['killed']}, timeout {c['timeout']}, survived {c['survived']}, score {100*det/max(1,len(rs)):.0f}% (+{inv} not compiling, excluded)")
    ops = collections.Counter((r["op"], r["status"]) for r in rs)
    print("   " + ", ".join(f"{op}:{ops[(op,'killed')]+ops[(op,'timeout')]}/{sum(v for (o,_),v in ops.items() if o==op)}" for op in sorted({o for o,_ in ops})))
    if show:
        for r in sorted(rs, key=lambda r: r["line"]):
            if r["status"] == "survived": print(f"   SURVIVED {r['file']}:{r['line']} [{r['op']}] {r['old'][:100]}  =>  {r['new'][:100]}")
n = sum(tot.values())
print(f"TOTAL {n} mutants, killed {tot['killed']}, timeout {tot['timeout']}, survived {tot['survived']}, score {100*(tot['killed']+tot['timeout'])/max(1,n):.0f}%")
