#!/usr/bin/env python3
"""Source-level mutation testing for C (KotlinX GC/runtime) and Zig (pyrlyn/jit) without touching the repo.

Each mutant is one textual change to one line of one file. The file is mutated in a private copy of the tree
under --work (never in the checkout); the test command runs inside that copy with a timeout.
Outcomes: killed (tests fail), survived (tests pass), timeout (counted as detected), build-error (counted as
killed only when the build is part of the tests, which it is for both repos), skipped (mutant identical).

usage: mutate.py list   --root R --file F [--seed S --max N --ops a,b]
       mutate.py run    --root R --file F --setup CMD --test CMD --work DIR [--timeout S --max N --seed S --jobs J]
       --setup: shell run once in the fresh copy dir ($MUT_DIR) to prepare it (copy/symlink files)
       --test : shell run in $MUT_DIR after the mutation is written; exit 0 = survived.
Results are appended as JSON lines to DIR/results.jsonl (resumable: done ids are skipped).
"""
import argparse, json, os, random, re, shutil, subprocess, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor

OPS = {
    # name: (regex, replacement)
    "eq-ne":   (r"(?<![=!<>])==(?!=)", "!="),
    "ne-eq":   (r"!=(?!=)", "=="),
    "lt-le":   (r"(?<![<\-])<(?![<=])", "<="),
    "le-lt":   (r"<=", "<"),
    "gt-ge":   (r"(?<![>\-])>(?![>=])", ">="),
    "ge-gt":   (r">=", ">"),
    "and-or":  (r"&&", "||"),
    "or-and":  (r"\|\|", "&&"),
    "zig-and-or": (r"\band\b", "or"),
    "zig-or-and": (r"\bor\b", "and"),
    "plus-minus": (r"(?<=\w|\))\s\+\s(?=[\w(])", " - "),
    "minus-plus": (r"(?<=\w|\))\s-\s(?=[\w(])", " + "),
    "inc-dec": (r"\+\+", "--"),
    "dec-inc": (r"(?<!-)--(?!-)", "++"),
    "const-0-1": (r"(?<![\w.])0(?![\w.x])", "1"),
    "const-1-0": (r"(?<![\w.])1(?![\w.])", "0"),
    "const-plus1": (r"(?<![\w.])([2-9]|[1-9]\d+)(?![\w.])", None),   # n -> n+1
    "const-minus1": (r"(?<![\w.])([2-9]|[1-9]\d+)(?![\w.])", None),  # n -> n-1
    "hex-flip": (r"0x([0-9A-Fa-f]+)", None),                          # flip lowest bit of hex constant (opcode encodings)
    "negate-if": (r"\bif \((.*)\)(?=\s*[{\w])", None),
    "shl-shr": (r"<<", ">>"),
    "shr-shl": (r">>", "<<"),
    "true-false": (r"\btrue\b", "false"),
    "false-true": (r"\bfalse\b", "true"),
    "del-stmt": (None, None),   # delete a whole statement line ending in ';'
    "drop-rc": (None, None),    # delete a line doing an RC inc/dec, barrier, fence or log of a reference op
}
RC_RE = re.compile(r"(\brc\w*(inc|dec|retain|release)\w*|\b(inc|dec)ref\w*|_rc\s*(\+\+|--|\+=|-=)|\brt_(retain|release)\w*|barrier|atomic_thread_fence|__atomic_\w+|atomic_fetch_\w+|log_(push|op|ref)\w*|possible_root)", re.I)
SKIP_LINE = re.compile(r"^\s*(#|//|/\*|\*|///|//!)|\btest\b\s*\"|@import|std\.debug\.print|fprintf|printf|assert")

def strip_comment(line, lang):
    i = line.find("//")
    return (line[:i], line[i:]) if i >= 0 else (line, "")

def mutants_for(path, lang, ops):
    lines = open(path, encoding="utf-8").read().split("\n")
    out = []
    in_block = False
    for ln, full in enumerate(lines):
        s = full.strip()
        if lang == "c":
            if "/*" in s and "*/" not in s: in_block = True; continue
            if in_block:
                if "*/" in s: in_block = False
                continue
        if not s or SKIP_LINE.search(full): continue
        code, cmt = strip_comment(full, lang)
        if '"' in code: code_q = re.sub(r'"(?:\\.|[^"\\])*"', lambda m: "\0" * len(m.group()), code)
        else: code_q = code
        for op in ops:
            rx, rep = OPS[op]
            if op == "del-stmt":
                if code.rstrip().endswith(";") and not re.match(r"\s*(return\b|break\b|continue\b|const\b|var\b|static\b|extern\b|typedef\b|struct\b|enum\b|[A-Za-z_][\w\s\*]*\b\w+\s*(=|;|\[))", code) \
                   or re.match(r"\s*return\b[^;]*;\s*$", code):
                    if re.match(r"\s*[A-Za-z_][\w:]*\s+\**\w+\s*=", code) and lang == "c": continue
                    out.append((op, ln, full, code[:len(code) - len(code.lstrip())] + (";" if lang == "c" else "{}") + "  // MUTANT: deleted"))
                continue
            if op == "drop-rc":
                if RC_RE.search(code) and code.rstrip().endswith(";") and "=" not in code.replace("==", ""):
                    indent = code[:len(code) - len(code.lstrip())]
                    out.append((op, ln, full, indent + ("(void)0;" if lang == "c" else "{}") + "  // MUTANT: dropped rc/barrier"))
                continue
            if lang == "zig" and op in ("and-or", "or-and"): continue
            if lang == "c" and op in ("zig-and-or", "zig-or-and"): continue
            for m in re.finditer(rx, code_q):
                a, b = m.span()
                if op == "const-plus1": r = str(int(m.group(1)) + 1)
                elif op == "const-minus1": r = str(int(m.group(1)) - 1)
                elif op == "hex-flip": r = "0x" + format(int(m.group(1), 16) ^ 1, "0%dX" % len(m.group(1)))
                elif op == "negate-if": r = "if (!(" + m.group(1) + "))"
                else: r = rep
                if lang == "zig" and op == "negate-if": r = r  # same syntax
                newc = code[:a] + r + code[b:]
                out.append((op, ln, full, newc + cmt))
    return lines, out

def mid(rel, op, ln, new):
    return hashlib.sha1(f"{rel}:{ln}:{op}:{new}".encode()).hexdigest()[:10]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--root", required=True); ap.add_argument("--file", required=True)
    ap.add_argument("--setup", default=""); ap.add_argument("--test", default="")
    ap.add_argument("--work", default=""); ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--ops", default="")
    ap.add_argument("--lines", default="", help="regex: only lines whose text matches")
    a = ap.parse_args()
    lang = "zig" if a.file.endswith(".zig") else "c"
    ops = a.ops.split(",") if a.ops else list(OPS)
    path = os.path.join(a.root, a.file)
    lines, ms = mutants_for(path, lang, ops)
    if a.lines: ms = [m for m in ms if re.search(a.lines, m[2])]
    random.Random(a.seed).shuffle(ms)
    if a.max: ms = ms[:a.max]
    if a.cmd == "list":
        for op, ln, old, new in ms: print(f"{a.file}:{ln+1} [{op}]\n  - {old.strip()}\n  + {new.strip()}")
        print(f"{len(ms)} mutants", file=sys.stderr); return
    os.makedirs(a.work, exist_ok=True)
    resf = os.path.join(a.work, "results.jsonl")
    done = set()
    if os.path.exists(resf):
        for l in open(resf): done.add(json.loads(l)["id"])
    todo = [m for m in ms if mid(a.file, m[0], m[1], m[3]) not in done]
    print(f"{len(ms)} sampled, {len(todo)} to run", flush=True)
    def one(slot_m):
        slot, (op, ln, old, new) = slot_m
        i = mid(a.file, op, ln, new)
        d = os.path.join(a.work, f"slot{slot}")
        env = dict(os.environ, MUT_DIR=d, MUT_ROOT=a.root, MUT_FILE=a.file, TMPDIR=os.path.join(d, ".tmp"))
        if not os.path.isdir(d):
            os.makedirs(d); os.makedirs(env["TMPDIR"], exist_ok=True)
            subprocess.run(a.setup, shell=True, cwd=d, env=env, check=True)
        target = os.path.join(d, a.file)
        if os.path.islink(target): os.unlink(target)
        new_lines = list(lines); new_lines[ln] = new
        with open(target, "w") as f: f.write("\n".join(new_lines))
        t0 = time.time()
        import signal
        p = subprocess.Popen(a.test, shell=True, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            raw, _ = p.communicate(timeout=a.timeout)
            tail = raw.decode("utf-8", "replace")
            status = "survived" if p.returncode == 0 else "killed"
            killers = sorted(set(re.findall(r"^FAIL ([^\n:(]+)", tail, re.M)))[:8] or sorted(set(re.findall(r"error: '([^']+)' failed|^(\S.*?) \.\.\. FAIL|test\.(\S+)\.\.\.FAIL", tail, re.M)))[:5]
            note = tail[-600:] if status == "killed" and not killers else ""
            # A mutant the compiler rejects (zig: unused local, type error; clang: error) is not a test result.
            base = os.path.basename(a.file)
            if status == "killed" and re.search(re.escape(base) + r":\d+:\d+: error:", tail): status, note = "invalid", tail[-600:]
        except subprocess.TimeoutExpired:
            status, killers, note = "timeout", [], ""
        finally:
            # The whole process group: run.sh / zig build and every test program they started (a hung mutant
            # program would otherwise keep spinning after the timeout).
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
        with open(target, "w") as f: f.write("\n".join(lines))  # restore copy
        r = dict(id=i, file=a.file, line=ln + 1, op=op, old=old.strip(), new=new.strip(), status=status,
                 killers=[k if isinstance(k, str) else next(x for x in k if x) for k in killers], secs=round(time.time() - t0, 1), note=note)
        return r
    slots = list(range(a.jobs))
    with ThreadPoolExecutor(a.jobs) as ex:
        import queue
        q = queue.Queue(); [q.put(s) for s in slots]
        def wrap(m):
            s = q.get()
            try: return one((s, m))
            finally: q.put(s)
        for r in ex.map(wrap, todo):
            with open(resf, "a") as f: f.write(json.dumps(r) + "\n")
            print(f"{r['status']:8} {r['file']}:{r['line']} [{r['op']}] {r['secs']}s {','.join(r['killers'])}", flush=True)

if __name__ == "__main__": main()
