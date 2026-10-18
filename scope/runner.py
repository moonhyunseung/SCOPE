"""Durable, bounded pilot runner. No model/LLM API is used."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, signal, sqlite3, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"PASS","FAIL","OOM","TIMEOUT","UNSUPPORTED","BLOCKED","SKIPPED_WITH_REASON"}
FILES = ("manifest.json","summary.json","metrics.csv","report.md","SHA256SUMS.txt")
def now(): return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(x): return hashlib.sha256(x).hexdigest()
def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n"); os.replace(tmp,path)
def command(args,timeout=30):
    p=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    return p.returncode,p.stdout
def source_hash():
    paths=sorted(list((ROOT/"scope").glob("*.py"))+list((ROOT/"tests").glob("*.py")))
    return digest(b"".join(p.relative_to(ROOT).as_posix().encode()+b"\0"+p.read_bytes() for p in paths))
def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())
def jobs_for(c):
    jobs=[{"id":"P0","block":"P0","deps":[],"params":{}}]
    captures=[]
    for m in c["models"]:
        for length in c["lengths"]:
            for category in c["categories"]:
                key=digest(canonical([m,length,category]).encode())[:12]
                p={"model":m,"length":length,"category":category}
                ident="P1-"+key; captures.append(ident)
                jobs.append({"id":ident,"block":"P1","deps":["P0"],"params":p})
                jobs.append({"id":"P3-"+key,"block":"P3","deps":[ident],"params":{"capture":ident}})
                jobs.append({"id":"P4-"+key,"block":"P4","deps":[ident,"P3-"+key],"params":{"capture":ident}})
    for b in ["P2","P5"]:
        jobs.append({"id":b,"block":b,"deps":captures,"allow_failed_deps":True,"params":{}})
    jobs.append({"id":"P6","block":"P6","deps":[j["id"] for j in jobs if j["block"]=="P4"],"allow_failed_deps":True,"params":{}})
    jobs.append({"id":"P7","block":"P7","deps":[j["id"] for j in jobs],"allow_failed_deps":True,"params":{}})
    return jobs
def connect(folder):
    db=sqlite3.connect(folder/"registry.sqlite",timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,spec TEXT,status TEXT,publication TEXT,attempts INT,updated TEXT)")
    return db
def prepare(config):
    c=load_config(config); key=digest(canonical(c).encode())[:16]
    folder=ROOT/"results"/"campaigns"/key; folder.mkdir(parents=True,exist_ok=True)
    meta=folder/"campaign.json"
    if meta.exists():
        old=json.loads(meta.read_text())
        if old["source_hash"]!=source_hash(): raise RuntimeError("SOURCE_DRIFT: resume requires frozen source")
    else:
        rc,sha=command(["git","rev-parse","HEAD"])
        atomic(meta,{"run_id":key,"created":now(),"code_sha":sha.strip(),"source_hash":source_hash(),"config":c})
    db=connect(folder)
    for spec in jobs_for(c):
        db.execute("INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?,?)",(spec["id"],canonical(spec),"PLANNED","NOT_READY",0,now()))
    db.commit()
    return c,folder,db
def gpu_gate(c):
    rc,out=command(["nvidia-smi","--query-gpu=index,name,memory.total,memory.free,utilization.gpu,driver_version","--format=csv,noheader,nounits"])
    if rc: raise RuntimeError("BLOCKED: nvidia-smi unavailable")
    lines=[x.strip().split(", ") for x in out.strip().splitlines()]
    if len(lines)!=1 or "H200" not in lines[0][1]: raise RuntimeError("BLOCKED: exactly one dedicated H200 required")
    if float(lines[0][3])<c["reserve_hbm_gib"]*1024+c["minimum_free_hbm_gib"]*1024: raise RuntimeError("BLOCKED: HBM budget")
    rc,pids=command(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"])
    if rc: raise RuntimeError("BLOCKED: process inventory unavailable")
    foreign=[x.strip() for x in pids.splitlines() if x.strip() and x.strip()!=str(os.getpid())]
    if foreign: raise RuntimeError("BLOCKED: GPU already has compute processes")
    import shutil
    if shutil.disk_usage(ROOT).free<c["reserve_disk_gib"]*2**30: raise RuntimeError("BLOCKED: disk reserve")
    return lines
def seal(folder,ident,status,detail,c,started):
    dest=folder/ident; dest.mkdir(exist_ok=True)
    output=dest/"output.json"
    data=json.loads(output.read_text()) if output.exists() else {}
    rows=data.get("metrics",[])
    atomic(dest/"summary.json",{"id":ident,"status":status,"detail":detail,"started":started,"ended":now(),"result":data})
    with (dest/"metrics.csv").open("w",newline="") as f:
        fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ["id","status"]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        writer.writerows(rows if rows else [{"id":ident,"status":status}])
    (dest/"report.md").write_text("# "+ident+"\n\nStatus: "+status+"\n\n"+data.get("interpretation",detail)+"\n\nPilot evidence only. Trace timing is not a performance benchmark. Full-model quality and resident-capacity frontier remain unestablished.\n")
    outputs={}
    for p in sorted(dest.iterdir()):
        if p.is_file() and p.name not in ["manifest.json","SHA256SUMS.txt"] and not p.name.endswith(".tmp"):
            outputs[p.name]={"sha256":digest(p.read_bytes()),"bytes":p.stat().st_size,"published":p.name in ["summary.json","metrics.csv","report.md"]}
    meta=json.loads((folder/"campaign.json").read_text())
    atomic(dest/"manifest.json",{"protocol_version":c["protocol_version"],"experiment_id":ident,"run_id":folder.name,"code_sha":meta["code_sha"],"source_hash":meta["source_hash"],"config_sha256":digest(canonical(c).encode()),"started":started,"ended":now(),"status":status,"seed":c["seed"],"planned_condition":json.loads(connect(folder).execute("SELECT spec FROM jobs WHERE id=?",(ident,)).fetchone()[0]),"outputs":outputs,"provenance":data.get("provenance",{}),"preflight":"results/preflight","raw_storage":"local experiment directory; see checksums; no public tensor download promised"})
    checks={p.name:digest(p.read_bytes()) for p in dest.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt" and not p.name.endswith(".tmp")}
    (dest/"SHA256SUMS.txt").write_text("".join(h+"  "+n+"\n" for n,h in sorted(checks.items())))
    verify(dest)
def verify(dest):
    for name in FILES:
        if not (dest/name).is_file() or not (dest/name).stat().st_size: raise RuntimeError("MISSING_ARTIFACT "+name)
    for line in (dest/"SHA256SUMS.txt").read_text().splitlines():
        sha,name=line.split("  ",1)
        if digest((dest/name).read_bytes())!=sha: raise RuntimeError("HASH_MISMATCH "+name)
    json.loads((dest/"manifest.json").read_text()); json.loads((dest/"summary.json").read_text())
def publish(folder,ident):
    dest=folder/ident; verify(dest)
    # Explicit compact-artifact allowlist. No raw log, prompt, sample, SQLite or environment secrets.
    paths=[str((dest/n).relative_to(ROOT)) for n in FILES]
    for p in paths:
        if (ROOT/p).stat().st_size>4*2**20: return False,"compact artifact size limit"
    env=os.environ.copy(); env["GIT_TERMINAL_PROMPT"]="0"; env["GIT_ASKPASS"]="/bin/false"; env["SSH_ASKPASS"]="/bin/false"
    def git(args):
        try:
            p=subprocess.run(["git"]+args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,timeout=60)
            return p.returncode,p.stdout
        except subprocess.TimeoutExpired:
            return 124,"git timeout; publication remains pending"
    for attempt in range(3):
        rc,msg=git(["add","-f","--"]+paths)
        if rc:return False,"git add failed"
        rc,msg=git(["diff","--cached","--name-only"])
        if set(msg.splitlines())-set(paths):return False,"unrelated staged files"
        rc,_=git(["diff","--cached","--quiet"])
        if rc:
            rc,msg=git(["commit","-m","results("+ident.split("-")[0]+"): "+ident])
            if rc:return False,"git commit identity/auth not configured"
        _,head=git(["rev-parse","HEAD"]); _,branch=git(["branch","--show-current"])
        if not branch.strip():return False,"detached publication branch"
        rc,msg=git(["push","origin","HEAD:refs/heads/"+branch.strip()])
        if rc==0:
            rc,remote=git(["ls-remote","origin","refs/heads/"+branch.strip()])
            if rc==0 and remote.split() and remote.split()[0]==head.strip():
                atomic(dest/"publication.json",{"status":"PUBLISHED","commit":head.strip(),"verified":now()})
                return True,head.strip()
        if attempt<2:time.sleep(2**attempt)
    return False,"push authentication/transport failure; GPU results preserved"
def report(folder,db):
    rows=db.execute("SELECT id,status,publication,attempts FROM jobs ORDER BY id").fetchall()
    counts={}
    for _,s,_,_ in rows:counts[s]=counts.get(s,0)+1
    atomic(folder/"status.json",{"run_id":folder.name,"updated":now(),"counts":counts,"jobs":[dict(zip(["id","status","publication","attempts"],r)) for r in rows]})
    return counts
def run(config,resume,publishing):
    c,folder,db=prepare(config)
    import fcntl
    lock=(folder/"worker.lock").open("w")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("another worker owns this campaign")
    if not resume and db.execute("SELECT COUNT(*) FROM jobs WHERE attempts>0").fetchone()[0]:raise RuntimeError("use --resume")
    # Lock proves an old RUNNING state has no surviving controller. Child has parent-death watchdog.
    for ident, in db.execute("SELECT id FROM jobs WHERE status='RUNNING'").fetchall():
        db.execute("UPDATE jobs SET status='PLANNED',updated=? WHERE id=?",(now(),ident))
    db.commit(); stop=threading.Event()
    def beat():
        while not stop.is_set():
            atomic(folder/"heartbeat.json",{"pid":os.getpid(),"updated":now(),"monotonic":time.monotonic()}); stop.wait(10)
    thread=threading.Thread(target=beat,daemon=True); thread.start(); began=time.monotonic(); attempted_publications=set()
    try:
        while True:
            rows=db.execute("SELECT id,spec,status,publication,attempts FROM jobs ORDER BY rowid").fetchall()
            states={r[0]:r[2] for r in rows}; progressed=False
            for ident,specjson,status,pub,attempts in rows:
                if publishing and status in TERMINAL and pub=="PENDING" and ident not in attempted_publications:
                    attempted_publications.add(ident)
                    ok,detail=publish(folder,ident)
                    db.execute("UPDATE jobs SET publication=? WHERE id=?",("PUBLISHED" if ok else "PENDING",ident));db.commit()
                if status!="PLANNED":continue
                spec=json.loads(specjson)
                if any(states[d] not in TERMINAL for d in spec["deps"]):continue
                started=now(); dest=folder/ident; dest.mkdir(exist_ok=True)
                if not spec.get("allow_failed_deps") and any(states[d]!="PASS" for d in spec["deps"]):
                    status="SKIPPED_WITH_REASON";detail="required predecessor did not PASS"
                elif time.monotonic()-began>c["campaign_timeout_seconds"]:
                    status="TIMEOUT";detail="campaign wall-clock budget exhausted"
                else:
                    try:
                        gpu_gate(c)
                        db.execute("UPDATE jobs SET status='RUNNING',attempts=attempts+1,updated=? WHERE id=?",(now(),ident));db.commit()
                        with (dest/"stdout.log").open("a") as log:
                            proc=subprocess.Popen([sys.executable,"-m","scope.experiments","--campaign",str(folder),"--job",ident],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            try:rc=proc.wait(timeout=c["job_timeout_seconds"])
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid,signal.SIGTERM)
                                try:proc.wait(15)
                                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                                rc=124
                        output=json.loads((dest/"output.json").read_text()) if (dest/"output.json").exists() else {}
                        status=output.get("status","TIMEOUT" if rc==124 else "FAIL")
                        if status not in TERMINAL:status="FAIL"
                        if rc and status=="PASS":status="FAIL"
                        detail=output.get("detail","worker exit "+str(rc))
                    except Exception as e:status="BLOCKED";detail=str(e)[:500]
                seal(folder,ident,status,detail,c,started)
                db.execute("UPDATE jobs SET status=?,publication='PENDING',updated=? WHERE id=?",(status,now(),ident));db.commit()
                if publishing:
                    attempted_publications.add(ident)
                    ok,detail=publish(folder,ident)
                    db.execute("UPDATE jobs SET publication=? WHERE id=?",("PUBLISHED" if ok else "PENDING",ident));db.commit()
                print(ident,status,flush=True);report(folder,db);progressed=True;break
            if not progressed:break
        print(json.dumps(report(folder,db)),flush=True)
    finally:
        stop.set();thread.join(12)
        atomic(folder/"heartbeat.json",{"pid":os.getpid(),"updated":now(),"state":"STOPPED"})
def main():
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["preflight","plan","run","status","reproduce"])
    ap.add_argument("--config",default="configs/h200_campaign.yaml");ap.add_argument("--resume",action="store_true")
    ap.add_argument("--publish-each-block",action="store_true");ap.add_argument("--run-id")
    a=ap.parse_args()
    if a.action=="preflight":
        from scope.experiments import preflight
        print(json.dumps(preflight(load_config(a.config)),indent=2));return
    if a.action=="reproduce":
        if not a.run_id:ap.error("--run-id is required")
        folder=ROOT/"results"/"campaigns"/a.run_id
        meta=json.loads((folder/"campaign.json").read_text())
        if meta["source_hash"]!=source_hash():raise RuntimeError("checkout code_sha "+meta["code_sha"])
        for dest in folder.iterdir():
            if dest.is_dir() and (dest/"manifest.json").exists():verify(dest)
        print("ARTIFACT_INTEGRITY_PASS; independent GPU rerun: use a fresh checkout at "+meta["code_sha"]+" and original config without copying results")
        return
    c,folder,db=prepare(a.config)
    if a.action in ["plan","status"]:
        print(json.dumps({"campaign":str(folder),"counts":report(folder,db),"planned":len(jobs_for(c))},indent=2));return
    db.close();run(a.config,a.resume,a.publish_each_block)
if __name__=="__main__":main()
