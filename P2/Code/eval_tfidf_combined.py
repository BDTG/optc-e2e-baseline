import json, random, re, base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

def normalize_text(s):
    s=s.lower()
    m=re.search(r'-enc\s+([A-Za-z0-9+/=]{20,})', s)
    if m:
        try:
            b64=m.group(1)
            b64+='='*(-len(b64)%4)
            dec=base64.b64decode(b64).decode('utf-16le', errors='ignore')
            s=s.replace(m.group(1), dec[:200])
        except: pass
    s=re.sub(r'c:\\windows\\system32\\svchost\.exe.*','svchost',s)
    s=re.sub(r'c:\\[^\s|]+\.exe','proc',s)
    s=re.sub(r'%[^%]+%','temp',s)
    s=re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b','guid',s)
    s=re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b','ip',s)
    s=re.sub(r'\b[a-f0-9]{32,64}\b','hash',s)
    s=re.sub(r'%temp%','temp',s)
    return s[:800]

def load_s1():
    gt=set(json.load(open("/home/vung2/P1/Output/data/gt_and_scores.json"))["gt_nids"])
    alerts_all=[json.loads(l) for l in open("/home/vung2/P1/Output/data/alerts-enriched-v2.jsonl",encoding='utf-8') if l.strip()]
    def has_cmd2(a):
        for c in a.get("parent_chain",[]) or []:
            m=c.get("msg") or ""
            if "| cmd:" in m:
                v=m.split("| cmd:",1)[1].strip()
                if v and v.lower()!="none": return True
        return False
    def has_chain2(a): return len(a.get("parent_chain",[]) or [])>=2
    def has_events2(a): return len(a.get("event_seq",[]) or [])>=3
    s1=[a for a in alerts_all if has_cmd2(a) and has_chain2(a) and has_events2(a)]
    print(f"S1 filtered {len(s1)}")
    return s1, gt

s1, gt = load_s1()
evtx=[json.loads(l) for l in open("/home/vung2/P1/Output/data/evtx-chains.jsonl",encoding='utf-8') if l.strip()]
sd=[json.loads(l) for l in open("/home/vung2/P1/Output/data/sd-chains.jsonl",encoding='utf-8') if l.strip()]
random.seed(42); random.shuffle(sd)
sd_sample = sd[:2000]
print(f"S1 {len(s1)} EVTX {len(evtx)} SD {len(sd)} sample {len(sd_sample)}")
def to_text(co):
    if "chain" in co:
        ch=co["chain"]
        if isinstance(ch, list):
            return " | ".join([normalize_text(c.get("msg","") or "") for c in ch[-3:]])
        else:
            return normalize_text(str(ch))
    else:
        return normalize_text(co.get("msg",""))

combined=[]
for a in s1:
    nid=a.get("nid") or a.get("node_id") or a.get("id")
    label=1 if str(nid) in gt or a.get("is_malicious") else 0
    combined.append((to_text(a), label))
for e in evtx:
    combined.append((to_text(e), 1))
for s in sd_sample:
    combined.append((to_text(s), 1))
alerts_full=[json.loads(l) for l in open("/home/vung2/P1/Output/data/alerts-enriched-v2.jsonl",encoding='utf-8') if l.strip()]
benign=[a for a in alerts_full if str(a.get("nid")) not in gt and a not in s1]
random.shuffle(benign)
benign_sample=benign[:1200]
for b in benign_sample:
    combined.append((to_text(b), 0))
print(f"Combined {len(combined)} pos {sum(1 for _,l in combined if l==1)} benign {sum(1 for _,l in combined if l==0)}")
random.seed(42); random.shuffle(combined)
split=int(len(combined)*0.8)
train=combined[:split]
test=combined[split:]
print(f"train {len(train)} test {len(test)} pos_train {sum(l for _,l in train)} pos_test {sum(l for _,l in test)}")

X_train=[t for t,_ in train]; y_train=[l for _,l in train]
X_test=[t for t,_ in test]; y_test=[l for _,l in test]

vec=TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_features=50000)
Xtr=vec.fit_transform(X_train)
Xte=vec.transform(X_test)
clf=LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1)
clf.fit(Xtr, y_train)
probs=clf.predict_proba(Xte)[:,1]
ap=average_precision_score(y_test, probs)
auc=roc_auc_score(y_test, probs)
print(f"RESULT TFIDF-COMBINED-4142 ap={ap:.5f} auc={auc:.5f}")
import json as js
open("/home/vung2/P1/Output/results_phase2/tfidf-combined-4142-result.json","w").write(js.dumps({"eval_ap":ap,"eval_auc":auc,"n":len(combined),"train":len(train),"test":len(test)}, indent=2))
print("SAVED")
