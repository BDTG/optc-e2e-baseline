import torch, csv, json

gt = set()
with open(r'D:\orthrus_laptop\optec-l6\orthrus\Ground_Truth\OPTC_H051\node_h051_0925.csv') as f:
    for row in csv.reader(f):
        gt.add(int(row[2]))

results = {}
for name, p in [('FLASH', r'D:\orthrus_laptop\fm_scores\backups\artifacts_flashmagic_full\evaluation\evaluation\abd2bffbd8e29156ae28ab444aa14b5717fccecfacd6cde08e76d5279e71463f\optc_h051\precision_recall_dir\scores_model_epoch_11.pkl'),
                ('MAGIC', r'D:\orthrus_laptop\fm_scores\backups\artifacts_flashmagic_full\evaluation\evaluation\3b2acf17e906e6eebace6e6761cbcfbb9340dcca78843ee99bfe5e8019f6bc3f\optc_h051\precision_recall_dir\scores_model_epoch_11.pkl')]:
    d = torch.load(p, weights_only=False, map_location='cpu')
    nodes = d['nodes']
    scores = d['pred_scores']
    pairs = sorted(zip(scores, nodes), key=lambda x: -x[0])
    node_set = set(nodes)
    active_gt = gt & node_set
    print(f'=== {name}: {len(node_set)} nodes, active GT {len(active_gt)}/114 ===')
    sweep = {}
    for k in [10000, 20000, 50000]:
        topk = set(nid for _, nid in pairs[:k])
        tp_a = len(topk & active_gt)
        tp_f = len(topk & gt)
        sweep[str(k)] = {'tp_active': int(tp_a), 'pct_ceiling': round(tp_a/len(active_gt)*100,1), 'tp_full': int(tp_f)}
        print(f'  @{k}: TP={tp_a}/{len(active_gt)} ({sweep[str(k)]["pct_ceiling"]}%) | {round(tp_f/114*100,1)}% of 114')
    results[name] = {'active_gt': int(len(active_gt)), 'sweep': sweep}

json.dump(results, open(r'D:\orthrus_laptop\FM_SWEEP.json','w'), indent=2)
print('saved FM_SWEEP.json')
