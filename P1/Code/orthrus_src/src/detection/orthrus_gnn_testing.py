from tqdm import tqdm

from encoders import OrthrusEncoder
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch


@torch.no_grad()
def test(
        data,
        full_data,
        model,
        nodeid2msg,
        split,
        model_epoch_file,
        cfg,
        device,
):
    model.eval()

    time_with_loss = {}  # key: time，  value： the losses
    edge_list = []
    unique_nodes = torch.tensor([]).to(device=device)
    start_time = data.t[0]
    event_count = 0
    tot_loss = 0
    start = time.perf_counter()

    # NOTE: warning, this may reindexes the graph
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        unique_nodes = torch.cat([unique_nodes, batch.edge_index.flatten()]).unique()

        each_edge_loss = model(batch, full_data, inference=True)
        tot_loss += each_edge_loss.sum().item()

        # If the graph has been reindexed in the loader, we retrieve original node IDs
        # to later find the labels
        if hasattr(batch, "original_edge_index"):
            edge_index = batch.original_edge_index
        else:
            edge_index = batch.edge_index
        
        num_events = each_edge_loss.shape[0]
        edge_types = torch.argmax(batch.edge_type, dim=1) + 1
        for i in range(num_events):
            srcnode = int(edge_index[0, i])
            dstnode = int(edge_index[1, i])

            srcmsg = nodeid2msg[srcnode]
            dstmsg = nodeid2msg[dstnode]
            t_var = int(batch.t[i])
            edge_type_idx = edge_types[i].item()
            edge_type = rel2id[edge_type_idx]
            loss = each_edge_loss[i]

            temp_dic = {
                'loss': float(loss),
                'srcnode': srcnode,
                'dstnode': dstnode,
                'srcmsg': srcmsg,
                'dstmsg': dstmsg,
                'edge_type': edge_type,
                'time': t_var,
            }
            edge_list.append(temp_dic)

        event_count += num_events
    tot_loss /= event_count

    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(edge_list[-1]["time"])

    end = time.perf_counter()
    logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval.replace(":", "-") + ".csv")

    df = pd.DataFrame(edge_list)
    df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

    # log(
    #     f'Time: {time_interval}, Loss: {tot_loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')


def main(cfg):
    # load the map between nodeID and node labels
    cur, _ = init_database_connection(cfg)
    nodeid2msg = gen_nodeid2msg(cur=cur)
    nodeid2msg = {k: str(v) for k, v in nodeid2msg.items()}  # pre-compute because it's too slow in main loop

    _, val_data, _skip_test, full_data, max_node_num = load_all_datasets(cfg)
    del val_data  # OpTC RAM fix: val not needed for testing pass
    import gc
    gc.collect()
    # OpTC-large RAM fix: test graphs are streamed ONE AT A TIME from disk.
    _test_dir = os.path.join(cfg.edge_featurization.embed_edges._edge_embeds_dir, "test")
    _test_files = sorted(os.listdir(_test_dir))
    # OpTC RAM fix part 2: full_data lacks the test region (test not loaded).
    # Global e_id indices reach into it, so pad full_data with zero-feature
    # slots covering all test edges (shapes scanned one file at a time).
    import torch as _torch
    _test_total = 0
    # OpTC RAM fix: build padded tails WITHOUT materializing old+new copies;
    # allocate the full-size buffers once and copy into them.
    _shapes = []
    for _f in _test_files:
        _d = _torch.load(os.path.join(_test_dir, _f), weights_only=False, map_location="cpu").to("cpu")
        _test_total += _d.msg.shape[0]
        _shapes.append(_d.msg.shape[1])
        del _d
    _msg_dim = _shapes[0] if _shapes else 0
    _et_dim = full_data.edge_type.shape[1] if full_data.edge_type.dim() > 1 else None
    _new_msg = _torch.zeros((full_data.msg.shape[0] + _test_total, _msg_dim),
                            dtype=full_data.msg.dtype)
    _new_msg[:full_data.msg.shape[0]] = full_data.msg
    del full_data.msg
    _new_t = _torch.zeros(full_data.t.shape[0] + _test_total, dtype=full_data.t.dtype)
    _new_t[:full_data.t.shape[0]] = full_data.t
    del full_data.t
    _tail_shape = (_test_total, _et_dim) if _et_dim is not None else (_test_total,)
    _cur_et_shape = tuple(full_data.edge_type.shape[1:]) if full_data.edge_type.dim() > 1 else ()
    _new_et = _torch.zeros((full_data.edge_type.shape[0] + _test_total,) + _cur_et_shape,
                           dtype=full_data.edge_type.dtype)
    _new_et[:full_data.edge_type.shape[0]] = full_data.edge_type
    del full_data.edge_type
    full_data.msg = _new_msg
    full_data.t = _new_t
    full_data.edge_type = _new_et
    del _new_msg, _new_t, _new_et
    import gc as _gc
    _gc.collect()
    log(f"OpTC pad: full_data extended by {_test_total} zero slots")

    # For each model trained at a given epoch, we test
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    all_trained_models = ["model_epoch_1"] if cfg._from_weights else listdir_sorted(gnn_models_dir)

    device = get_device(cfg)

    for trained_model in all_trained_models:
        log(f"Evaluation with model {trained_model}...")
        torch.cuda.empty_cache()
        _first_test = extract_msg_from_data([torch.load(os.path.join(_test_dir, _test_files[0]), weights_only=False, map_location="cpu").to("cpu")], cfg)[0]
        model = build_model(data_sample=_first_test, device=device, cfg=cfg, max_node_num=max_node_num)
        del _first_test
        model = load_model(model, os.path.join(gnn_models_dir, trained_model))
        # OpTC fix: checkpoint was built with train-only max_node; resize the
        # neighbor loader buffers to cover global node ids (test graphs).
        _nl = model.encoder.neighbor_loader
        if _nl.neighbors.shape[0] < max_node_num:
            import torch as _t2
            def _resize(buf):
                pad = _t2.full((max_node_num,) + tuple(buf.shape[1:]), -1,
                               dtype=buf.dtype)
                pad[:buf.shape[0]] = buf
                return pad
            _nl.neighbors = _resize(_nl.neighbors)
            _nl.e_id = _resize(_nl.e_id)  # padded with -1 => filtered as invalid
            _nl._assoc = _t2.zeros(max_node_num, dtype=_nl._assoc.dtype)
            # also resize assoc on the OrthrusEncoder itself (separate array)
            if hasattr(model.encoder, "assoc") and model.encoder.assoc.shape[0] < max_node_num:
                model.encoder.assoc = _t2.zeros(max_node_num, dtype=model.encoder.assoc.dtype)
            _nl.cur_e_id = int(_nl.e_id.max().item()) + 1 if _test_total > 0 else 0
        
        if cfg._from_weights:
            _wp = os.path.join(cfg._from_weights_path, f"{cfg.dataset.name}.pkl")
            if os.path.exists(_wp):  # OpTC: no pretrained weights for this dataset; use checkpoint trained above
                model.load_state_dict(torch.load(_wp, weights_only=False, map_location="cpu"))

        # TODO: we may want to move the validation set into the training for early stopping
        for split in ["val", "test"]:
            log(f"    Testing {split} set...")
            _split_dir = os.path.join(cfg.edge_featurization.embed_edges._edge_embeds_dir, split)
            _split_files = sorted(os.listdir(_split_dir))
            for _f in tqdm(_split_files, desc=f"{split} set with {trained_model}"):
                g = extract_msg_from_data([torch.load(os.path.join(_split_dir, _f), weights_only=False, map_location="cpu").to("cpu")], cfg)[0]
                g.to(device=device)
                test(
                    data=g,
                    full_data=full_data,
                    model=model,
                    nodeid2msg=nodeid2msg,
                    split=split,
                    model_epoch_file=trained_model,
                    cfg=cfg,
                    device=device,
                )
                g.to("cpu")

        del model


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
