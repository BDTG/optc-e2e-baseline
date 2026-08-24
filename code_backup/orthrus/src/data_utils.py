import os

import pickle
import torch
from torch_geometric.data import Data, TemporalData
from torch_geometric.loader import TemporalDataLoader

from encoders import OrthrusEncoder


def load_all_datasets(cfg):
    # OpTC-large RAM fix: train graphs are also streamed from disk during
    # training (orthrus_gnn_training loads them lazily), so nothing is
    # preloaded here. full_data is built by scanning msg shapes only.
    train_data = []
    val_data = []
    test_data = []
    import torch as _t
    _dir_base = cfg.edge_featurization.embed_edges._edge_embeds_dir
    _all_msg, _all_t, _all_et = [], [], []
    max_node = 0
    for _sp in ["train", "val", "test"]:
        _sd = os.path.join(_dir_base, _sp)
        for _f in sorted(os.listdir(_sd)):
            _d = _t.load(os.path.join(_sd, _f)).to("cpu")
            _all_msg.append(_d.msg)
            _all_t.append(_d.t)
            # raw TemporalData may lack edge_type (added later by
            # extract_msg_from_data); fall back to zeros of the right length.
            _et_off = 2 * 3 + 2 * cfg.edge_featurization.embed_nodes.emb_dim // 2 + 10 - 10
            # robust: locate via expected layout from extract_msg_from_data:
            # [src_type(nt) | src_emb(e) | edge_type(et) | dst_type(nt) | dst_emb(e)]
            _nt = cfg.dataset.num_node_types
            _e = cfg.edge_featurization.embed_nodes.emb_dim
            _etn = cfg.dataset.num_edge_types
            _off_et = _nt + _e
            _all_et.append(_d.msg[:, _off_et:_off_et + _etn])
            max_node = max(max_node, _t.cat([_d.src, _d.dst]).max().item())
            del _d
    _train_dir = os.path.join(_dir_base, "train")
    train_data = _LazyGraphList(_train_dir, cfg)
    full_data = Data(msg=_t.cat(_all_msg), t=_t.cat(_all_t), edge_type=_t.cat(_all_et))
    del _all_msg, _all_t, _all_et
    max_node = max_node + 1
    print(f"Max node in {cfg.dataset.name}: {max_node}")
    return train_data, [], [], full_data, max_node


class _LazyGraphList(list):
    """List-like that loads graphs from disk on index access (RAM-safe)."""
    def __init__(self, dir_path, cfg):
        self._dir = dir_path
        self._cfg = cfg
        self._files = sorted(os.listdir(dir_path))
    def __len__(self):
        return len(self._files)
    def __getitem__(self, i):
        _d = torch.load(os.path.join(self._dir, self._files[i])).to("cpu")
        out = extract_msg_from_data([_d], self._cfg)[0]
        # extract_msg_from_data splits msg into fields and deletes the raw
        # `msg` attr; the batch loader + encoder expect `msg` present.
        if not hasattr(out, "msg") or out.msg is None:
            out.msg = torch.cat(
                [out.src_type, out.src_emb, out.edge_type, out.dst_type, out.dst_emb],
                dim=-1,
            )
        return out
    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)
    

def load_data_set(cfg, path: str, split: str) -> list[TemporalData]:
    """
    Returns a list of time window graphs for a given `split` (train/val/test set).
    """
    # In case we run unit tests, only some edges in the train set are present
    if cfg._test_mode:
        split = "train"

    data_list = []
    for f in sorted(os.listdir(os.path.join(path, split))):
        filepath = os.path.join(path, split, f)
        data = torch.load(filepath).to("cpu")
        data_list.append(data)

    if cfg.edge_featurization.embed_nodes.used_method.strip() == "only_type":
        data_list = extract_msg_node_type_only(data_list, cfg)
    else:
        data_list = extract_msg_from_data(data_list, cfg)
    return data_list

def extract_msg_node_type_only(data_set: list[TemporalData], cfg) -> list[TemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types

    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (node_type_dim * 2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")

    field_to_size = [
        ("src_type", node_type_dim),
        ("edge_type", edge_type_dim),
        ("dst_type", node_type_dim),
    ]
    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size:
            fields[field] = g.msg[:, idx: idx + size]
            idx += size

        x_src = fields["src_type"]
        x_dst = fields["dst_type"]

        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
            
        edge_feats = build_edge_feats(fields, msg, cfg)

        g.x_src = x_src
        g.x_dst = x_dst
        g.msg = msg
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        g.edge_index = torch.stack([g.src, g.dst])

    return data_set

def extract_msg_from_data(data_set: list[TemporalData], cfg) -> list[TemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    emb_dim = cfg.edge_featurization.embed_nodes.emb_dim
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types
    
    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (emb_dim*2) + (node_type_dim*2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")
    
    field_to_size = [
        ("src_type", node_type_dim),
        ("src_emb", emb_dim),
        ("edge_type", edge_type_dim),
        ("dst_type", node_type_dim),
        ("dst_emb", emb_dim),
    ]
    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size:
            fields[field] = g.msg[:, idx: idx + size]
            idx += size
            
        x_src = fields["src_emb"]
        x_dst = fields["dst_emb"]
        
        if cfg.detection.gnn_training.encoder.use_node_type_in_node_feats:
            x_src = torch.cat([x_src, fields["src_type"]], dim=-1)
            x_dst = torch.cat([x_dst, fields["dst_type"]], dim=-1)
        
        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
        
        edge_feats = build_edge_feats(fields, msg, cfg)
        
        g.x_src = x_src
        g.x_dst = x_dst
        g.msg = msg
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        g.edge_index = torch.stack([g.src, g.dst])
    
    return data_set

def build_edge_feats(fields, msg, cfg):
    edge_features = list(map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.edge_features.split(",")))
    edge_feats = []
    if "edge_type" in edge_features:
        edge_feats.append(fields["edge_type"])
    if "msg" in edge_features:
        edge_feats.append(msg)
    edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
    return edge_feats

def custom_temporal_data_loader(data: TemporalData, batch_size: int, *args, **kwargs):
    """
    A simple `TemporalDataLoader` which also update the edge_index with the
    sampled edges of size `batch_size`. By default, only attributes of shape (E, d)
    are updated, `edge_index` is thus not updated automatically.
    """
    loader = TemporalDataLoader(data, batch_size=batch_size, *args, **kwargs)
    for batch in loader:
        batch.edge_index = torch.stack([batch.src, batch.dst])
        yield batch

def temporal_data_to_data(data: TemporalData) -> Data:
    """
    NeighborLoader requires a `Data` object.
    We need to convert `TemporalData` to `Data` before using it.
    """
    return Data(num_nodes=data.x_src.shape[0], **{k: v for k, v in data._store.items()})

class GraphReindexer:
    """
    Simply transforms an edge_index and its src/dst node features of shape (E, d)
    to a reindexed edge_index with node IDs starting from 0 and src/dst node features of shape
    (max_num_node + 1, d).
    This reindexing is essential for the graph to be computed by a standard GNN model with PyG.
    """
    def __init__(self, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device
        
        self.assoc = None
        self.x_src_cache = None
        self.x_dst_cache = None

    def node_features_reshape(self, edge_index, x_src, x_dst, max_num_node=None):
        """
        Converts node features in shape (E, d) to a shape (N, d).
        Returns x as a tuple (x_src, x_dst).
        """
        if self.x_src_cache is None:
            self.x_src_cache = torch.zeros((self.num_nodes, x_src.shape[1]), device=self.device)
            self.x_dst_cache = torch.zeros((self.num_nodes, x_src.shape[1]), device=self.device)
            
        max_num_node = max_num_node + 1 if max_num_node else edge_index.max() + 1
        
        # To avoid storing gradients from all nodes, we detach() BEFORE caching. If we detach()
        # after storing, we loose the gradient for all operations happening before the reindexing.
        self.x_src_cache = self.x_src_cache.detach()
        self.x_dst_cache = self.x_dst_cache.detach()
        
        self.x_src_cache[edge_index[0, :]] = x_src
        self.x_dst_cache[edge_index[1, :]] = x_dst
        x = (self.x_src_cache[:max_num_node, :], self.x_dst_cache[:max_num_node, :])
        
        return x
    
    def reindex_graph(self, data):
        """
        Reindexes edge_index from 0 + reshapes node features.
        The old edge_index is stored in `data.original_edge_index`
        """
        data = data.clone()
        data.original_edge_index = data.edge_index
        (data.x_src, data.x_dst), data.edge_index = self._reindex_graph(data.edge_index, data.x_src, data.x_dst)
        return data
    
    def _reindex_graph(self, edge_index, x_src, x_dst):
        """
        Reindexes edge_index with indices starting from 0.
        Also reshapes the node features.
        """
        if self.assoc is None:
            self.assoc = torch.empty((self.num_nodes, ), dtype=torch.long, device=self.device)

        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=edge_index.device)
        edge_index = self.assoc[edge_index]
        
        # Associates each feature vector to each reindexed node ID
        x = self.node_features_reshape(edge_index, x_src, x_dst)
        
        return x, edge_index

def save_model(model, path: str, neigh_loader: bool=True):
    """
    Saves only the required weights and tensors on disk.
    Using torch.save() directly on the model is very long (up to 10min),
    so we select only the tensors we want to save/load.
    """
    os.makedirs(path, exist_ok=True)
    
    # We only save specific tensors, as the other tensors are not useful to save (assoc, cache, etc)
    torch.save(model.state_dict(), os.path.join(path, "state_dict.pkl"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    
    if neigh_loader and isinstance(model.encoder, OrthrusEncoder):
        torch.save(model.encoder.neighbor_loader, os.path.join(path, "neighbor_loader.pkl"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model, path: str, neigh_loader: bool=True):
    """
    Loads weights and tensors from disk into a model.
    """
    model.load_state_dict(
        torch.load(os.path.join(path, "state_dict.pkl")))
    
    if neigh_loader and isinstance(model.encoder, OrthrusEncoder):
        model.encoder.neighbor_loader = torch.load(os.path.join(path, "neighbor_loader.pkl"))

    return model
