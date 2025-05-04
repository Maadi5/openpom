import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Callable, Dict

# Assuming deepchem and openpom are available, otherwise you may need to stub these.
from deepchem.models.losses import Loss, L2Loss
# from deepchem.models.torch_models.torch_model import TorchModel
from openpom.models.torch_model5 import TorchModel  # Use only one of these as needed.
from deepchem.models.optimizers import Optimizer, LearningRateSchedule

from openpom.layers.pom_ffn2 import CustomPositionwiseFeedForward
from openpom.utils.loss import CustomMultiLabelLoss
from openpom.utils.optimizer import get_optimizer

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn import Set2Set, GlobalAttentionPooling
    from openpom.layers.pom_mpnn_gnn2 import CustomMPNNGNN
except (ImportError, ModuleNotFoundError):
    raise ImportError('This module requires dgl and dgllife')

########################################################################
# 1) Feed-Forward with Batch Normalization
########################################################################
class CustomPositionwiseFeedForwardBN(nn.Module):
    """
    A feed-forward layer that supports multiple hidden layers,
    dropout, activation, and batch normalization.
    """
    def __init__(self,
                 d_input: int,
                 d_hidden_list: List[int],
                 d_output: int,
                 activation: str = 'relu',
                 dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = True):
        super(CustomPositionwiseFeedForwardBN, self).__init__()

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(0.1)
        elif activation == 'prelu':
            self.activation_fn = nn.PReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'selu':
            self.activation_fn = nn.SELU()
        elif activation == 'elu':
            self.activation_fn = nn.ELU()
        elif activation == 'linear':
            self.activation_fn = lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout_at_input_no_act = dropout_at_input_no_act

        self.n_layers = len(d_hidden_list) + 1
        hidden_dims = [d_input] + d_hidden_list + [d_output]

        # Create linear layers
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(self.n_layers)]
        )

        # Create dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(self.n_layers)])

        # Create batchnorm layers (skipping BN on final output)
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_features=hidden_dims[i+1])
            if i < self.n_layers - 1 else nn.Identity()
            for i in range(self.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dropout, batch normalization, and activation.
        """
        for i in range(self.n_layers):
            # Linear transformation
            x = self.linears[i](x)
            # Batch normalization (skip BN on final layer)
            x = self.bns[i](x)
            # Activation and dropout for hidden layers
            if i < self.n_layers - 1:
                x = self.activation_fn(x)
                x = self.dropouts[i](x)
            else:
                # Optional dropout for final layer if specified
                if self.dropout_at_input_no_act:
                    x = self.dropouts[i](x)
        return x

########################################################################
# 2) EmbeddingLayer using graph-based readout and FFN
########################################################################
class EmbeddingLayer(nn.Module):
    def __init__(self, mpnn, readout, ffn, readout_type='global_sum_pooling'):
        super(EmbeddingLayer, self).__init__()
        self.mpnn = mpnn
        self._readout = readout
        self.ffn = ffn
        self.readout_type = readout_type

    def forward(self, node_feats, edge_feats):
        # print('embedding layer shapes: ', node_feats.shape, edge_feats.shape)
        node_encodings = self.mpnn(node_feats, edge_feats)
        molecular_encodings = self._readout(node_encodings, edge_feats)

        # Optionally apply softmax for 'global_sum_pooling'
        if self.readout_type == 'global_sum_pooling':
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        embeddings = self.ffn(molecular_encodings)
        return embeddings

########################################################################
# 3) MPNNPOM with different readout types
########################################################################
class MPNNPOM(nn.Module):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'classification',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr',
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True):
        """
        Supports three readout types: 'set2set', 'global_sum_pooling', and 'global_attention_pooling'.
        Uses CustomPositionwiseFeedForwardBN for batch-normalized feed-forward layers.
        """
        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'")

        super(MPNNPOM, self).__init__()

        self.n_tasks: int = n_tasks
        self.mode: str = mode
        self.n_classes: int = n_classes
        self.nfeat_name: str = nfeat_name
        self.efeat_name: str = efeat_name
        self.readout_type: str = readout_type
        self.ffn_embeddings: int = ffn_embeddings
        self.ffn_activation: str = ffn_activation
        self.ffn_dropout_p: float = ffn_dropout_p

        # Set model output dimension based on mode
        if mode == 'classification':
            self.ffn_output: int = n_tasks * n_classes
        else:
            self.ffn_output = n_tasks

        # MPNN module from external library (or local implementation)
        self.mpnn: nn.Module = CustomMPNNGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type
        )

        # Edge feature projection
        self.project_edge_feats: nn.Module = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats), nn.ReLU()
        )

        # Readout and setting FFN input dimensions
        if self.readout_type == 'set2set':
            self.readout_set2set: nn.Module = Set2Set(
                input_dim=node_out_feats + edge_out_feats,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set)
            ffn_input: int = 2 * (node_out_feats + edge_out_feats)

        elif self.readout_type == 'global_sum_pooling':
            ffn_input = node_out_feats + edge_out_feats

        elif self.readout_type == 'global_attention_pooling':
            # Gate MLP for attention
            self.attention_gate = nn.Sequential(
                nn.Linear(node_out_feats + edge_out_feats, 1),
                nn.Sigmoid()
            )
            self.global_attention_pool = GlobalAttentionPooling(self.attention_gate)
            ffn_input = node_out_feats + edge_out_feats
        else:
            raise Exception("readout_type invalid. Choose from [set2set, global_sum_pooling, global_attention_pooling]")

        # Final feed-forward network with batch normalization
        if ffn_embeddings is not None:
            d_hidden_list = ffn_hidden_list + [ffn_embeddings]
        else:
            d_hidden_list = ffn_hidden_list

        self.ffn: nn.Module = CustomPositionwiseFeedForwardBN(
            d_input=ffn_input,
            d_hidden_list=d_hidden_list,
            d_output=self.ffn_output,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act)

        # Save references for convenience
        self.n_layers: int = self.ffn.n_layers
        self.linears = self.ffn.linears
        self.dropouts = self.ffn.dropouts

        # External DGL graph stored for reuse
        self.g = None

        # Embedding layer wraps the MPNN, readout, and FFN
        self.get_embeddings: nn.Module = EmbeddingLayer(
            mpnn=self.mpnn, readout=self._readout, ffn=self.ffn,
            readout_type=self.readout_type
        )

    def _readout(self, node_encodings: torch.Tensor,
                 edge_feats: torch.Tensor) -> torch.Tensor:
        """
        Custom readout: combines atom and bond features then applies set2set, sum pooling, or attention pooling.
        """
        self.g.ndata['node_emb'] = node_encodings
        self.g.edata['edge_emb'] = self.project_edge_feats(edge_feats)

        def message_func(edges) -> Dict:
            src_msg: torch.Tensor = torch.cat(
                (edges.src['node_emb'], edges.data['edge_emb']), dim=1)
            return {'src_msg': src_msg}

        def reduce_func(nodes) -> Dict:
            # Sum over mailbox messages
            src_msg_sum: torch.Tensor = torch.sum(nodes.mailbox['src_msg'], dim=1)
            return {'src_msg_sum': src_msg_sum}

        # Send and receive messages on the graph
        self.g.send_and_recv(self.g.edges(), message_func=message_func, reduce_func=reduce_func)

        if self.readout_type == 'set2set':
            batch_mol_hidden_states: torch.Tensor = self.readout_set2set(self.g, self.g.ndata['src_msg_sum'])
        elif self.readout_type == 'global_sum_pooling':
            batch_mol_hidden_states = dgl.sum_nodes(self.g, 'src_msg_sum')
        elif self.readout_type == 'global_attention_pooling':
            batch_mol_hidden_states = self.global_attention_pool(self.g, self.g.ndata['src_msg_sum'])
        else:
            raise ValueError("Invalid readout type encountered in forward.")

        return batch_mol_hidden_states

    def load_graph_obj(self, g):
        self.mpnn.load_graph(g)
        self.g = g

    def multi_graph(self, num):
        """
        Batching hack for demonstration. Replicates the graph if needed.
        For real use, rely on dgl.batch in the data loader.
        """
        num = list(num.shape)[0]
        list_of_copies = [self.g] * num
        g = dgl.batch(list_of_copies)
        self.load_graph_obj(g)

    def forward(self, node_feats, edge_feats, multi) -> torch.Tensor:
        """
        Forward pass:
         1) Replicate the graph for the given mini-batch size.
         2) Get embeddings via MPNN, readout, and FFN.
         3) Return logits (for classification) or raw outputs.
        """
        self.multi_graph(multi)  # set self.g as batched graph
        embeddings = self.get_embeddings(node_feats, edge_feats)

        if self.mode == 'classification':
            if self.n_tasks == 1:
                return embeddings.view(-1, self.n_classes)
            else:
                return embeddings.view(-1, self.n_tasks, self.n_classes)
        else:
            return embeddings

    def get_penultimate_activations(self, g: DGLGraph) -> torch.Tensor:
        """
        Retrieves penultimate embeddings from the last FFN hidden layer.
        """
        node_feats: torch.Tensor = g.ndata[self.nfeat_name]
        edge_feats: torch.Tensor = g.edata[self.efeat_name]
        node_encodings: torch.Tensor = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings: torch.Tensor = self._readout(node_encodings, edge_feats)

        if self.readout_type == 'global_sum_pooling':
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        penultimate = molecular_encodings
        for i in range(self.ffn.n_layers - 1):
            penultimate = self.ffn.linears[i](penultimate)
            if i < self.ffn.n_layers - 2:
                penultimate = self.ffn.bns[i](penultimate)
                penultimate = self.ffn.activation_fn(penultimate)
                penultimate = self.ffn.dropouts[i](penultimate)
        return penultimate

########################################################################
# 4) Simple MLP to encode fingerprints
########################################################################
class FingerprintEncoder(nn.Module):
    def __init__(self, fp_dim: int, out_dim: int, dropout_p: float = 0.1):
        super(FingerprintEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.encoder(fp)

########################################################################
# 5) MPNNPOMWithFingerprint - Combining graph-based embeddings with fingerprint data
########################################################################
class MPNNPOMWithFingerprint(nn.Module):
    """
    Incorporates an additional fingerprint vector.
    """
    def __init__(self, base_mpnn_pom: MPNNPOM, fp_dim: int = 1024):
        super(MPNNPOMWithFingerprint, self).__init__()

        # Base graph model
        self.base_model = base_mpnn_pom
        self.load_graph_obj = self.base_model.load_graph_obj
        self.mode = self.base_model.mode
        self.n_classes = self.base_model.n_classes
        self.n_tasks = self.base_model.n_tasks

        # Fingerprint encoder dimension matching
        self.fp_dim = fp_dim
        self.fp_out_dim = self.base_model.ffn_embeddings or 128

        self.fp_encoder = FingerprintEncoder(
            fp_dim=self.fp_dim,
            out_dim=self.fp_out_dim,
            dropout_p=0.1  # example dropout
        )

        # Combine base model output and fingerprint embedding
        combined_in_dim = self.base_model.ffn_output + self.fp_out_dim
        self.combined_head = nn.Linear(combined_in_dim, self.base_model.ffn_output)

    def forward(self, node_feats, edge_feats, fp_vector, multi):
        """
        Args:
          node_feats (Tensor): Graph node features.
          edge_feats (Tensor): Graph edge features.
          fp_vector (Tensor): Fingerprint vector [batch_size, fp_dim].
          multi (Tensor): Used for batching the graph.
        """
        # Get base graph-based embeddings
        embeddings = self.base_model(node_feats, edge_feats, multi)
        # Encode fingerprint vector
        fp_encoded = self.fp_encoder(fp_vector).view(fp_vector.shape[0], self.fp_out_dim, 1)
        # Concatenate both embeddings
        combined = torch.cat([embeddings, fp_encoded], dim=1)
        combined = combined.squeeze(-1)
        # Final head produces output
        out = self.combined_head(combined)

        if self.mode == 'classification':
            if self.n_tasks == 1:
                return out.view(-1, self.n_classes)
            else:
                return out.view(-1, self.n_tasks, self.n_classes)
        else:
            return out

########################################################################
# 6) Model class derived from TorchModel (DeepChem integration)
########################################################################
class MPNNPOMModel(TorchModel):
    def __init__(self,
                 n_tasks: int,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr',
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = 'sum',
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 batch_size: int = 100,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'regression',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 weight_decay: float = 1e-5,
                 self_loop: bool = False,
                 optimizer_name: str = 'adam',
                 device_name: Optional[str] = None,
                 fp_dim=1024,
                 **kwargs):

        # Instantiate base model
        base_model: nn.Module = MPNNPOM(
            n_tasks=n_tasks,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            edge_out_feats=edge_out_feats,
            num_step_message_passing=num_step_message_passing,
            mpnn_residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            mode= mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes,
            readout_type=readout_type,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act
        )

        model = MPNNPOMWithFingerprint(base_model, fp_dim = fp_dim)

        if class_imbalance_ratio and (len(class_imbalance_ratio) != n_tasks):
            raise Exception("size of class_imbalance_ratio should match n_tasks")

        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types: List = ['prediction']
        else:
            loss = CustomMultiLabelLoss(
                class_imbalance_ratio=class_imbalance_ratio,
                loss_aggr_type=loss_aggr_type,
                device=device_name
            )
            output_types = ['loss']

        optimizer: Optimizer = get_optimizer(optimizer_name)
        optimizer.learning_rate = learning_rate
        device = torch.device(device_name) if device_name is not None else None

        super(MPNNPOMModel, self).__init__(
            model,
            loss=loss,
            output_types=output_types,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            **kwargs
        )

        self.weight_decay: float = weight_decay
        self._self_loop: bool = self_loop
        self.regularization_loss: Callable = self._regularization_loss
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name

    def _regularization_loss(self) -> torch.Tensor:
        """
        Computes L1 and L2 regularization losses.
        """
        l1_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        l2_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
                l2_regularization = l2_regularization + torch.norm(param, p=2)
        return self.weight_decay * (l1_regularization + l2_regularization)

    def _prepare_batch(self, batch: Tuple[List, List, List]
                       ) -> Tuple[DGLGraph, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepares batch data for MPNN.
        """
        inputs, labels, weights = batch
        dgl_graphs: List[DGLGraph] = [graph['graph'].to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]]
        g: DGLGraph = dgl.batch(dgl_graphs).to(self.device)
        node_feats: torch.Tensor = torch.tensor(g.ndata[self.nfeat_name], requires_grad=True)
        edge_feats: torch.Tensor = torch.tensor(g.edata[self.efeat_name], requires_grad=True)
        _, labels, weights = super(MPNNPOMModel, self)._prepare_batch(([], labels, weights))
        return g, node_feats, edge_feats, labels, weights, torch.tensor([graph['fp_vec'] for graph in inputs[0]], device=self.device, dtype=torch.float32)
        
    def forward(self, g: DGLGraph) -> torch.Tensor:
        """
        Forward pass for MPNNPOMModel.
        """
        return self.model(g)

########################################################################
# 7) Example main block
########################################################################
if __name__ == "__main__":
    # Here you would normally add your data loading, training loop, etc.
    # Example of constructing an MPNNPOMWithFingerprint model:
    # base_mpnn = MPNNPOM(n_tasks=3, mode='classification')
    # model_with_fp = MPNNPOMWithFingerprint(base_mpnn, fp_dim=1024)
    # Then use your training routine.
    pass
