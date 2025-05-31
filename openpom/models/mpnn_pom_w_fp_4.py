import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Callable, Dict

from deepchem.models.losses import Loss, L2Loss
# from deepchem.models.torch_models.torch_model import TorchModel
from openpom.models.torch_model5 import TorchModel # Assuming this is your custom TorchModel
from deepchem.models.optimizers import Optimizer, LearningRateSchedule

from openpom.layers.pom_ffn2 import CustomPositionwiseFeedForward
from openpom.utils.loss import CustomMultiLabelLoss
from openpom.utils.optimizer import get_optimizer

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn.pytorch import Set2Set
    from dgl.nn import GlobalAttentionPooling

    from openpom.layers.pom_mpnn_gnn2 import CustomMPNNGNN # This is the critical MPNNGNN
except (ImportError, ModuleNotFoundError):
    raise ImportError('This module requires dgl and dgllife')


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
        if fp.ndim == 1:
            fp = fp.unsqueeze(0)
        return self.encoder(fp)


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 mpnn: nn.Module, # Instance of CustomMPNNGNN
                 readout_module: nn.Module,
                 project_edge_feats_module: nn.Module,
                 readout_type: str,
                 fp_dim: int,
                 fp_embed_dim: int,
                 node_out_feats: int,
                 edge_out_feats: int,
                 ffn_hidden_list: List[int],
                 ffn_embeddings: int,
                 ffn_activation: str,
                 ffn_dropout_p: float,
                 ffn_dropout_at_input_no_act: bool):
        super(EmbeddingLayer, self).__init__()
        self.mpnn = mpnn
        self.readout_module = readout_module
        self.project_edge_feats_module = project_edge_feats_module
        self.readout_type = readout_type
        self.ffn_embeddings = ffn_embeddings
        self.fp_dim = fp_dim
        self.fp_embed_dim = fp_embed_dim

        if self.fp_dim > 0 :
            self.fp_encoder = FingerprintEncoder(
                fp_dim=fp_dim,
                out_dim=fp_embed_dim,
                dropout_p=0.1
            )
        else:
            self.fp_encoder = None

        if self.readout_type == 'set2set':
            gnn_readout_dim = 2 * (node_out_feats + edge_out_feats)
        elif self.readout_type in ['global_sum_pooling', 'global_attention_pooling']:
            gnn_readout_dim = node_out_feats + edge_out_feats
        else:
            raise ValueError(f"Unsupported readout_type: {self.readout_type}")

        current_fp_embed_dim = fp_embed_dim if self.fp_encoder is not None else 0
        combined_ffn_input_dim = gnn_readout_dim + current_fp_embed_dim

        self.combined_ffn = CustomPositionwiseFeedForward(
            d_input=combined_ffn_input_dim,
            d_hidden_list=ffn_hidden_list,
            d_output=self.ffn_embeddings,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act
        )

    def _internal_readout(self, g: DGLGraph, node_encodings: torch.Tensor,
                          edge_feats_input: torch.Tensor) -> torch.Tensor:
        g.ndata['node_emb'] = node_encodings
        g.edata['edge_emb'] = self.project_edge_feats_module(edge_feats_input)

        def message_func(edges) -> Dict:
            src_msg: torch.Tensor = torch.cat(
                (edges.src['node_emb'], edges.data['edge_emb']), dim=1)
            return {'src_msg': src_msg}

        def reduce_func(nodes) -> Dict:
            src_msg_sum: torch.Tensor = torch.sum(nodes.mailbox['src_msg'], dim=1)
            return {'src_msg_sum': src_msg_sum}

        g.send_and_recv(g.edges(), message_func=message_func, reduce_func=reduce_func)
        node_plus_edge_feats_sum = g.ndata['src_msg_sum']

        if self.readout_type == 'set2set':
            molecular_encodings = self.readout_module(g, node_plus_edge_feats_sum)
        elif self.readout_type == 'global_sum_pooling':
            molecular_encodings = dgl.sum_nodes(g, 'src_msg_sum')
            molecular_encodings = F.softmax(molecular_encodings, dim=1)
        elif self.readout_type == 'global_attention_pooling':
            molecular_encodings = self.readout_module(g, node_plus_edge_feats_sum)
        else:
            raise ValueError(f"Unsupported readout_type: {self.readout_type}")
        return molecular_encodings

    def forward(self, g: DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor, fp_vector: Optional[torch.Tensor]) -> torch.Tensor:
        # CRITICAL CHANGE HERE:
        # Assuming self.mpnn (CustomMPNNGNN) has its graph loaded via a 'load_graph' method
        # and its forward signature is forward(self, node_feats, edge_feats)
        node_encodings = self.mpnn(node_feats, edge_feats)

        # The graph 'g' passed to this forward method is used for the readout part
        molecular_gnn_representation = self._internal_readout(g, node_encodings, edge_feats)

        if self.fp_encoder is not None and fp_vector is not None:
            if fp_vector.shape[1] != self.fp_dim :
                 raise ValueError(f"Fingerprint vector dimension mismatch. Expected {self.fp_dim}, got {fp_vector.shape[1]}")
            fp_encoded = self.fp_encoder(fp_vector)
        else:
            batch_size = molecular_gnn_representation.shape[0]
            fp_encoded = torch.zeros(batch_size, self.fp_embed_dim, device=molecular_gnn_representation.device)

        if fp_encoded.shape[0] != molecular_gnn_representation.shape[0]:
            if fp_encoded.shape[0] == 1 and molecular_gnn_representation.shape[0] > 1 :
                fp_encoded = fp_encoded.repeat(molecular_gnn_representation.shape[0], 1)
            else:
                raise ValueError(f"Batch size mismatch between GNN ({molecular_gnn_representation.shape[0]}) and FP ({fp_encoded.shape[0]}) representations after potential repeat.")

        combined_representation = torch.cat([molecular_gnn_representation, fp_encoded], dim=1)
        embeddings = self.combined_ffn(combined_representation)
        return embeddings


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
                 number_atom_features: int = 134, # Stored for reference if CustomMPNNGNN needs it
                 number_bond_features: int = 6, # Stored for reference
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
                 ffn_dropout_at_input_no_act: bool = True,
                 fp_dim: int = 1024,
                 fp_embed_dim: int = 128):
        super(MPNNPOM, self).__init__()

        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'")

        self.n_tasks: int = n_tasks
        self.mode: str = mode
        self.n_classes: int = n_classes
        self.ffn_embeddings: int = ffn_embeddings
        self.fp_dim = fp_dim
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        # Store these if needed for empty tensor creation in _prepare_batch
        self.number_atom_features = number_atom_features
        self.number_bond_features = number_bond_features


        if mode == 'classification':
            self.ffn_output_dim: int = n_tasks * n_classes
        else:
            self.ffn_output_dim = n_tasks

        # This is the CustomMPNNGNN instance
        mpnn_actual_module: nn.Module = CustomMPNNGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type)

        project_edge_feats_module: nn.Module = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats), nn.ReLU())

        gnn_readout_input_dim = node_out_feats + edge_out_feats
        if readout_type == 'set2set':
            readout_module_instance: nn.Module = Set2Set(
                input_dim=gnn_readout_input_dim,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set)
        elif readout_type == 'global_sum_pooling':
            readout_module_instance = None
        elif readout_type == 'global_attention_pooling':
            attention_gate = nn.Sequential(
                nn.Linear(gnn_readout_input_dim, 1),
                nn.Sigmoid())
            readout_module_instance = GlobalAttentionPooling(attention_gate)
        else:
            raise ValueError(f"Unsupported readout_type: {readout_type}")

        self.get_embeddings_layer: nn.Module = EmbeddingLayer(
            mpnn=mpnn_actual_module, # Pass the CustomMPNNGNN instance
            readout_module=readout_module_instance,
            project_edge_feats_module=project_edge_feats_module,
            readout_type=readout_type,
            fp_dim=fp_dim,
            fp_embed_dim=fp_embed_dim,
            node_out_feats=node_out_feats,
            edge_out_feats=edge_out_feats,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=self.ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act
        )
        print(f"[MPNNPOM INIT] self.ffn_embeddings (for pred_head_linear input): {self.ffn_embeddings}")
        print(f"[MPNNPOM INIT] self.ffn_output_dim (for pred_head_linear output): {self.ffn_output_dim}")
        # self.pred_head_linear = nn.Linear(self.ffn_embeddings, self.ffn_output_dim)

        if ffn_activation == 'relu': self.pred_head_activation = nn.ReLU()
        elif ffn_activation == 'leakyrelu': self.pred_head_activation = nn.LeakyReLU(0.1)
        elif ffn_activation == 'prelu': self.pred_head_activation = nn.PReLU()
        elif ffn_activation == 'tanh': self.pred_head_activation = nn.Tanh()
        elif ffn_activation == 'selu': self.pred_head_activation = nn.SELU()
        elif ffn_activation == 'elu': self.pred_head_activation = nn.ELU()
        elif ffn_activation == 'linear': self.pred_head_activation = lambda x: x
        else: self.pred_head_activation = nn.ReLU()

        self.pred_head_dropout = nn.Dropout(ffn_dropout_p)
        self.pred_head_linear = nn.Linear(self.ffn_embeddings, self.ffn_output_dim)

        self.g: Optional[DGLGraph] = None # Stores template graph for MPNNPOM itself, and current batched graph during forward
        self._template_loaded: bool = False

    def load_graph_obj(self, g_template: DGLGraph):
        self.g = g_template # Store template in MPNNPOM
        self._template_loaded = True
        # Also load template into the CustomMPNNGNN instance
        mpnn_module = self.get_embeddings_layer.mpnn
        if hasattr(mpnn_module, 'load_graph'):
            mpnn_module.load_graph(g_template.to(next(mpnn_module.parameters()).device if list(mpnn_module.parameters()) else torch.device("cpu"))) # Ensure on correct device
        elif hasattr(mpnn_module, 'g'): # Fallback if it just has a 'g' attribute
             mpnn_module.g = g_template.to(next(mpnn_module.parameters()).device if list(mpnn_module.parameters()) else torch.device("cpu"))


    def multi_graph(self, num_or_tensor_shape_provider: Union[torch.Tensor, int]):
        if not self._template_loaded or self.g is None:
            raise RuntimeError("Template graph not loaded. Call load_graph_obj first.")

        if isinstance(num_or_tensor_shape_provider, torch.Tensor):
            num = num_or_tensor_shape_provider.shape[0]
            device = num_or_tensor_shape_provider.device
        elif isinstance(num_or_tensor_shape_provider, int):
            num = num_or_tensor_shape_provider
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            raise TypeError("multi_graph expects a tensor or an int for num_of_copies")

        if num < 1: # Allow num=0 for empty batches if dgl.batch([]) works
            if num == 0:
                self.g = dgl.batch([]) # Create an empty batched graph
                # Also load this empty batched graph into CustomMPNNGNN
                mpnn_module = self.get_embeddings_layer.mpnn
                if hasattr(mpnn_module, 'load_graph'):
                    mpnn_module.load_graph(self.g.to(device))
                elif hasattr(mpnn_module, 'g'):
                    mpnn_module.g = self.g.to(device)
                return # Exit early for empty batch
            raise ValueError("multi_graph expects num_of_copies to be non-negative.")

        # self.g currently holds the template graph. Ensure it's on the target device for batching.
        template_g_on_device = self.g.to(device)
        
        list_of_copies = [template_g_on_device] * num
        batched_g = dgl.batch(list_of_copies)
        self.g = batched_g # MPNNPOM's self.g is now the batched graph for this forward call

        # Load the batched graph into the CustomMPNNGNN instance
        mpnn_module = self.get_embeddings_layer.mpnn
        if hasattr(mpnn_module, 'load_graph'):
            mpnn_module.load_graph(self.g) # self.g is already on the correct device
        elif hasattr(mpnn_module, 'g'): # Fallback
            mpnn_module.g = self.g


    def get_logits_from_output(self, output_tensor: torch.Tensor) -> torch.Tensor:
        if self.n_tasks == 1:
            logits: torch.Tensor = output_tensor.view(-1, self.n_classes)
        else:
            logits = output_tensor.view(-1, self.n_tasks, self.n_classes)
        return logits

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor, fp_vec: torch.Tensor, multi: torch.Tensor) -> torch.Tensor:
        if not self._template_loaded:
            raise RuntimeError("MPNNPOM template graph not loaded.")

        self.multi_graph(multi) # Sets self.g (MPNNPOM's graph) and loads graph into CustomMPNNGNN

        if self.g.batch_size == 0 and node_feats.shape[0] == 0 : # Handle empty batch
             # Output should be (0, n_tasks * n_classes) or (0, n_tasks)
             output_shape_feat_dim = self.ffn_output_dim
             return torch.empty(0, output_shape_feat_dim, device=node_feats.device)


        # self.g is batched, CustomMPNNGNN has self.g loaded.
        # EmbeddingLayer's self.mpnn will use its internally loaded graph.
        # EmbeddingLayer's _internal_readout will use MPNNPOM's self.g.
        embeddings = self.get_embeddings_layer(self.g, node_feats, edge_feats, fp_vec)

        x = self.pred_head_dropout(self.pred_head_activation(embeddings))
        out = self.pred_head_linear(x)

        if self.mode == 'classification':
            logits = self.get_logits_from_output(output_tensor=out)
            return logits
        else:
            return out

    def get_penultimate_activations(self, g_input: DGLGraph, fp_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Determine device from model parameters
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu") # Default if model has no parameters or not on device

        # Prepare graph and features for a single instance
        g_input_batched = dgl.batch([g_input.to(model_device)])
        node_feats = g_input_batched.ndata[self.nfeat_name].to(model_device)
        edge_feats = g_input_batched.edata[self.efeat_name].to(model_device)

        if fp_vec is not None:
            fp_vec = fp_vec.to(model_device)
            if fp_vec.ndim == 1:
                fp_vec = fp_vec.unsqueeze(0)

        # Manage CustomMPNNGNN's graph state
        mpnn_module = self.get_embeddings_layer.mpnn
        original_mpnn_graph_state = None
        has_load_graph = hasattr(mpnn_module, 'load_graph')
        has_g_attr = hasattr(mpnn_module, 'g')

        if has_g_attr: # Save current graph state of CustomMPNNGNN
            original_mpnn_graph_state = mpnn_module.g
        
        # Load the new graph for this specific call
        if has_load_graph:
            mpnn_module.load_graph(g_input_batched)
        elif has_g_attr: # Fallback if no load_graph, try setting .g
             mpnn_module.g = g_input_batched
        # If neither, CustomMPNNGNN might not support this dynamic graph change well for this function.

        # Call EmbeddingLayer; it will use g_input_batched for readout,
        # and mpnn_module will use its (now updated) internal graph.
        embeddings = self.get_embeddings_layer(g_input_batched, node_feats, edge_feats, fp_vec)

        # Restore CustomMPNNGNN's original graph state
        if original_mpnn_graph_state is not None:
            if has_load_graph:
                mpnn_module.load_graph(original_mpnn_graph_state)
            elif has_g_attr:
                 mpnn_module.g = original_mpnn_graph_state
        
        return embeddings.squeeze(0)


class MPNNPOMModel(TorchModel):
    def __init__(self,
                 n_tasks: int,
                 fp_embed_dim: int = 128,
                 fp_dim: int = 1024,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr',
                 # ... (rest of the parameters are the same)
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
                 **kwargs):

        model_instance: nn.Module = MPNNPOM(
            n_tasks=n_tasks,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            edge_out_feats=edge_out_feats,
            num_step_message_passing=num_step_message_passing,
            mpnn_residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            mode=mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes,
            nfeat_name=nfeat_name,
            efeat_name=efeat_name,
            readout_type=readout_type,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act,
            fp_dim=fp_dim,
            fp_embed_dim=fp_embed_dim
        )

        if class_imbalance_ratio and (len(class_imbalance_ratio) != n_tasks):
            raise Exception("size of class_imbalance_ratio should be equal to n_tasks")

        actual_device_for_loss = torch.device(device_name) if device_name else None
        if mode == 'regression':
            loss_fn: Loss = L2Loss()
            output_types: List = ['prediction']
        else:
            loss_fn = CustomMultiLabelLoss(
                class_imbalance_ratio=class_imbalance_ratio,
                loss_aggr_type=loss_aggr_type,
                device=actual_device_for_loss)
            output_types = ['prediction']

        optimizer_instance: Optimizer = get_optimizer(optimizer_name)
        actual_device = torch.device(device_name) if device_name else None

        super(MPNNPOMModel, self).__init__(model_instance,
                                           loss=loss_fn,
                                           output_types=output_types,
                                           optimizer=optimizer_instance,
                                           learning_rate=learning_rate,
                                           batch_size=batch_size,
                                           device=actual_device,
                                           **kwargs)

        self.weight_decay: float = weight_decay
        self._self_loop: bool = self_loop
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        # self._mpnnpom_template_graph_loaded flag is internal to MPNNPOM instance (self.model._template_loaded)

    def _regularization_loss(self) -> torch.Tensor:
        l1_regularization: torch.Tensor = torch.tensor(0., requires_grad=True).to(self.device)
        l2_regularization: torch.Tensor = torch.tensor(0., requires_grad=True).to(self.device)
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
                l2_regularization = l2_regularization + torch.norm(param, p=2)
        l1_norm: torch.Tensor = self.weight_decay * l1_regularization
        l2_norm: torch.Tensor = self.weight_decay * l2_regularization
        return l1_norm + l2_norm

    def _prepare_batch(
            self, batch: Tuple[List, List, List]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        inputs_dc, labels_dc, weights_dc = batch

        dgl_graphs_list: List[DGLGraph] = [
            item['graph'].to_dgl_graph(self_loop=self._self_loop) for item in inputs_dc[0]
        ]

        if not dgl_graphs_list: # Handle empty batch from iterator
             # Use properties from the model instance (MPNNPOM) to get feature dimensions
             # Ensure model is MPNNPOM to access these properties
             mpnnpom_instance = self.model
             node_in_feats_dim = mpnnpom_instance.number_atom_features if hasattr(mpnnpom_instance, 'number_atom_features') else 0
             edge_in_feats_dim = mpnnpom_instance.number_bond_features if hasattr(mpnnpom_instance, 'number_bond_features') else 0
             fp_actual_dim = mpnnpom_instance.fp_dim if hasattr(mpnnpom_instance, 'fp_dim') else 0

             empty_node_feats = torch.empty(0, node_in_feats_dim, device=self.device)
             empty_edge_feats = torch.empty(0, edge_in_feats_dim, device=self.device)
             empty_fp_vec = torch.empty(0, fp_actual_dim, device=self.device) # Use actual fp_dim
             empty_multi_indicator = empty_node_feats
             
             _, processed_labels, processed_weights = super(MPNNPOMModel, self)._prepare_batch(
                ([], [], []))

             return [empty_node_feats, empty_edge_feats, empty_fp_vec, empty_multi_indicator], processed_labels, processed_weights

        if not self.model._template_loaded:
            template_g = dgl_graphs_list[0].cpu()
            self.model.load_graph_obj(template_g)

        current_batch_g: DGLGraph = dgl.batch(dgl_graphs_list).to(self.device)
        node_feats = current_batch_g.ndata[self.nfeat_name].to(self.device)
        edge_feats = current_batch_g.edata[self.efeat_name].to(self.device)

        fp_vectors_list = [torch.tensor(item['fp_vec'], dtype=torch.float32) for item in inputs_dc[0]]
        fp_tensor = torch.stack(fp_vectors_list).to(self.device)
        multi_indicator = node_feats

        _, processed_labels, processed_weights = super(MPNNPOMModel, self)._prepare_batch(
            ([], labels_dc, weights_dc)
        )
        model_inputs = [node_feats, edge_feats, fp_tensor, multi_indicator]
        return current_batch_g, node_feats, edge_feats, processed_labels, processed_weights, fp_tensor