import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Callable, Dict

from deepchem.models.losses import Loss, L2Loss
# from deepchem.models.torch_models.torch_model import TorchModel
from  openpom.models.torch_model2 import TorchModel
# from openpom.models.mpnn_pom2 import TorchModel

from deepchem.models.optimizers import Optimizer, LearningRateSchedule

from openpom.layers.pom_ffn2 import CustomPositionwiseFeedForward
from openpom.utils.loss import CustomMultiLabelLoss
from openpom.utils.optimizer import get_optimizer

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn.pytorch import Set2Set
    from openpom.layers.pom_mpnn_gnn import CustomMPNNGNN
except (ImportError, ModuleNotFoundError):
    raise ImportError('This module requires dgl and dgllife')


class MPNNPOM(nn.Module):
    """
    MPNN model computes a principal odor map
    using multilabel-classification based on the pre-print:
    "A Principal Odor Map Unifies DiverseTasks in Human
        Olfactory Perception" [1]

    This model proceeds as follows:

    * Combine latest node representations and edge features in
        updating node representations, which involves multiple
        rounds of message passing.
    * For each graph, compute its representation by radius 0 combination
        to fold atom and bond embeddings together, followed by
        'set2set' or 'global_sum_pooling' readout.
    * Perform the final prediction using a feed-forward layer.

    References
    ----------
    .. [1] Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling,
        Jennifer N. Wei, Wesley W. Qian, Kelsie Little, Matthew Andres,
        Britney B. Nguyen, Theresa Moloy, Jane K. Parker, Richard C. Gerkin,
        Joel D. Mainland, Alexander B. Wiltschko
        `A Principal Odor Map Unifies Diverse Tasks
        in Human Olfactory Perception preprint
        <https://www.biorxiv.org/content/10.1101/2022.09.01.504602v4>`_.

    .. [2] Benjamin Sanchez-Lengeling, Jennifer N. Wei, Brian K. Lee,
        Richard C. Gerkin, Alán Aspuru-Guzik, Alexander B. Wiltschko
        `Machine Learning for Scent:
        Learning Generalizable Perceptual Representations
        of Small Molecules <https://arxiv.org/abs/1910.10685>`_.

    .. [3] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley,
        Oriol Vinyals, George E. Dahl.
        "Neural Message Passing for Quantum Chemistry." ICML 2017.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl)
    and DGL-LifeSci (https://github.com/awslabs/dgl-lifesci)
    to be installed.
    """

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
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        node_out_feats: int
            The length of the final node representation vectors
            before readout. Default to 64.
        edge_hidden_feats: int
            The length of the hidden edge representation vectors
            for mpnn edge network. Default to 128.
        edge_out_feats: int
            The length of the final edge representation vectors
            before readout. Default to 64.
        num_step_message_passing: int
            The number of rounds of message passing. Default to 3.
        mpnn_residual: bool
            If true, adds residual layer to mpnn layer. Default to True.
        message_aggregator_type: str
            MPNN message aggregator type, 'sum', 'mean' or 'max'.
            Default to 'sum'.
        mode: str
            The model type, 'classification' or 'regression'.
            Default to 'classification'.
        number_atom_features: int
            The length of the initial atom feature vectors. Default to 134.
        number_bond_features: int
            The length of the initial bond feature vectors. Default to 6.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification'). Default to 1.
        nfeat_name: str
            For an input graph ``g``, the model assumes that it stores
            node features in ``g.ndata[nfeat_name]`` and will retrieve
            input node features from that. Default to 'x'.
        efeat_name: str
            For an input graph ``g``, the model assumes that it stores
            edge features in ``g.edata[efeat_name]`` and will retrieve
            input edge features from that. Default to 'edge_attr'.
        readout_type: str
            The Readout type, 'set2set' or 'global_sum_pooling'.
            Default to 'set2set'.
        num_step_set2set: int
            Number of steps in set2set readout.
            Used if, readout_type == 'set2set'.
            Default to 6.
        num_layer_set2set: int
            Number of layers in set2set readout.
            Used if, readout_type == 'set2set'.
            Default to 3.
        ffn_hidden_list: List
            List of sizes of hidden layer in the feed-forward network layer.
            Default to [300].
        ffn_embeddings: int
            Size of penultimate layer in the feed-forward network layer.
            This determines the Principal Odor Map dimension.
            Default to 256.
        ffn_activation: str
            Activation function to be used in feed-forward network layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU,
            'prelu' for PReLU, 'tanh' for TanH, 'selu' for SELU,
            and 'elu' for ELU.
        ffn_dropout_p: float
            Dropout probability for the feed-forward network layer.
            Default to 0.0
        ffn_dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor.
            For single layer, it is not passed to an activation function.
        """
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

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

        if mode == 'classification':
            self.ffn_output: int = n_tasks * n_classes
        else:
            self.ffn_output = n_tasks

        self.mpnn: nn.Module = CustomMPNNGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type)

        self.project_edge_feats: nn.Module = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats), nn.ReLU())

        if self.readout_type == 'set2set':
            self.readout_set2set: nn.Module = Set2Set(
                input_dim=node_out_feats + edge_out_feats,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set)
            ffn_input: int = 2 * (node_out_feats + edge_out_feats)
        elif self.readout_type == 'global_sum_pooling':
            ffn_input = node_out_feats + edge_out_feats
        else:
            raise Exception("readout_type invalid")

        if ffn_embeddings is not None:
            d_hidden_list: List = ffn_hidden_list + [ffn_embeddings]

        self.ffn: nn.Module = CustomPositionwiseFeedForward(
            d_input=ffn_input,
            d_hidden_list=d_hidden_list,
            d_output=self.ffn_output,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act)

        if ffn_activation == 'relu':
            self.activation = nn.ReLU()

        elif ffn_activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif ffn_activation == 'prelu':
            self.activation = nn.PReLU()

        elif ffn_activation == 'tanh':
            self.activation = nn.Tanh()

        elif ffn_activation == 'selu':
            self.activation = nn.SELU()

        elif ffn_activation == 'elu':
            self.activation = nn.ELU()

        elif ffn_activation == 'linear':
            self.activation = lambda x: x

        self.n_layers: int = len(d_hidden_list) + 1

        # Set linear layers
        if self.n_layers == 1:
            linears: List = [nn.Linear(ffn_input, self.ffn_output)]

        else:
            linears = [nn.Linear(ffn_input, d_hidden_list[0])]
            for idx in range(1, len(d_hidden_list)):
                linears.append(
                    nn.Linear(d_hidden_list[idx - 1], d_hidden_list[idx]))
            linears.append(nn.Linear(d_hidden_list[-1], self.ffn_output))

        self.linears: nn.ModuleList = nn.ModuleList(linears)
        dropout_layer: nn.Dropout = nn.Dropout(ffn_dropout_p)
        self.dropout_p: nn.ModuleList = nn.ModuleList(
            [dropout_layer for _ in range(self.n_layers)])

    def _readout(self, g: DGLGraph, node_encodings: torch.Tensor,
                 edge_feats: torch.Tensor) -> torch.Tensor:
        """
        Method to execute the readout phase.
        (compute molecules encodings from atom hidden states)

        Readout phase consists of radius 0 combination to fold atom
        and bond embeddings together,
        followed by:
            - a reduce-sum across atoms
                if `self.readout_type == 'global_sum_pooling'`
            - set2set pooling
                if `self.readout_type == 'set2set'`

        Parameters
        ----------
        g: DGLGraph
            A DGLGraph for a batch of graphs.
            It stores the node features in
            ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
            ``dgl_graph.edata[self.efeat_name]``.

        node_encodings: torch.Tensor
            Tensor containing node hidden states.

        edge_feats: torch.Tensor
            Tensor containing edge features.

        Returns
        -------
        batch_mol_hidden_states: torch.Tensor
            Tensor containing batchwise molecule encodings.
        """

        g.ndata['node_emb'] = node_encodings
        g.edata['edge_emb'] = self.project_edge_feats(edge_feats)

        def message_func(edges) -> Dict:
            """
            The message function to generate messages
            along the edges for DGLGraph.send_and_recv()
            """
            src_msg: torch.Tensor = torch.cat(
                (edges.src['node_emb'], edges.data['edge_emb']), dim=1)
            return {'src_msg': src_msg}

        def reduce_func(nodes) -> Dict:
            """
            The reduce function to aggregate the messages
            for DGLGraph.send_and_recv()
            """
            src_msg_sum: torch.Tensor = torch.sum(nodes.mailbox['src_msg'],
                                                  dim=1)
            return {'src_msg_sum': src_msg_sum}

        # radius 0 combination to fold atom and bond embeddings together
        g.send_and_recv(g.edges(),
                        message_func=message_func,
                        reduce_func=reduce_func)

        if self.readout_type == 'set2set':
            batch_mol_hidden_states: torch.Tensor = self.readout_set2set(
                g, g.ndata['src_msg_sum'])
        elif self.readout_type == 'global_sum_pooling':
            batch_mol_hidden_states = dgl.sum_nodes(g, 'src_msg_sum')

        # batch_size x (node_out_feats + edge_out_feats)
        return batch_mol_hidden_states

    def get_embeddings(self, g):
        node_feats: torch.Tensor = g.ndata[self.nfeat_name]
        edge_feats: torch.Tensor = g.edata[self.efeat_name]

        node_encodings: torch.Tensor = self.mpnn(g, node_feats, edge_feats)

        molecular_encodings: torch.Tensor = self._readout(
            g, node_encodings, edge_feats)
        if self.readout_type == 'global_sum_pooling':
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        embeddings: torch.Tensor
        embeddings = self.ffn(molecular_encodings)
        return embeddings

    def get_logits_from_output(self, output):
        if self.n_tasks == 1:
            logits: torch.Tensor = output.view(-1, self.n_classes)
        else:
            logits = output.view(-1, self.n_tasks, self.n_classes)

        return logits

    def forward(
            self, g: DGLGraph
    ) -> torch.Tensor:
        """
        Foward pass for MPNNPOM class. It also returns embeddings for POM.

        Parameters
        ----------
        g: DGLGraph
            A DGLGraph for a batch of graphs. It stores the node features in
            ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
            ``dgl_graph.edata[self.efeat_name]``.

        Returns
        -------
        Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            The model output.

        * When self.mode = 'regression',
            its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
        * When self.mode = 'classification',
            the output consists of probabilities for classes.
            Its shape will be
            ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)``
            if self.n_tasks > 1;
            its shape will be ``(dgl_graph.batch_size, self.n_classes)``
            if self.n_tasks is 1.
        """
        embeddings = self.get_embeddings(g)
        x = self.dropout_p[self.n_layers - 2](
            self.activation(embeddings))
        out = self.linears[-1](x)

        if self.mode == 'classification':
            logits = self.get_logits_from_output(output=out)
            # proba: torch.Tensor = F.sigmoid(logits)  # (batch, n_tasks, classes)
            # if self.n_classes == 1:
            #     proba = proba.squeeze(-1)  # (batch, n_tasks)

            # pdb.set_trace()
            return logits  # proba, logits, embeddings
        else:
            return out

    def get_penultimate_activations(self, g: DGLGraph) -> torch.Tensor:
        node_feats: torch.Tensor = g.ndata[self.nfeat_name]
        edge_feats: torch.Tensor = g.edata[self.efeat_name]

        node_encodings: torch.Tensor = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings: torch.Tensor = self._readout(g, node_encodings, edge_feats)

        if self.readout_type == 'global_sum_pooling':
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        embeddings = self.ffn(molecular_encodings)
        return embeddings


class MPNNPOMModel(TorchModel):
    """
    MPNNPOMModel for obtaining a principal odor map
    using multilabel-classification based on the pre-print:
    "A Principal Odor Map Unifies DiverseTasks in Human
        Olfactory Perception" [1]

    * Combine latest node representations and edge features in
        updating node representations, which involves multiple
        rounds of message passing.
    * For each graph, compute its representation by radius 0 combination
        to fold atom and bond embeddings together, followed by
        'set2set' or 'global_sum_pooling' readout.
    * Perform the final prediction using a feed-forward layer.

    References
    ----------
    .. [1] Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling,
        Jennifer N. Wei, Wesley W. Qian, Kelsie Little, Matthew Andres,
        Britney B. Nguyen, Theresa Moloy, Jane K. Parker, Richard C. Gerkin,
        Joel D. Mainland, Alexander B. Wiltschko
        `A Principal Odor Map Unifies Diverse Tasks
        in Human Olfactory Perception preprint
        <https://www.biorxiv.org/content/10.1101/2022.09.01.504602v4>`_.

    .. [2] Benjamin Sanchez-Lengeling, Jennifer N. Wei, Brian K. Lee,
        Richard C. Gerkin, Alán Aspuru-Guzik, Alexander B. Wiltschko
        `Machine Learning for Scent:
        Learning Generalizable Perceptual Representations
        of Small Molecules <https://arxiv.org/abs/1910.10685>`_.

    .. [3] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley,
        Oriol Vinyals, George E. Dahl.
        "Neural Message Passing for Quantum Chemistry." ICML 2017.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
    (https://github.com/awslabs/dgl-lifesci) to be installed.

    The featurizer used with MPNNPOMModel must produce a Deepchem GraphData
    object which should have both 'edge' and 'node' features.
    """

    def __init__(self,
                 n_tasks: int,
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
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        class_imbalance_ratio: Optional[List]
            List of imbalance ratios per task.
        loss_aggr_type: str
            loss aggregation type; 'sum' or 'mean'. Default to 'sum'.
            Only applies to CustomMultiLabelLoss for classification
        learning_rate: Union[float, LearningRateSchedule]
            Learning rate value or scheduler object. Default to 0.001.
        batch_size: int
            Batch size for training. Default to 100.
        node_out_feats: int
            The length of the final node representation vectors
            before readout. Default to 64.
        edge_hidden_feats: int
            The length of the hidden edge representation vectors
            for mpnn edge network. Default to 128.
        edge_out_feats: int
            The length of the final edge representation vectors
            before readout. Default to 64.
        num_step_message_passing: int
            The number of rounds of message passing. Default to 3.
        mpnn_residual: bool
            If true, adds residual layer to mpnn layer. Default to True.
        message_aggregator_type: str
            MPNN message aggregator type, 'sum', 'mean' or 'max'.
            Default to 'sum'.
        mode: str
            The model type, 'classification' or 'regression'.
            Default to 'classification'.
        number_atom_features: int
            The length of the initial atom feature vectors. Default to 134.
        number_bond_features: int
            The length of the initial bond feature vectors. Default to 6.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification'). Default to 1.
        readout_type: str
            The Readout type, 'set2set' or 'global_sum_pooling'.
            Default to 'set2set'.
        num_step_set2set: int
            Number of steps in set2set readout.
            Used if, readout_type == 'set2set'.
            Default to 6.
        num_layer_set2set: int
            Number of layers in set2set readout.
            Used if, readout_type == 'set2set'.
            Default to 3.
        ffn_hidden_list: List
            List of sizes of hidden layer in the feed-forward network layer.
            Default to [300].
        ffn_embeddings: int
            Size of penultimate layer in the feed-forward network layer.
            This determines the Principal Odor Map dimension.
            Default to 256.
        ffn_activation: str
            Activation function to be used in feed-forward network layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU,
            'prelu' for PReLU, 'tanh' for TanH, 'selu' for SELU,
            and 'elu' for ELU.
        ffn_dropout_p: float
            Dropout probability for the feed-forward network layer.
            Default to 0.0
        ffn_dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor.
            For single layer, it is not passed to an activation function.
        weight_decay: float
            weight decay value for L1 and L2 regularization. Default to 1e-5.
        self_loop: bool
            Whether to add self loops for the nodes, i.e. edges
            from nodes to themselves. Generally, an MPNNPOMModel
            does not require self loops. Default to False.
        optimizer_name: str
            Name of optimizer to be used from
            [adam, adagrad, adamw, sparseadam, rmsprop, sgd, kfac]
            Default to 'adam'.
        device_name: Optional[str]
            The device on which to run computations. If None, a device is
            chosen automatically.
        kwargs
            This can include any keyword argument of TorchModel.
        """
        model: nn.Module = MPNNPOM(
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
            readout_type=readout_type,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act)

        if class_imbalance_ratio and (len(class_imbalance_ratio) != n_tasks):
            raise Exception("size of class_imbalance_ratio \
                            should be equal to n_tasks")

        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types: List = ['prediction']
        else:
            loss = CustomMultiLabelLoss(
                class_imbalance_ratio=class_imbalance_ratio,
                loss_aggr_type=loss_aggr_type,
                device=device_name)
            output_types = ['loss'] #, 'loss', 'embedding']

        optimizer: Optimizer = get_optimizer(optimizer_name)
        optimizer.learning_rate = learning_rate
        if device_name is not None:
            device: Optional[torch.device] = torch.device(device_name)
        else:
            device = None
        super(MPNNPOMModel, self).__init__(model,
                                           loss=loss,
                                           output_types=output_types,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           batch_size=batch_size,
                                           device=device,
                                           **kwargs)

        self.weight_decay: float = weight_decay
        self._self_loop: bool = self_loop
        self.regularization_loss: Callable = self._regularization_loss

    def _regularization_loss(self) -> torch.Tensor:
        """
        L1 and L2-norm losses for regularization

        Returns
        -------
        torch.Tensor
            sum of l1_norm and l2_norm
        """
        l1_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        l2_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
                l2_regularization = l2_regularization + torch.norm(param, p=2)
        l1_norm: torch.Tensor = self.weight_decay * l1_regularization
        l2_norm: torch.Tensor = self.weight_decay * l2_regularization
        return l1_norm + l2_norm

    def _prepare_batch(
            self, batch: Tuple[List, List, List]
    ) -> Tuple[DGLGraph, List[torch.Tensor], List[torch.Tensor]]:
        """Create batch data for MPNN.

        Parameters
        ----------
        batch: Tuple[List, List, List]
            The tuple is ``(inputs, labels, weights)``.

        Returns
        -------
        g: DGLGraph
            DGLGraph for a batch of graphs.
        labels: list of torch.Tensor or None
            The graph labels.
        weights: list of torch.Tensor or None
            The weights for each sample or
            sample/task pair converted to torch.Tensor.
        """
        inputs: List
        labels: List
        weights: List

        inputs, labels, weights = batch
        dgl_graphs: List[DGLGraph] = [
            graph.to_dgl_graph(self_loop=self._self_loop)
            for graph in inputs[0]
        ]
        g: DGLGraph = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(MPNNPOMModel, self)._prepare_batch(
            ([], labels, weights))
        return g, labels, weights

    def forward(self, g: DGLGraph) -> torch.Tensor:
        """
        Forward pass for MPNNPOMModel.
        """
        return self.model(g)

if __name__ == "__main__":

    TASKS = [
        'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
        'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
        'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
        'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
        'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
        'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
        'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
        'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
        'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
        'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
        'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
        'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
        'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
        'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
        'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
        'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
        'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
        'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
        'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
        'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
    ]

    print("No of tasks: ", len(TASKS))
    n_tasks = len(TASKS)

    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
    from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
    from openpom.models.mpnn_pom import MPNNPOMModel, MPNNPOM
    from datetime import datetime
    from tqdm import tqdm
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from rdkit import Chem
    import pdb

    # uncomment and run if no splits saved yet

    # download curated dataset
    # !wget https://raw.githubusercontent.com/ARY2260/openpom/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv

    # The curated dataset can also found at `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv` in the repo.

    input_file = r'openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv'  # or new downloaded file path

    # get dataset

    featurizer = GraphFeaturizer()
    smiles_field = 'nonStereoSMILES'
    loader = dc.data.CSVLoader(tasks=TASKS,
                               feature_field=smiles_field,
                               featurizer=featurizer)
    dataset = loader.create_dataset(inputs=[input_file])
    n_tasks = len(dataset.tasks)

    # get train valid test splits
    splitter = IterativeStratifiedSplitter(order=2)
    train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8, train_dir='./splits/train_data',
                                                            test_dir='./splits/test_data')

    print("train_dataset: ", len(train_dataset))
    print("test_dataset: ", len(test_dataset))

    train_dataset = dc.data.DiskDataset('./splits/train_data')
    test_dataset = dc.data.DiskDataset('./splits/test_data')
    print("train_dataset: ", len(train_dataset))
    print("test_dataset: ", len(test_dataset))

    train_ratios = get_class_imbalance_ratio(train_dataset)
    assert len(train_ratios) == n_tasks

    # learning_rate = 0.001
    learning_rate = dc.models.optimizers.ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32 * 20,
                                                          staircase=True)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    accuracy_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    precision_metric = dc.metrics.Metric(dc.metrics.precision_score)
    recall_metric = dc.metrics.Metric(dc.metrics.recall_score)

    # no of models in the ensemble
    n_models = 10

    # no of epochs each model is trained for
    nb_epoch = 62

    for i in tqdm(range(n_models)):
        model = MPNNPOMModel(n_tasks=n_tasks,
                             batch_size=128,
                             learning_rate=learning_rate,
                             class_imbalance_ratio=train_ratios,
                             loss_aggr_type='sum',
                             node_out_feats=100,
                             edge_hidden_feats=75,
                             edge_out_feats=100,
                             num_step_message_passing=5,
                             mpnn_residual=True,
                             message_aggregator_type='sum',
                             mode='classification',
                             number_atom_features=GraphConvConstants.ATOM_FDIM,
                             number_bond_features=GraphConvConstants.BOND_FDIM,
                             n_classes=1,
                             readout_type='set2set',
                             num_step_set2set=3,
                             num_layer_set2set=2,
                             ffn_hidden_list=[392, 392],
                             ffn_embeddings=256,
                             ffn_activation='relu',
                             ffn_dropout_p=0.12,
                             ffn_dropout_at_input_no_act=False,
                             weight_decay=1e-5,
                             self_loop=False,
                             optimizer_name='adam',
                             log_frequency=32,
                             model_dir=f'./ensemble_models2/experiments_{i + 1}',
                             device_name='cuda')

        start_time = datetime.now()

        # fit model
        loss = model.fit(
            train_dataset,
            nb_epoch=nb_epoch,
            max_checkpoints_to_keep=1,
            deterministic=False,
            restore=False)
        end_time = datetime.now()

        # pdb.set_trace()
        train_acc = model.evaluate(train_dataset, [accuracy_metric])['accuracy_score']
        test_acc = model.evaluate(test_dataset, [accuracy_metric])['accuracy_score']

        train_precision = model.evaluate(train_dataset, [precision_metric])['precision_score']
        test_precision = model.evaluate(test_dataset, [precision_metric])['precision_score']

        train_recall = model.evaluate(train_dataset, [recall_metric])['recall_score']
        test_recall = model.evaluate(test_dataset, [recall_metric])['recall_score']

        train_scores = model.evaluate(train_dataset, [metric])['roc_auc_score']
        test_scores = model.evaluate(test_dataset, [metric])['roc_auc_score']

        print(
            f"loss = {loss}; train_acc = {train_acc}, train_precision = {train_precision}, train_recall= {train_recall}; test_acc = {test_acc}, test_precision = {test_precision}, test_recall = {test_recall}; time_taken = {str(end_time - start_time)}")
        print(f"train roc_auc = {train_scores}; test_roc_auc = {test_scores}")
        model.save_checkpoint()  # saves final checkpoint => `checkpoint2.pt`
        del model
        torch.cuda.empty_cache()
