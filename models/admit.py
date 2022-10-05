import torch
import torch.nn as nn

from models.dynamic_net import Dynamic_FC

class ADMIT(nn.Module):
    def __init__(self, args):
        super(ADMIT, self).__init__()
        self.args = args
        input_dim = args.input_dim
        dynamic_type=args.dynamic_type, 
        init=args.init
        dynamic_type = args.dynamic_type
        self.cfg_hidden = [(input_dim, 50, 1, 'relu'), (50, 50, 1, 'relu')]
        self.cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
        self.degree = 2
        self.knots = [0.33, 0.66]

        # construct the representation network
        hidden_blocks = []
        hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(self.cfg_hidden):
            # fc layer
            if layer_idx == 0:
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                hidden_blocks.append(self.feature_weight)
            else:
                hidden_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                hidden_blocks.append(nn.ReLU(inplace=True))
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*hidden_blocks)
        self.drop_hidden = nn.Dropout(p=self.args.dropout)

        self.hidden_dim = hidden_dim

        # construct the inference network
        blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg):
            if layer_idx == len(self.cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1, dynamic_type=dynamic_type)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0, dynamic_type=dynamic_type))
        blocks.append(last_layer)

        self.out = nn.Sequential(*blocks)

        # construct the rw-weighting network
        rwt_blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg):
            if layer_idx == len(self.cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1, dynamic_type='mlp')
                
            else:
                rwt_blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0, dynamic_type='mlp'))
        rwt_blocks.append(last_layer)

        self.rwt = nn.Sequential(*rwt_blocks)

        self._initialize_weights(init)

    def forward(self, x, t):
        hidden = self.hidden_features(x)
        hidden = self.drop_hidden(hidden)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        w = self.rwt(t_hidden)
        w = torch.sigmoid(w) * 2
        w = torch.exp(w) / torch.exp(w).sum() * w.shape[0]
        
        out = self.out(t_hidden)

        return out, w, hidden

    def _initialize_weights(self, init):
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                # m.weight.data.normal_(0, 1.)
                m.weight.data.normal_(0, init)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()