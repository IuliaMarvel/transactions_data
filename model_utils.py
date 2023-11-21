import torch
from functools import partial
import os


from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule


'Embedder models hyperparameters'
trx_encoder_params = dict(
    embeddings_noise=0.001,
    numeric_values={'amount_rur': 'identity'},
    embeddings={
        'trans_date': {'in': 800, 'out': 16},
        'small_group': {'in': 250, 'out': 16},
    },
)

def embedder_model(hidden_size=128, lr=5e-3, checkpoint_path=None):
    
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(**trx_encoder_params),
        hidden_size=hidden_size,
        type='gru',
    )
    
    if os.path.exists('coles-emb.pt'):
        print('Load embedding model from checkpoint')
        seq_encoder.load_state_dict(torch.load('coles-emb.pt'))

    model = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=lr),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.1), # 30 and 0.1
    )
    
    return model