import torch
import pytorch_lightning as pl
from ptls.data_load.datasets import inference_data_loader


def get_trainer(train=True, n_epochs=15):
    if train:
        trainer = pl.Trainer(
            max_epochs=n_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            enable_progress_bar=True,
        )
    else:
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0
        )
    return trainer

def train_emb_model(model, data_loader,  n_epochs=15, save_model=True):
    trainer = get_trainer(n_epochs=n_epochs)
    trainer.fit(model, data_loader)
    print(trainer.logged_metrics)
    
    if save_model:
        torch.save(model.seq_encoder.state_dict(), "coles-emb.pt")
    
    return trainer 

    
def get_embeddings(model, data):
    model.eval()
    trainer = get_trainer(train=False)
    dataloader = inference_data_loader(data, num_workers=0, batch_size=256)
    data_embeds = torch.vstack(trainer.predict(model, dataloader))
    print(f'Getting embedding of shape {data_embeds.shape}')
    return data_embeds