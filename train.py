
import torch
import lightning
import torchvision
from dataset import Dataset
from model import Autoencoder, Critic
from losses import VGGLoss


class AdversarialAutoencoder(lightning.LightningModule):

    def __init__(self, lr=2e-4, beta=[1.0, 0.1, 1.0], valset=None):
        super().__init__()
        self.save_hyperparameters("lr", "beta")
        self.automatic_optimization = False
        self.autoencoder = Autoencoder().train()
        self.critic = Critic().train()
        self.vgg = VGGLoss().eval()
        if valset is None: self.valset = None
        else: self.valset = torch.stack([sample for sample in valset])

    def training_step(self, batch, batch_index):
        autoencoder_optimizer, critic_optimizer = self.optimizers()
        self.train_critic(critic_optimizer, batch["critic"])
        self.train_autoencoder(autoencoder_optimizer, batch["autoencoder"])

    def train_critic(self, optimizer, batch):
        self.toggle_optimizer(optimizer)
        optimizer.zero_grad()
        with torch.no_grad():
            latent_fake = self.autoencoder.encoder(batch)
            latent_real = torch.randn(*latent_fake.shape).type_as(latent_fake)
        loss_fake = 0.5 * (self.critic(latent_fake)+1).pow(2).mean()
        loss_real = 0.5 * (self.critic(latent_real)-1).pow(2).mean()
        self.log("critic/fake", loss_fake)
        self.log("critic/real", loss_real)
        self.manual_backward(loss_fake+loss_real)
        optimizer.step()
        self.untoggle_optimizer(optimizer)

    def train_autoencoder(self, optimizer, batch):
        self.toggle_optimizer(optimizer)
        optimizer.zero_grad()
        output, latent = self.autoencoder(batch)
        loss_output = (output-batch).abs().mean()
        loss_vgg = (self.vgg(output)-self.vgg(batch)).abs().mean()
        loss_latent = 0.5 * (self.critic(latent)-1).pow(2).mean()
        self.log("loss/output", loss_output)
        self.log("loss/vgg", loss_vgg)
        self.log("loss/latent", loss_latent)
        self.manual_backward(self.hparams.beta[0]*loss_output 
                             + self.hparams.beta[1]*loss_vgg 
                             + self.hparams.beta[2]*loss_latent)
        optimizer.step()
        self.untoggle_optimizer(optimizer)
        
    def configure_optimizers(self):
        autoencoder_optimizer = torch.optim.AdamW(
            self.autoencoder.parameters(), lr=self.hparams.lr)
        critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self.hparams.lr)
        return [autoencoder_optimizer, critic_optimizer], []

    def on_train_epoch_end(self):
        if self.valset is None: return
        batch = self.valset.type_as(self.autoencoder.encoder[0].layers[0].weight)
        grid = torchvision.utils.make_grid(self.autoencoder(batch)[0], padding=0)
        self.logger.experiment.add_image("reconstruction", grid, self.current_epoch)


if __name__ == "__main__":
    dataset = Dataset("faces")
    trainset, valset = torch.utils.data.random_split(dataset, [len(dataset) - 4, 4])
    def get_loader():
        return torch.utils.data.DataLoader(trainset, 16, shuffle=True, 
                                           num_workers=2, persistent_workers=True)
    trainloader = lightning.pytorch.utilities.CombinedLoader({
        "autoencoder": get_loader(), "critic": get_loader()})
    model = AdversarialAutoencoder(valset=valset)
    trainer = lightning.Trainer(max_epochs=1000)
    trainer.fit(model, trainloader)
        
