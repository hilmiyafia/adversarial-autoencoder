
import torch
import torch.utils.tensorboard
from dataset import Dataset
from model import Autoencoder, Critic
from losses import VGGLoss


PATH = "faces"
LR = 2e-4
BATCH = 16
BETA = 0.1
EPOCHS = 1000
DEVICE = "cuda"


if __name__ == "__main__":
    critic = Critic().to(DEVICE)
    critic.load_state_dict(torch.load("critic.pt", weights_only=False))
    critic.optimizer = torch.optim.AdamW(critic.parameters(), lr=LR)

    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load("model.pt", weights_only=False))
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    dataset = Dataset(PATH)
    vgg_loss = VGGLoss().to(DEVICE)
    writer = torch.utils.tensorboard.SummaryWriter()
    test = torch.stack([dataset[i] for i in range(4)]).to(DEVICE)
    step = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        data_critic = torch.utils.data.DataLoader(dataset, BATCH, shuffle=True)
        data_model = torch.utils.data.DataLoader(dataset, BATCH, shuffle=True)
        for batch_critic, batch_model in zip(data_critic, data_model):
            step += 1

            batch_critic = batch_critic.to(DEVICE)
            critic.optimizer.zero_grad()
            with torch.no_grad():
                latent_fake = model.encoder(batch_critic)
                latent_real = torch.randn(*latent_fake.shape).to(DEVICE)
            loss_fake = 0.5 * (critic(latent_fake) + 1).pow(2).mean()
            loss_real = 0.5 * (critic(latent_real) - 1).pow(2).mean()
            writer.add_scalar("critic/fake", loss_fake, step)
            writer.add_scalar("critic/real", loss_real, step)
            loss = loss_fake + loss_real
            loss.backward()
            critic.optimizer.step()

            batch_model = batch_model.to(DEVICE)
            model.optimizer.zero_grad()
            output, latent = model(batch_model)
            loss_output = (output - batch_model).abs().mean()
            loss_vgg = (vgg_loss(output) - vgg_loss(batch_model)).abs().mean()
            loss_latent = 0.5 * (critic(latent) - 1).pow(2).mean()
            writer.add_scalar("loss/output", loss_output, step)
            writer.add_scalar("loss/vgg", loss_vgg, step)
            writer.add_scalar("loss/latent", loss_latent, step)
            loss = loss_output + BETA*loss_vgg + loss_latent
            loss.backward()
            model.optimizer.step()

        with torch.no_grad():
            writer.add_images("test", model(test)[0], step)

        torch.save(model.state_dict(), "model.pt")
        torch.save(critic.state_dict(), "critic.pt")
        
            
