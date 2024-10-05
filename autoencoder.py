
import torch
import lightning
import math

#Defining the encoder

class ConvEncoder(torch.nn.Module):
    def __init__(self,base_channel_size: int, 
                 latent_dim: int, 
                 num_input_channels: int = 1, 
                 width: int = 32,
                 height: int = 32,
                 act_fn: object = torch.nn.GELU, 
                 ids_nr: int = 64,
                 dropout_p=0.5):
        super().__init__()
        c_hid = base_channel_size
        self.net1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=[1,2], dtype=torch.float32),  # 32x32 => 16x16
            act_fn(),
            #torch.nn.Dropout(dropout_p), 
            torch.nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=[1,2], dtype=torch.float32),  # 16x16 => 8x8
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=[1,2], dtype=torch.float32),  # 8x8 => 4x4
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Flatten(),  # Image grid to single feature vector
        )
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(2 * c_hid * width * math.ceil(height/2**3) + (ids_nr + 1), latent_dim * 4, dtype=torch.float32),
            act_fn(),
            torch.nn.Linear(latent_dim * 4, latent_dim * 2, dtype=torch.float32),
            act_fn(),
            torch.nn.Linear(latent_dim * 2, latent_dim, dtype=torch.float32),
        )

    def forward(self, x, ids):
        x = self.net1(x)
        x = torch.cat((x, ids), dim=1)
        return self.net2(x)


class FcEncoder(torch.nn.Module):
    def __init__(self,base_channel_size: int, 
                 latent_dim: int, 
                 num_input_channels: int = 1, 
                 width: int = 32,
                 height: int = 32,
                 act_fn: object = torch.nn.GELU, 
                 ids_nr: int = 64, 
                 dropout_p=0.5):
        super().__init__()
        c_hid = base_channel_size*width
        self.net1 = torch.nn.Sequential(
            torch.nn.Flatten(),  # Image grid to single feature vector
            torch.nn.Linear(num_input_channels*width*height, c_hid, dtype=torch.float32),
            act_fn(),
            #torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(c_hid, math.ceil(c_hid/2), dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(math.ceil(c_hid/2), math.ceil(c_hid/4), dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(math.ceil(c_hid/4), math.ceil(c_hid/8), dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(math.ceil(c_hid/8), math.ceil(c_hid/16), dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
        )
        
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(math.ceil(c_hid/16) + (ids_nr +1), latent_dim * 4, dtype=torch.float32),
            act_fn(),
            torch.nn.Linear(latent_dim * 4, latent_dim * 2, dtype=torch.float32),
            act_fn(),
            torch.nn.Linear(latent_dim * 2, latent_dim, dtype=torch.float32),
        )
    
    def forward(self, x, ids):
        x = self.net1(x)
        x = torch.cat((x, ids), dim=1)
        return self.net2(x)


#Defining the decoder

class ConvDecoder(torch.nn.Module):
    def __init__(self, base_channel_size: int, 
                 latent_dim: int, 
                 num_input_channels: int = 1,
                 width: int = 32,
                 height: int = 32,
                 act_fn: object = torch.nn.GELU, 
                 ids_nr: int = 64,
                 dropout_p=0.5):
        super().__init__()
        c_hid = base_channel_size
        self.width = width
        self.height = height
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + (ids_nr +1), latent_dim * 2, dtype=torch.float32),
            act_fn(),
            torch.nn.Linear(latent_dim * 2, latent_dim * 4, dtype=torch.float32), 
            act_fn(),
            torch.nn.Linear(latent_dim * 4, 2 * c_hid * width * math.ceil(height/2**3), dtype=torch.float32), 
            act_fn(),
            )
        self.net1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=[0,1], padding=[1,1], stride=[1,2], dtype=torch.float32
            ),  # 4x4 => 8x8
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=[0,1], padding=[1,1], stride=[1,2], dtype=torch.float32),  # 8x8 => 16x16
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, dtype=torch.float32),
            act_fn(),
            #torch.nn.Dropout(dropout_p), 
            torch.nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=[0,1], padding=[1,1], stride=[1,2], dtype=torch.float32
            ),  # 16x16 => 32x32
            #torch.nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            act_fn(),
        )

    def forward(self, x, ids):
        x = torch.cat((x, ids), dim=1)
        x = self.net2(x)
        x = x.reshape(x.shape[0], -1, self.width, math.ceil(self.height/2**3))
        x = self.net1(x)
        return x

class FcDecoder(torch.nn.Module):
    def __init__(self, base_channel_size: int, 
                 latent_dim: int, 
                 num_input_channels: int = 1,
                 width: int = 32,
                 height: int = 32,
                 act_fn: object = torch.nn.GELU, 
                 ids_nr: int = 64,
                 dropout_p=0.5):
        super().__init__()
        c_hid = base_channel_size*width
        print(math.ceil(c_hid/16))
        self.width = width 
        self.height = height
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + (ids_nr + 1), latent_dim * 2, dtype=torch.float32),
            act_fn(),
            torch.nn.Linear(latent_dim * 2, latent_dim * 4, dtype=torch.float32),
            act_fn(),
            torch.nn.Linear(latent_dim * 4, math.ceil(c_hid/16), dtype=torch.float32),
            act_fn(),
            )
        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(math.ceil(c_hid/16), math.ceil(c_hid/8), dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(math.ceil(c_hid/8), math.ceil(c_hid/4), dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(math.ceil(c_hid/4), math.ceil(c_hid/2), dtype=torch.float32),
            act_fn(),
            torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(math.ceil(c_hid/2), c_hid, dtype=torch.float32),
            act_fn(),
            #torch.nn.Dropout(dropout_p), 
            torch.nn.Linear(c_hid, num_input_channels*width*height, dtype=torch.float32),
            #torch.nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            act_fn(),
            )

    def forward(self, x, ids):
        x = torch.cat((x, ids), dim=1)
        x = self.net2(x)
        x = self.net1(x)
        x = x.reshape(x.shape[0], -1, self.width, self.height)
        return x


#Defining the autoencoder

class Autoencoder(lightning.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = FcEncoder,
        decoder_class: object = FcDecoder,
        num_input_channels: int = 1,
        width: int = 6,
        height: int = 128,
        dropout_p: float = 0.0,
        act_fn: object = torch.nn.GELU,
        ids_nr: int = 64,
        learning_rate: float = 1e-3,
        exclude_outliers: bool = False,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels=num_input_channels, 
                                     base_channel_size=base_channel_size,
                                     width=width,
                                     height=height, 
                                     latent_dim=latent_dim, 
                                     act_fn=act_fn,
                                     ids_nr=ids_nr,
                                     dropout_p=dropout_p)
        self.decoder = decoder_class(num_input_channels=num_input_channels, 
                                     base_channel_size=base_channel_size, 
                                     width=width,
                                     height=height, 
                                     latent_dim=latent_dim, 
                                     act_fn=act_fn,
                                     ids_nr=ids_nr,
                                     dropout_p=dropout_p)
        # Example input array needed for visualizing the graph of the network
        self.example_signals = torch.zeros(2, num_input_channels, width, height)
        self.example_ids = torch.zeros(2, ids_nr + 1)
        self.example_input_array = (self.example_signals, self.example_ids)
        self.learning_rate = learning_rate
        self.exclude_outliers = exclude_outliers

    def forward(self, x, ids):
        z = self.encoder(x, ids)
        x_hat = self.decoder(z, ids)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x, y = batch  # Y contains channels and energy
        x_hat = self.forward(x, y)
        if not self.exclude_outliers:
            loss = torch.nn.functional.mse_loss(x, x_hat, reduction="mean")
        else:
            loss = torch.nn.functional.mse_loss(x, x_hat, reduction="none")
            loss = loss.mean(dim=[1, 2, 3])
            loss_threshold = 8 * loss.median()
            loss = loss.clamp(max=loss_threshold.item()).mean()
        return loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5, min_lr=5e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        return loss

class VarAutoencoder(Autoencoder):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = FcEncoder,
        decoder_class: object = FcDecoder,
        num_input_channels: int = 1,
        width: int = 6,
        height: int = 128,
        dropout_p: float = 0.0,
        act_fn: object = torch.nn.GELU,
        ids_nr: int = 64,
        learning_rate: float = 1e-3,
        exclude_outliers: bool = False,
    ):
        super().__init__(base_channel_size=base_channel_size, 
                         latent_dim=latent_dim, 
                         encoder_class=encoder_class, 
                         decoder_class=decoder_class, 
                         num_input_channels=num_input_channels, 
                         width=width, 
                         height=height, 
                         dropout_p=dropout_p,
                         act_fn=act_fn,
                         ids_nr=ids_nr,
                         learning_rate=learning_rate)
        
        # latent mean and variance 
        self.mean = torch.nn.Linear(latent_dim, latent_dim)
        self.logvar = torch.nn.Linear(latent_dim, latent_dim)
        
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(mean.device)
        #eps = torch.zeros_like(std)
        return eps * std + mean

        
    def forward_withpars(self, x, ids):
        """Reimplementation of forward function to return both the reconstructed image and the latent representation."""
        z = self.encoder(x, ids)
        #act functions?
        mean, logvar = self.mean(z), self.logvar(z)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z, ids)
        return x_hat, mean, logvar

    def forward(self, x, ids):
        x_hat, _, _ = self.forward_withpars(x, ids)
        return x_hat

    def _get_KL_loss(self, batch):
        """Reimplementation to include the KL divergence term."""
        x, y = batch  # We do not need the labels
        x_hat, mean, logvar = self.forward_withpars(x, y)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        # Total loss is the sum of reconstruction and KL divergence loss
        return kl_loss

    def _get_total_loss(self, batch):
        reconstruction_bias = 100
        reconstruction_loss = self._get_reconstruction_loss(batch) * reconstruction_bias
        kl_loss = self._get_KL_loss(batch)
        return [reconstruction_loss, kl_loss]

    def training_step(self, batch, batch_idx):
        loss = self._get_total_loss(batch)
        self.log("reconstruction_train_loss", loss[0])
        self.log("KL_train_loss", loss[1])
        loss = loss[0]+loss[1]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_total_loss(batch)
        self.log("reconstruction_val_loss", loss[0])
        self.log("KL_val_loss", loss[1])
        loss = loss[0]+loss[1]
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_total_loss(batch)
        self.log("reconstruction_test_loss", loss[0])
        self.log("KL_test_loss", loss[1])
        loss = loss[0]+loss[1]
        self.log("test_loss", loss)
        return loss