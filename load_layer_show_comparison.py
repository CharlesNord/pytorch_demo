# import the network structure
from linear_ae import Encoder, Decoder, AE 
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

print "deploy"
autoencoder = AE()
encoder = Encoder()
decoder = Decoder()
weights = torch.load('pretrain_ae.pkl')

# initialize the whole network
autoencoder.load_state_dict(weights)

# initialize encoder & decoder by keys
encoder_state = encoder.state_dict()
for key in encoder_state.keys():
    encoder_state[key].copy_(weights[key])
    # encoder_state[key] = weights[key] can't do this`

decoder_state = decoder.state_dict()
for key in decoder_state.keys():
    decoder_state[key].copy_(weights[key])

test_data = torch.load('../mnist/processed/test.pt')
test_data = Variable(test_data[0].type(torch.FloatTensor)[0:5] / 255.0)
z = encoder.forward(test_data.unsqueeze(1))
reconstruct = decoder(z)

# compare the origin image and the reconstructed image
comparison = torch.cat([test_data.view(-1, 1, 28, 28), reconstruct.view(-1, 1, 28, 28)[:5]])
comparison = make_grid(comparison.data, 5, 2)
comparison = comparison.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
plt.imshow(comparison)
plt.show()
