import torch
import numpy as np
from tqdm import tqdm
from model import UNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

data = np.load('features.npz')
# B = data['B'].transpose((2, 0, 1))
features = ['Bx', 'By', 'Bz', 'rho', 'anisoP', 'agyrotropy', 'absE']
X = np.stack([data[feature] for feature in features], axis=0)
labeled_domain = data['labeled_domain']

height_out = labeled_domain.shape[0]
width_out = labeled_domain.shape[1]

print('X original shape:', X.shape)
print('y original shape:', labeled_domain.shape)

tensor_X = torch.tensor(X[np.newaxis, ...], dtype=torch.float32)
tensor_y = torch.tensor(labeled_domain[np.newaxis, ...], dtype=torch.float32)

print('X tensor shape:', tensor_X.size())
print('y tensor shape:', tensor_y.size())

train_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
train_loader = torch.utils.data.DataLoader(train_dataset)

unet = UNet(
    enc_chs=(len(features), 64, 128),
    dec_chs=(128, 64),
    num_class=1,
    retain_dim=True,
    out_sz=(height_out, width_out)
).to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)

batch_size = 1
epochs = 20

for epoch in range(epochs):

    running_loss = 0.0
    for data in tqdm(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs) # (batch_size, n_classes, img_cols, img_rows)

        outputs = outputs.reshape(batch_size*width_out*height_out)
        labels = labels.reshape(batch_size*width_out*height_out)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print(f'{(epoch + 1):3d} loss: {loss.item()}')

print('Finished training!')
