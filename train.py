import torch
from tqdm import tqdm
from glob import glob
from data import NpzDataset
from model import UNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

files = glob('data/*.npz')
features = ['Bx', 'By', 'Bz', 'rho', 'anisoP', 'agyrotropy', 'absE']
height_out = 216
width_out = 535
batch_size = 1
epochs = 20

train_dataset = NpzDataset(files, features)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

unet = UNet(
    enc_chs=(len(features), 64, 128),
    dec_chs=(128, 64),
    num_class=1,
    retain_dim=True,
    out_sz=(height_out, width_out)
).to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001)

for epoch in range(epochs):

    running_loss = 0.0
    for data in tqdm(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['X'].to(device), data['y'].to(device)

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
