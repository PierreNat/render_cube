import torch
import os
import torch.nn as nn
import argparse

def train(dataloader, model, n_epochs, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device=torch.device('cpu')):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), 0.001)
    model.train()
    for epoch in range(n_epochs):
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        print('Epoch {}, loss: {}'.format(epoch+1, loss.item()))
        
def train_mse(dataloader, model, n_epochs, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device=torch.device('cpu')):
    train(dataloader, model, n_epochs, optimizer=None, loss_fn=nn.MSELoss(), device=torch.device('cpu'))


# database creation ---------------------------------------
# in: number of images
# out: validation bool

def creation_database(Obj_Name, nb_im=10000):
    print(nb_im)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, '{}.obj'.format(Obj_Name)))
    parser.add_argument('-c', '--color_input', type=str, default=os.path.join(data_dir, '{}.mtl'.format(Obj_Name)))

    for i in nb_im:
        parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'cube_{}.png'.format(i)))
        parser.add_argument('-f', '--filename_output2', type=str, default=os.path.join(data_dir, 'silhouette_{}.png'.format(i)))
        parser.add_argument('-g', '--gpu', type=int, default=0)
        args = parser.parse_args()

# set intrinsic camera parameters ---------------------------------------

