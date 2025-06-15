import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tensorboardX import SummaryWriter

from tqdm import tqdm
from dataloader import load_dataset
from model import EfficientNet


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print(f"{batch_idx*len(data)}/{len(train_loader.dataset)}")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            writer.add_scalar("Test Loss", test_loss, epoch)
            pred = output.argmax(1)
            correct += float((pred == target).sum())
            writer.add_scalar("Test Accuracy", correct, epoch)
            
        test_loss /= len(test_loader.dataset)
        correct /= len(test_loader.dataset)
        return test_loss, correct
        writer.close()

if __name__ == "__main__":

    num_epochs = 10
    learning_rate = 0.001

    trainloader, testloader = load_dataset()

    use_cuda = torch.cuda.is_available()
    print("use_cuda : ", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = EfficientNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set Summary writer for Tensorboard visualization
    writer = SummaryWriter("./EfficientNet/tensorboard") 

    for epoch in tqdm(range(1, num_epochs + 1)):
        train(model, device, trainloader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, testloader)
        writer.add_scalar("Test Loss", test_loss, epoch)
        writer.add_scalar("Test Accuracy", test_accuracy, epoch)
        print(f"Processing Result = Epoch : {epoch}   Loss : {test_loss}   Accuracy : {test_accuracy}")
        writer.close()
        
    print(f"Result of EfficientNet = Epoch : {epoch}   Loss : {test_loss}   Accuracy : {test_accuracy}")