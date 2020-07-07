from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import datetime
import os
from norb import smallNORB
from Routing_Methods import EMRouting
import ResidualBlocks

#CURRENT VERSION USE RESIDUAL BLOCKS INSTEAD OF PLAIN CONVOLUTIONAL LAYERS
#epsilon:
eps = 1e-6
max_acc = 0.

#PARAMS:
numEpoch = 350
batch_size = 32
numiter = 0
log_it = 10
epoch = 0
m = 0.2
routing_iterations = 3
train_mode = True

save_path="models"
dataset = "smallnorb"  # "smallnorb", "smallnorb_azimuth", "smallnorb_elevation" "mnist", "cifar10", "fashion-mnist", "svhn"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if dataset == "smallnorb":
    input_size=32
    num_class = 5
    input_size = 32
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="default"), batch_size=batch_size,
                                                shuffle=True)
    Dataset_test = torch.utils.data.DataLoader(smallNORB(path, train=False, mode="default"), batch_size=32, shuffle=False)

    writer_tag = 'EM_Routing_{}_Quaternion_ResidualExtractor_NormalCaps64'.format(dataset)


elif dataset == 'smallnorb_azimuth':
    num_class = 5
    input_size = 32
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="azimuth"),
                                                batch_size=batch_size,
                                                shuffle=True)
    Dataset_test = torch.utils.data.DataLoader(smallNORB(path, train=False, mode="azimuth"), batch_size=32,
                                               shuffle=False)

    writer_tag = 'EM_Routing_{}_Quaternion_ResidualExtractor_NormalCaps64'.format(dataset)

    validation_limit = 3.7

elif dataset == 'smallnorb_elevation':
    num_class = 5
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="elevation"),
                                                batch_size=batch_size,
                                                shuffle=True)
    Dataset_test = torch.utils.data.DataLoader(smallNORB(path, train=False, mode="elevation"), batch_size=32,
                                               shuffle=False)

    writer_tag = 'EM_Routing_{}_Quaternion_ResidualExtractor_NormalCaps64'.format(dataset)

    validation_limit = 4.3


elif dataset == "mnist":
    num_class = 10
    input_size = 28
    path = "mnist/"
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(path, train=True, download=True, transform=dataset_transform)
    Dataset_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(path, train=False, download=True, transform=dataset_transform)
    Dataset_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    writer_tag = 'EM_Routing3_MNIST_Quaternion_ResidualExtractor_NormalCaps64'


elif dataset == "fashion-mnist":
    num_class = 10
    path = "fashion-mnist/"
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.FashionMNIST(path, train=True, download=True, transform=dataset_transform)
    Dataset_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.FashionMNIST(path, train=False, download=True, transform=dataset_transform)
    Dataset_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    writer_tag = 'EM_Routing3_FashionMNIST_Quaternion_ResidualExtractor_NormalCaps64'


elif dataset =='svhn':
    num_class = 10
    path='svhn/'
    transform = transforms.Compose([
                    transforms.Scale(32, 32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    svhn_test = datasets.SVHN(root=path, split='test', download=True, transform=transform)
    Dataset_test = torch.utils.data.DataLoader(dataset=svhn_test,
                                              batch_size=32,
                                              shuffle=True)

    svhn_train = datasets.SVHN(root=path, split='train', download=True, transform=transform)
    Dataset_train = torch.utils.data.DataLoader(dataset=svhn_train,
                                               batch_size=32,
                                               shuffle=True)

    writer_tag = 'EM_Routing3_SVHN_Quaternion_ResidualExtractor_NormalCaps64'

elif dataset == "cifar10":
    num_class = 10
    path = "cifar/"
    input_size = 32

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    Dataset_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    Dataset_test = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    writer_tag = 'EM_Routing3_CIFAR10_Quaternion_ResidualExtractor_NormalCaps64'



def runTagGen(runtag=''):
    date = datetime.datetime.now().strftime('%b%d-%y_%H-%M')

    return os.path.join('runs', date + '_' + runtag)

# Create receptive field ofssets NO PADDING ASSUMED
# Inputs:
def receptive_offset(imgSize, start, j_out, r_out, M_size):

    receptiveCenters_x = torch.arange(start[0], imgSize[0] - r_out[0] / 2, step=j_out[0])
    receptiveCenters_y = torch.arange(start[1], imgSize[1] - r_out[0] / 2, step=j_out[1])
    receptiveCenters_x = receptiveCenters_x.repeat(receptiveCenters_y.size(0), 1).t()
    receptiveCenters_y = receptiveCenters_y.repeat(receptiveCenters_x.size(0), 1)
    receptiveCenters = torch.cat((receptiveCenters_x.unsqueeze(2), receptiveCenters_y.unsqueeze(2)), 2)
    scale = torch.tensor(imgSize, dtype=torch.float).unsqueeze(0).unsqueeze(1)
    scaled_coords = (receptiveCenters / scale).unsqueeze(2).permute(0, 1, 3, 2)
    scaled_coords = nn.functional.pad(scaled_coords, (M_size[0] - 1, 0, 0, M_size[1] - 2), 'constant', 0)

    return scaled_coords


# Receptive field calculator:
# r     : Receptive field size
# j     : jump on original dimensions(stride but on the original spatial coordinates)
# start : Center of the receptive field of the right top feature(first one.).
# returns receptive field center given current layers stride, padding, kernel size and previous layers r_in, start_in, j_in,
# stride, kernel_size, padding vars are pytorch conv layer compatible.
def receptive_field(stride, kernel_size, padding, r_in=(1, 1), start_in=(0.5, 0.5), j_in=(1, 1)):

    j_out = torch.tensor([j_in[0] * stride[0], j_in[1] * stride[1]], dtype=torch.float)
    r_out = torch.tensor([r_in[0] + (kernel_size[0] - 1) * j_in[0],
                          r_in[1] + (kernel_size[1] - 1) * j_in[1]], dtype=torch.float)

    start_out = torch.tensor([start_in[0] + ((kernel_size[0] - 1) / 2 - padding[0]) * j_in[0],
                              start_in[1] + ((kernel_size[1] - 1) / 2 - padding[1]) * j_in[1]], dtype=torch.float)

    return r_out, start_out, j_out



class FCCapsMatrix(nn.Module):

    def __init__(self, inCaps, outCaps, M_size, routing_iterations, routing, receptive_centers):
        super(FCCapsMatrix, self).__init__()
        self.W = nn.Parameter(torch.randn(1, 1, 1, inCaps, outCaps, *M_size))
        self.Beta_a = nn.Parameter(torch.Tensor(1, outCaps))
        self.Beta_u = nn.Parameter(torch.Tensor(1, outCaps, 1))
        nn.init.uniform_(self.Beta_a.data)
        nn.init.uniform_(self.Beta_u.data)
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.M_size = M_size
        self.routing = routing(routing_iterations)
        self.register_buffer("receptive_centers", receptive_centers)


    def forward(self, x, a):
        V = x.unsqueeze(4) @ self.W
        receptive_centers = self.receptive_centers.unsqueeze(0).unsqueeze(3).unsqueeze(4).to(device)
        V = V + receptive_centers
        V = V.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        a = a.flatten(start_dim=1, end_dim=a.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        R = (torch.ones(*a.size(), self.outCaps) / self.outCaps).to(device)  # R: <B, Spatial, Kernel Size, Caps_in, Caps_out>
        a = a.unsqueeze(6)

        return self.routing(V, a, self.Beta_u, self.Beta_a, R,
                            (x.size(0), self.outCaps, *self.M_size))


class FCCapsMatrixNorecept(nn.Module):

    def __init__(self, inCaps, outCaps, M_size, routing_iterations, routing):
        super(FCCapsMatrixNorecept, self).__init__()
        self.W = nn.Parameter(torch.randn(1, 1, 1, inCaps, outCaps, *M_size))
        self.Beta_a = nn.Parameter(torch.Tensor(1, outCaps))
        self.Beta_u = nn.Parameter(torch.Tensor(1, outCaps, 1))
        nn.init.uniform_(self.Beta_a.data)
        nn.init.uniform_(self.Beta_u.data)
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.M_size = M_size
        self.routing = routing(routing_iterations)



    def forward(self, x, a):
        V = x.unsqueeze(4) @ self.W
        V = V
        V = V.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        a = a.flatten(start_dim=1, end_dim=a.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        R = (torch.ones(*a.size(), self.outCaps) / self.outCaps).to(device)  # R: <B, Spatial, Kernel Size, Caps_in, Caps_out>
        a = a.unsqueeze(6)

        return self.routing(V, a, self.Beta_u, self.Beta_a, R,
                            (x.size(0), self.outCaps, *self.M_size))

class PrimaryMatCaps(nn.Module):
    def __init__(self, in_channels, outCaps, M_size):
        super(PrimaryMatCaps, self).__init__()
        self.activation_layer = nn.Conv2d(in_channels=in_channels, out_channels=outCaps, kernel_size=1)
        self.pose_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=outCaps * (torch.prod(torch.tensor(M_size, dtype=torch.int)).item()),
                                    kernel_size=1)
        self.M_size = M_size
        self.stride = (1, 1)
        self.kernel_size = (1, 1)
        self.padding = (0, 0)

    def forward(self, x):
        M = self.pose_layer(x)
        a = torch.sigmoid(self.activation_layer(x))
        # reshape ops:
        M = M.permute(0, 2, 3, 1)
        M = M.view(M.size(0), M.size(1), M.size(2), -1, *self.M_size)
        a = a.permute(0, 2, 3, 1)

        return M, a


# W         : <1, 1, Kx, Ky, inCaps, outCaps, Mx,My>
class ConvCapsMatrix(nn.Module):

    def __init__(self, kernel_size, stride, inCaps, outCaps, M_size, routing_iterations, routing):
        super(ConvCapsMatrix, self).__init__()
        self.W = nn.Parameter(torch.rand(1, 1, 1, *kernel_size, inCaps, outCaps, *M_size))
        self.Beta_a = nn.Parameter(torch.Tensor(1, outCaps))
        self.Beta_u = nn.Parameter(torch.Tensor(1, outCaps, 1))
        nn.init.uniform_(self.Beta_a.data)
        nn.init.uniform_(self.Beta_u.data)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (0, 0)
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.M_size = M_size
        self.routing = routing(routing_iterations)

    #   Pose Input dims      : <B, Spatial, Caps, CapsDims<dx,dy>>
    #   Votes dim       : <B, Spatial, Caps_in, Caps_out, CapsDims<dx,dy>>
    def forward(self, x, a):
        # Messy unfold for convolutionally connected capsules. Faster computation this way.
        x = x.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    6,
                                                                                                                    7,
                                                                                                                    3,
                                                                                                                    4,
                                                                                                                    5).unsqueeze(6).contiguous()
        a = a.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    4,
                                                                                                                    5,
                                                                                                                    3).contiguous()
        V = (x @ self.W)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1)
        R = (torch.ones(*a.size(), self.outCaps) / self.outCaps).to(device)  # R: <B, Spatial, Kernel Size, Caps_in, Caps_out>
        a = a.unsqueeze(6)

        return self.routing(V, a, self.Beta_u, self.Beta_a, R,
                            (x.size(0), x.size(1), x.size(2), self.outCaps, *self.M_size))


# Exactly same network with the paper:
class MatCapNet(nn.Module):

    def __init__(self, inputSize, routing_iterations):
        super(MatCapNet, self).__init__()


        if dataset == "smallnorb" or dataset == "smallnorb_azimuth" or dataset == "smallnorb_elevation":
            #self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=2)
            self.resblock1 = ResidualBlocks.BasicPreActResBlock(2, 64, 1)
            self.resblock2 = ResidualBlocks.BasicPreActResBlock(64, 64 + 32, 2)
            self.primarycaps = PrimaryMatCaps(in_channels=96, outCaps=32, M_size=(4, 4))
        elif dataset == "cifar10" or dataset == 'svhn':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=2)
            self.primarycaps = PrimaryMatCaps(in_channels=256, outCaps=32, M_size=(4, 4))
        else:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
            self.primarycaps = PrimaryMatCaps(in_channels=32, outCaps=32, M_size=(4, 4))

        self.convcapmat1 = ConvCapsMatrix(kernel_size=(3, 3), stride=(2, 2), inCaps=32, outCaps=32, M_size=(4, 4),
                                          routing_iterations=routing_iterations, routing=EMRouting)
        self.convcapmat2 = ConvCapsMatrix(kernel_size=(3, 3), stride=(1, 1), inCaps=32, outCaps=32, M_size=(4, 4),
                                          routing_iterations=routing_iterations, routing=EMRouting)
        # Find a better WAY!
        # r_out, start_out, j_out = receptive_field(self.conv1.stride, self.conv1.kernel_size, self.conv1.padding)
        # r_out, start_out, j_out = receptive_field(self.primarycaps.stride, self.primarycaps.kernel_size,
        #                                           self.primarycaps.padding, r_out, start_out, j_out)
        # r_out, start_out, j_out = receptive_field(self.convcapmat1.stride, self.convcapmat1.kernel_size,
        #                                           self.convcapmat1.padding, r_out, start_out, j_out)
        # r_out, start_out, j_out = receptive_field(self.convcapmat2.stride, self.convcapmat2.kernel_size,
        #                                           self.convcapmat2.padding, r_out, start_out, j_out)
        #
        # scaled_receptive_centers = receptive_offset(imgSize=inputSize, start=start_out, j_out=j_out, M_size=(4, 4),
        #                                             r_out=r_out)

        self.classcapmat = FCCapsMatrixNorecept(inCaps=32, outCaps=num_class, M_size=(4, 4), routing_iterations=routing_iterations, routing=EMRouting)


    def forward(self, x):
       # x = F.relu(self.conv1(x))
        x = self.resblock2(self.resblock1(x))
        x = self.primarycaps(F.relu(x))

        x = self.convcapmat1(x[0], x[1])
        x = self.convcapmat2(x[0], x[1])
        x = self.classcapmat(x[0], x[1])

        return x


class SpreadLoss(nn.Module):
    def __init__(self):
        super(SpreadLoss, self).__init__()
        self.m = 0.2

    def forward(self, y_hat, Y):
        Y_onehot = torch.eye(num_class).index_select(dim=0, index=Y.squeeze())
        a_t = (Y_onehot * y_hat).sum(dim=1)
        margins = (self.m - (a_t.unsqueeze(1) - y_hat)) * (1 - Y_onehot)
        Loss_perCaps = (torch.max(margins, torch.zeros(margins.shape)) ** 2)
        Loss = Loss_perCaps.sum(dim=1)
        # the m schedule stated in open review: https://openreview.net/forum?id=HJWLfGWRb
        if self.m < 0.9:
            self.m = 0.2 + 0.79 * torch.sigmoid(torch.min(torch.tensor([10, numiter / 50000. - 4])))
        return Loss.mean()




y_onehot = torch.zeros(10, 5)
loss_fn = SpreadLoss()
model = MatCapNet((input_size, input_size), 3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)
optimizer = Adam(model.parameters(), weight_decay=0.0000002, lr=3e-3)



decay_steps = 20000
decay_rate = .96
exponential_lambda = lambda x: decay_rate ** (x / decay_steps)
scheduler = lr_scheduler.LambdaLR(optimizer, exponential_lambda)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
# setup writer.
writer = SummaryWriter(logdir=runTagGen(runtag=writer_tag))



def train(epoch):
    global numiter
    model.train()
    total_loss = 0.
    for batch_idx, (X, Y) in enumerate(Dataset_train):
        optimizer.zero_grad()
        Xgpu = X.to(device)
        Y_hat_pose_mean, Y_hat, Y_hat_pose_sigma = model(Xgpu)
        loss = loss_fn(Y_hat.cpu(), Y.squeeze())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        print('NumIter: {} \t\tTrain Epoch: {} '
              '[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(numiter, epoch,
                                                         batch_idx * len(X),
                                                         len(Dataset_train.dataset),
                                                         100 * batch_idx / float(len(Dataset_train)),
                                                         loss.item()), end='\r')
        scheduler.step()
        numiter = numiter + 1

    writer.add_scalar("Loss/Train Epoch Loss", total_loss, numiter)
    return 0


def test(path=''):
    global max_acc
    with torch.no_grad():
        print("Test PHASE:\n")
        if path:
            model.load_state_dict(torch.load(path))
        model.eval()
        true_positive = torch.tensor(0)
        total_loss = 0.

        for batch_idx, (X, Y) in enumerate(Dataset_test):
            if dataset == "smallnorb" or dataset == "smallnorb_azimuth" or dataset == "smallnorb_elevation":
                Y = Y.squeeze()
            Xgpu = X.to(device)
            Y_hat_pose_mean, Y_hat, Y_hat_pose_sigma = model(Xgpu)
            loss = loss_fn(Y_hat.cpu(), Y.squeeze())
            total_loss += loss
            _, class_predictions = Y_hat.max(dim=1)
            true_positive += class_predictions.eq(Y.to(device)).sum()

            print('Test Progress: [{}/{} ({:.0f}%)]'.format(batch_idx * len(X), len(Dataset_test.dataset),
                                                            100. * batch_idx / len(Dataset_test)), end='\r')

        accuracy = float(100) * float(true_positive) / float(len(Dataset_test.dataset))
        if max_acc < accuracy:
            max_acc = accuracy
            torch.save(model.state_dict(),
                       '{}/{}_maxAcc.pth'.format(save_path, dataset))
            writer.add_text("maximum accuracy:", "{}".format(max_acc))

        print("Test accuracy: {}%".format(accuracy))
        print("Max accuracy: {}%".format(max_acc))
        writer.add_scalar("Test Accuracy", accuracy, numiter)
        writer.add_scalar("Loss/Test Epoch Loss", total_loss, numiter)


def train_test(path=''):
    with torch.no_grad():
        print("Test PHASE:\n")
        if path:
            model.load_state_dict(torch.load(path))
        model.eval()
        true_positive = torch.tensor(0)
        for batch_idx, (X, Y) in enumerate(Dataset_train):
            if dataset == "smallnorb" or dataset == "smallnorb_azimuth" or dataset == "smallnorb_elevation":
                Y = Y.squeeze()
            Xgpu = X.to(device)
            Y_hat_pose_mean, Y_hat, Y_hat_pose_sigma = model(Xgpu)
            _, class_predictions = Y_hat.max(dim=1)
            true_positive += class_predictions.eq(Y.to(device)).sum()

            print('Training test Progress: [{}/{} ({:.0f}%)]\t\t'.format(batch_idx * len(X),
                                                                         len(Dataset_train.dataset),
                                                                         100. * batch_idx / len(Dataset_train)),
                  end='\r')

        accuracy = float(100) * float(true_positive) / float(len(Dataset_train.dataset))
        print("Training set accuracy: {}%".format(accuracy))
        writer.add_scalar("Training set Accuracy", accuracy, numiter)







if __name__ == "__main__":

    print(torch.__version__)
    print("model info:",
                    "number of parameters: {}\nBatch size: {}".format(
                        count_parameters(model), batch_size))
    writer.add_text("model info:",
                    "number of parameters: {}\nBatch size: {}".format(
                        count_parameters(model), batch_size))
    if train_mode==True:
        print("Initiating Training...")
        for epoch in range(numEpoch):
            train(epoch)
            if epoch%10 == 0 and epoch != 0:
                torch.save(model.state_dict(), '{}/{}_{:03d}EM_Routing_paper.pth'.format(save_path, dataset, epoch + 1))
            test()
            train_test()

        torch.save(model.state_dict(), '{}/{}_{:03d}EM_Routing_paper.pth'.format(save_path, dataset, epoch + 1))
        test()
        train_test()


    else:
        print("Initiating test only mode...")
        test(path=' '.format(routing_iterations))
