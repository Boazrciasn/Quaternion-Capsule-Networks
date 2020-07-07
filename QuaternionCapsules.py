import torch
from norb import smallNORB
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import datetime
import os
from torchvision import datasets, transforms
from mydatasets import AffNIST, RotMNIST
import math
import sys
import ResidualBlocks
from Routing_Methods import EMRouting
from modules import QuaternionLayer
from modules import STRoutedQCLayer


eps = 1e-10
max_acc = 0.
# Pick which dataset to work on.

dataset = "smallnorb"     # "smallnorb", "smallnorb_azimuth", "smallnorb_elevation", "mnist", "cifar10",
                        # "fashion-mnist", "svhn" ,"affnist", "rotnist"
branched = True        #Branching in pose and activation extraction
loss_function = "spread"  # spread or crossentropy
batch_size = 2
init_type = "uniform_pi" # "normal" or "uniform_pi" for theta init

def runTagGen(test_type='', runtag=''):
    date = datetime.datetime.now().strftime('%b%d-%y_%H-%M')

    return os.path.join('runs', date + '_' + runtag)

if dataset == "smallnorb":
    num_class = 5
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="default"), batch_size=batch_size,
                                                shuffle=True)
    Dataset_test = torch.utils.data.DataLoader(smallNORB(path, train=False, mode="default"), batch_size=32, shuffle=False)

    writer_tag = 'EM_Routing_{}_Quaternion_ResidualExtractor_NormalCaps64_{}'.format(dataset, init_type)


elif dataset == 'smallnorb_azimuth':
    num_class = 5
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="azimuth"),
                                                batch_size=batch_size,
                                                shuffle=True)
    Dataset_test = torch.utils.data.DataLoader(smallNORB(path, train=False, mode="azimuth"), batch_size=32,
                                               shuffle=False)

    writer_tag = 'EM_Routing_{}_Quaternion_ResidualExtractor_NormalCaps64_{}'.format(dataset, init_type)

    validation_limit = 3.7

elif dataset == 'smallnorb_elevation':
    num_class = 5
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="elevation"),
                                                batch_size=batch_size,
                                                shuffle=True)
    Dataset_test = torch.utils.data.DataLoader(smallNORB(path, train=False, mode="elevation"), batch_size=32,
                                               shuffle=False)

    writer_tag = 'EM_Routing_{}_Quaternion_ResidualExtractor_NormalCaps64_{}'.format(dataset, init_type)

    validation_limit = 4.3


elif dataset == 'visualize':

    num_class = 5
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="visualize"),
                                                batch_size=18,
                                                shuffle=False)


elif dataset == "mnist":
    num_class = 10
    path = "mnist/"
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(path, train=True, download=True, transform=dataset_transform)
    Dataset_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(path, train=False, download=True, transform=dataset_transform)
    Dataset_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    writer_tag = 'EM_Routing3_MNIST_Quaternion_ResidualExtractor_NormalCaps64_{}'.format(init_type)


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
    writer_tag = 'EM_Routing3_FashionMNIST_Quaternion_ResidualExtractor_NormalCaps64_{}'.format(init_type)


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

    writer_tag = 'EM_Routing3_SVHN_Quaternion_ResidualExtractor_NormalCaps64_{}'.format(init_type)

elif dataset == "cifar10":
    num_class = 10
    path = "cifar/"

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

    writer_tag = 'EM_Routing3_CIFAR10_Quaternion_ResidualExtractor_NormalCaps64_{}'.format(init_type)

elif dataset == "affnist":
    num_class = 10
    path = "mnist/"

    train_transform = transforms.Compose(
        [   transforms.Pad(10),
            transforms.RandomAffine(0, translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])


    train_dataset = datasets.MNIST(path, train=True, download=True, transform=train_transform)
    Dataset_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_transform = transforms.Compose([transforms.Pad(4), transforms.ToTensor()])
    test_dataset = AffNIST("Affnist", train=False, transform=test_transform)
    Dataset_test = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)
    writer_tag = 'affnist'.format(init_type)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Pure quaternions --> 3 dimensional caps
class PrimaryQuatCaps(nn.Module):
    def __init__(self, in_channels_pose, in_channels_act, outCaps, quat_dims=3):
        super(PrimaryQuatCaps, self).__init__()
        self.activation_layer = nn.Conv2d(in_channels=in_channels_act, out_channels=outCaps, kernel_size=1)
        self.pose_layer = nn.Conv2d(in_channels=in_channels_pose,
                                    out_channels=outCaps * 3,
                                    kernel_size=1)

        torch.nn.init.xavier_uniform_(self.pose_layer.weight)
        self.batchNormPose = nn.BatchNorm2d(quat_dims * outCaps)
        self.batchNormA = nn.BatchNorm2d(outCaps)
        self.quat_dims = quat_dims
        self.stride = (1, 1)
        self.kernel_size = (1, 1)

    def forward(self, x, a):
        M = self.batchNormPose(self.pose_layer(x))
        a = torch.sigmoid(self.batchNormA(self.activation_layer(a)))
        M = M.permute(0, 2, 3, 1)
        M = M.view(M.size(0), M.size(1), M.size(2), -1, self.quat_dims)
        a = a.permute(0, 2, 3, 1)

        return M, a

# Pure quaternions --> 3 dimensional caps
class PrimaryQuatCapsNobranch(nn.Module):
    def __init__(self, in_channels, outCaps, quat_dims=3):
        super(PrimaryQuatCapsNobranch, self).__init__()
        self.activation_layer = nn.Conv2d(in_channels=in_channels, out_channels=outCaps, kernel_size=1)
        self.pose_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=outCaps * 3,
                                    kernel_size=1)

        torch.nn.init.xavier_uniform_(self.pose_layer.weight)
        self.batchNormPose = nn.BatchNorm2d(quat_dims * outCaps)
        self.batchNormA = nn.BatchNorm2d(outCaps)
        self.quat_dims = quat_dims
        self.stride = (1, 1)
        self.kernel_size = (1, 1)

    def forward(self, x):
        M = self.batchNormPose(self.pose_layer(x))
        a = torch.sigmoid(self.batchNormA(self.activation_layer(x)))
        M = M.permute(0, 2, 3, 1)
        M = M.view(M.size(0), M.size(1), M.size(2), -1, self.quat_dims)
        a = a.permute(0, 2, 3, 1)

        return M, a

class ConvQuaternionCapsLayer(QuaternionLayer):
    def __init__(self, kernel_size, stride, inCaps, outCaps, routing_iterations, routing):
        super(ConvQuaternionCapsLayer, self).__init__()

        self.W_theta = nn.Parameter(torch.zeros(1, 1, 1, *kernel_size, inCaps, outCaps, 1, 1, 1))
        if init_type == "uniform_pi":
            nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        elif init_type == "normal":
            nn.init.normal_(self.W_theta)

        self.W_hat = nn.Parameter(torch.zeros(1, 1, 1, *kernel_size, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)
        self.Beta_a = nn.Parameter(torch.zeros(1, outCaps))
        self.Beta_u = nn.Parameter(torch.zeros(1, outCaps, 1))

        self.kernel_size = kernel_size
        self.stride = stride

        self.inCaps = inCaps
        self.outCaps = outCaps
        self.routing = routing(routing_iterations).to(device)


    #   Pose Input dims      : <B, Spatial, Caps, Quaternion>
    #   Votes dim       : <B, Spatial, Caps_in, Caps_out, Quaternion>
    def forward(self, x, a):

        x = x.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]) \
            .permute(0,
                     1,
                     2,
                     5,
                     6,
                     3,
                     4).unsqueeze(6).unsqueeze(8).contiguous()


        a = a.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    4,
                                                                                                                    5,
                                                                                                                    3).contiguous()

        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=7, keepdim=True))

        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=7)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=7)
        W_conj = self.left2right * W_.permute(0, 1, 2, 3, 4, 5, 6, 8, 7)
        V = W_conj @ W_ @ x
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1)

        R = (torch.ones(*a.size(), self.outCaps) / self.outCaps).to(device)
        a = a.unsqueeze(6)
        return self.routing(V, a, self.Beta_u, self.Beta_a, R,
                            (x.size(0), x.size(1), x.size(2), self.outCaps, 4, 1))


class FCQuatCaps(QuaternionLayer):

    def __init__(self, inCaps, outCaps, quat_dims, routing_iterations, routing):
        super(FCQuatCaps, self).__init__()

        self.W_theta = nn.Parameter(torch.zeros(1, 1, 1, inCaps, outCaps, 1, 1, 1))

        if init_type == "uniform_pi":
            nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        elif init_type == "normal":
            nn.init.normal_(self.W_theta)

        self.W_hat = nn.Parameter(torch.zeros(1, 1, 1, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)
        self.Beta_a = nn.Parameter(torch.zeros(1, outCaps))
        self.Beta_u = nn.Parameter(torch.zeros(1, outCaps, 1))
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.quat_dims = quat_dims
        self.routing = routing(routing_iterations)



    def forward(self, x, a):
        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=5, keepdim=True))
        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=5)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=5)  # MAY TRANSFER THIS TO CTOR.
        W_conj = self.left2right * W_.permute(0, 1, 2, 3, 4, 6, 5)
        V = W_conj @ W_ @ x.unsqueeze(4)
        V = V.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        a = a.flatten(start_dim=1, end_dim=a.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        R = (torch.ones(*a.size(), self.outCaps) / self.outCaps).to(device)  # R: <B, Spatial, Kernel Size, Caps_in, Caps_out>
        a = a.unsqueeze(6)

        return self.routing(V, a, self.Beta_u, self.Beta_a, R,
                            (x.size(0), self.outCaps, self.quat_dims, 1))


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
        # m = min(0.1 + 0.1 * epoch, 0.9) --> Group equivariant capsule code alternative.
        if self.m < 0.9:
            self.m = 0.2 + 0.79 * torch.sigmoid(torch.min(torch.tensor([10, numiter / 50000. - 4])))
        return Loss.mean()


class MatQuatCapNet(nn.Module):

    def __init__(self):
        super(MatQuatCapNet, self).__init__()

        if branched:
            if dataset == "smallnorb" or dataset == "smallnorb_azimuth" or dataset == "smallnorb_elevation" or dataset == 'visualize':
                self.resblock1_pose = ResidualBlocks.BasicPreActResBlock(2, 32, 1)
                self.resblock1_activation = ResidualBlocks.BasicPreActResBlock(2, 32, 2)

            elif dataset == "cifar10" or dataset == 'svhn':
                self.resblock1_pose = ResidualBlocks.BasicPreActResBlock(3, 32, 1)
                self.resblock1_activation = ResidualBlocks.BasicPreActResBlock(3, 32, 2)

            else:
                self.resblock1_pose = ResidualBlocks.BasicPreActResBlock(1, 32, 1)
                self.resblock1_activation = ResidualBlocks.BasicPreActResBlock(1, 32, 2)

            self.resblock2_pose = ResidualBlocks.BasicPreActResBlock(32, 64, 2)
            self.primarycaps = PrimaryQuatCaps(in_channels_pose=64, in_channels_act=32, outCaps=32)

        else:
            if dataset == "smallnorb" or dataset == "smallnorb_azimuth" or dataset == "smallnorb_elevation" or dataset == 'visualize':
                self.resblock1 = ResidualBlocks.BasicPreActResBlock(2, 64, 1)


            elif dataset == "cifar10" or dataset == 'svhn':
                self.resblock1 = ResidualBlocks.BasicPreActResBlock(3, 64, 1)


            else:
                self.resblock1 = ResidualBlocks.BasicPreActResBlock(1, 64, 1)

            self.resblock2 = ResidualBlocks.BasicPreActResBlock(64, 64 + 32, 2)
            self.primarycaps = PrimaryQuatCapsNobranch(in_channels=96, outCaps=32)




        self.convquatcapmat1 = ConvQuaternionCapsLayer(kernel_size=(5, 5), stride=(1, 1), inCaps=32, outCaps=16,
                                                       routing_iterations=2,
                                                       routing=EMRouting)

        self.convquatcapmat2 = ConvQuaternionCapsLayer(kernel_size=(5, 5), stride=(1, 1), inCaps=16, outCaps=16,
                                                       routing_iterations=2,
                                                       routing=EMRouting)

        self.convquatcapmat3 = ConvQuaternionCapsLayer(kernel_size=(5, 5), stride=(1, 1), inCaps=16, outCaps=16,
                                                       routing_iterations=2,
                                                       routing=EMRouting)

        self.classquatcapmat = FCQuatCaps(inCaps=16, outCaps=num_class, quat_dims=4, routing_iterations=2, routing=EMRouting)

    def forward(self, x):

        if branched:
            x_pose = self.resblock2_pose(self.resblock1_pose(x))
            x_activation = self.resblock1_activation(x)
            x = self.primarycaps(F.relu(x_pose), F.relu(x_activation))
        else:
            x = self.resblock2(self.resblock1(x))
            x = self.primarycaps(F.relu(x))


        z = torch.zeros(x[0].size(0), x[0].size(1), x[0].size(2), x[0].size(3), 1).to(device)
        x_quat = torch.cat((z, x[0]), 4)

        x1 = self.convquatcapmat1(x_quat, x[1])
        l2_output = self.convquatcapmat2(x1[0].squeeze(), x1[1])
        l3_output = self.convquatcapmat3(l2_output[0].squeeze(), l2_output[1])

        x = self.classquatcapmat(l3_output[0], l3_output[1])

        return x


class st_qcn(nn.Module):

    def __init__(self):
        super(st_qcn, self).__init__()
        if dataset == "smallnorb" or dataset == "smallnorb_azimuth" or dataset == "smallnorb_elevation" or dataset == 'visualize':
            self.resblock1_pose = ResidualBlocks.BasicPreActResBlock(2, 32, 1)
            self.resblock1_activation = ResidualBlocks.BasicPreActResBlock(2, 32, 2)

        elif dataset == "cifar10" or dataset == 'svhn':
            self.resblock1_pose = ResidualBlocks.BasicPreActResBlock(3, 32, 1)
            self.resblock1_activation = ResidualBlocks.BasicPreActResBlock(3, 32, 2)

        else:
            self.resblock1_pose = ResidualBlocks.BasicPreActResBlock(1, 32, 1)
            self.resblock1_activation = ResidualBlocks.BasicPreActResBlock(1, 32, 2)

        self.resblock2_pose = ResidualBlocks.BasicPreActResBlock(32, 64, 2)

        self.primarycaps = PrimaryQuatCaps(in_channels_pose=64, in_channels_act=32, outCaps=32)
        self.st_cap = STRoutedQCLayer(inCaps=32, outCaps=5, quat_dims=3)

    def forward(self, x):

        x_pose = self.resblock2_pose(self.resblock1_pose(x))
        x_activation = self.resblock1_activation(x)
        x = self.primarycaps(F.relu(x_pose), F.relu(x_activation))


        z = torch.zeros(x[0].size(0), x[0].size(1), x[0].size(2), x[0].size(3), 1).to(device)
        x_quat = torch.cat((z, x[0]), 4)

        out = self.st_cap(x_quat.unsqueeze(5), x[1].unsqueeze(4))
        out = F.softmax(out)
        return None, out, None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numEpoch = 300

numiter = 0
epoch = 0
m = 0.9
routing_iterations = 3
train_mode = True

if loss_function == 'spread':
    loss_fn = SpreadLoss()
elif loss_function == 'crossentropy':
    loss_fn = nn.CrossEntropyLoss()
else:
    print("Select a LOSS FUNCTION!")
    sys.exit()


model = MatQuatCapNet()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)
optimizer = Adam(model.parameters(), lr=3e-3)

decay_steps = 20000
decay_rate = .96
exponential_lambda = lambda x: decay_rate ** (numiter / decay_steps)
scheduler = lr_scheduler.LambdaLR(optimizer, exponential_lambda)
novelmaxacc = 0.
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# setup writer.
if not dataset == 'visualize':
    writer = SummaryWriter(logdir=runTagGen(runtag=writer_tag))


def train_test(path='', log=True):
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
        if log:
            writer.add_scalar("Training set Accuracy", accuracy, numiter)

    return accuracy

def train(epoch):
    global numiter
    model.train()
    total_loss = 0.
    for batch_idx, (X, Y) in enumerate(Dataset_train):
        optimizer.zero_grad()
        Xgpu = X.to(device)
        Y_hat_pose_mean, Y_hat, Y_hat_pose_sigma = model(Xgpu)
        if loss_function == "spread":
            loss = loss_fn(Y_hat.cpu(), Y.squeeze())
        else:
            loss = loss_fn(Y_hat.cpu(), Y.unsqueeze(1))

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        print('NumIter: {} \t\tTrain Epoch: {} '
              '[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(numiter, epoch,
                                                         batch_idx * len(X),
                                                         len(Dataset_train.dataset),
                                                         100 * batch_idx / float(len(Dataset_train)),
                                                         loss.item()), end='\r')

        if (dataset == "smallnorb_azimuth" or dataset == "smallnorb_elevation") and (epoch > 5 and numiter % 20 == 0):

            trainacc = train_test()

            if trainacc < 96.5 and trainacc > 90:
                print("\nTESTING FOR GENERALIZATION:\n")
                test()
            
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
                       'models/{}_{}_{}_maxAcc_{}.pth'.format(dataset, init_type, loss_function, max_acc))
            writer.add_text("maximum accuracy:", "{}".format(max_acc))

        print("Test accuracy: {}%".format(accuracy))
        print("Max accuracy: {}%".format(max_acc))
        writer.add_scalar("Test Accuracy", accuracy, numiter)
        writer.add_scalar("Loss/Test Epoch Loss", total_loss, numiter)


if __name__ == "__main__":
    weight_path = ''  # SHOULD ENTER PATH FOR TEST ONLY

    if not train_mode and not weight_path:
        print("ENTER WEIGHT PATH FOR TEST MODE!")
        sys.exit()

    if train_mode:

        print(torch.__version__)
        print(count_parameters(model))
        writer.add_text("model info:",
                        "number of parameters: {}\nBatch size: {}\nInit method:\t theta --> N[0,1]l\tW_hat --> N[0,1]\n loos fn :{}".format(
                            count_parameters(model), batch_size, loss_function))


        print("Initiating Training...")
        for epoch in range(numEpoch):
            train(epoch)

            if loss_function == "spread":
                writer.add_scalar("m-schedule:", loss_fn.m, numiter)

            writer.add_scalar("learning rate:", scheduler.get_lr(), numiter)
            if dataset != "smallnorb_azimuth" or dataset != "smallnorb_elevation":
                test()
                train_test()
            scheduler.step()
            print("lr: {}".format(scheduler.get_lr()))
            
            torch.save(model.state_dict(), 'models/{}_{}_{}_{:03d}Quaternion_caps_paperArch.pth'.format(dataset, init_type,
                                                                                                          loss_function, epoch + 1))

    elif dataset == 'visualize':


        model.load_state_dict(torch.load(weight_path))
        output_list = []
        for batch_idx, (X, Y, info) in enumerate(Dataset_train):
            Y = Y.squeeze()
            Xgpu = X.to(device)
            Y_hat_pose_mean, Y_hat, Y_hat_pose_sigma = model(Xgpu)
            print(info)
            output_list.append(Y_hat_pose_mean)


    else:
        print("Initiating test only mode...")
        test(path=weight_path, log=False)

