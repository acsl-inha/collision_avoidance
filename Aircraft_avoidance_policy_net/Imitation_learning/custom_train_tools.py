import random
import time
import torch
import numpy as np
import sys
import logging
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch import Tensor
from torchsummary import summary
from torch.utils.data import Dataset

#####################################
########## Hyper Parmeters ##########
g = 9.8                                 # gravity acceleration
K_alt = .8*2                            # hdot loop gain
RoC = 20                                # maximum rate of climb (max. of hdot)
AoA0 = -1.71*np.pi/180                  # zero lift angle of attack
Acc2AoA = 0.308333*np.pi/180            # 1m/s^2 ACC corresponds to 0.308333deg AOA
zeta_ap = 0.7                           # pitch acceleration loop damping
omega_ap = 4                            # pitch acceleration loop bandwidth
dist_sep = 100                          # near mid-air collision range
t = np.arange(0, 30, 0.1)               # time range for each episode
N = len(t)                              # number of time samples
total_sim = 500                         # number of test simulations
total_epoch = 100                       # number of train epochs
mean = np.load('mean.npy').tolist()     # load average of dataset
std = np.load('std.npy').tolist()       # load standard division of dataset
#####################################


def system_log(log_file):                                       # define system logger

    system_logger = logging.getLogger()
    system_logger.setLevel(logging.INFO)

    out_put_file_handler = logging.FileHandler(log_file)        # handler for log text file
    stdout_handler = logging.StreamHandler(sys.stdout)          # handler for log printing

    system_logger.addHandler(out_put_file_handler)
    system_logger.addHandler(stdout_handler)

    return system_logger


class CustomDataset(Dataset):                                   # define custom dataset (x: r, vc, los, daz, dlos), (y: hcmd)
    def __init__(self, path):
        xy = np.loadtxt(path,
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy[:, 0:5])
        xy = xy.astype('int_')
        self.y_data = torch.tensor(xy[:, 5])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class FClayer(nn.Module):                                       # define fully connected layer with Leaky ReLU activation function
    def __init__(self, innodes, nodes):
        super(FClayer, self).__init__()
        self.fc = nn.Linear(innodes, nodes)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = self.act(out)
        return out


class WaveNET(nn.Module):                                       # define custom model named wave net, which was coined after seeing the nodes sway
    def __init__(self, block, planes, nodes, num_classes=3):
        super(WaveNET, self).__init__()
        self.innodes = 5

        self.layer1 = self._make_layer(block, planes[0], nodes[0])
        self.layer2 = self._make_layer(block, planes[1], nodes[1])
        self.layer3 = self._make_layer(block, planes[2], nodes[2])

        self.fin_fc = nn.Linear(self.innodes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, nodes):

        layers = []
        layers.append(block(self.innodes, nodes))
        self.innodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.innodes, nodes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fin_fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def train_model(num_layers, _nodes, lr, batch_size, train_loader, val_loader, model_char, system_logger):       # Function for train model

    model = WaveNET(FClayer, num_layers, _nodes).cuda()
    system_logger.info(summary(model, (1, 5)))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)

    saving_path = "./res_model/"
    trn_loss_list = []                                      # list for saving train loss
    val_loss_list = []                                      # list for saving validation loss
    val_acc_list = []

    for epoch in range(total_epoch):
        trn_loss = 0.0

        # train model
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            output = model(inputs)
            # calculate loss
            loss = criterion(output, labels)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()
            # train loss summary
            trn_loss += loss.item()

        # validation
        with torch.no_grad():
            val_loss = 0.0
            cor_match = 0
            for j, val in enumerate(val_loader):
                val_x, val_label = val
                if torch.cuda.is_available():
                    val_x = val_x.cuda()
                    val_label = val_label.cuda()
                val_output = model(val_x)
                v_loss = criterion(val_output, val_label)
                val_loss += v_loss
                _, predicted = torch.max(val_output, 1)
                cor_match += np.count_nonzero(predicted.cpu().detach()
                                              == val_label.cpu().detach())
        # update lr scheduler
        scheduler.step()

        # calculate average of train loss and validation loss for all batch
        trn_loss_list.append(trn_loss/len(train_loader))
        val_loss_list.append(val_loss/len(val_loader))
        val_acc = cor_match/(len(val_loader)*batch_size)
        val_acc_list.append(val_acc)
        now = time.localtime()

        # print logs
        system_logger.info("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year,
                                                  now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
        system_logger.info("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | val accuracy: {:.4f}% \n".format(
            epoch+1, total_epoch, trn_loss /
            len(train_loader), val_loss / len(val_loader), val_acc*100
        ))
    # save model
    model_name = saving_path+"Custom_model_"+model_char+"_fin"
    torch.save(model, model_name)
    return model


def integral_model(z, t, hdot_cmd, Vm):                                     # hdot loop dynamics definition
    a, adot, h, hdot, R = z
    gamma = np.arcsin(hdot/Vm)
    ac = K_alt * (hdot_cmd - hdot) + g/np.cos(gamma)
    ac = np.clip(ac, -30, 30)

    addot = omega_ap*omega_ap*(ac-a) - 2*zeta_ap*omega_ap*adot
    hddot = a*np.cos(gamma) - g
    Rdot = Vm*np.cos(gamma)
    return np.array([adot, addot, hdot, hddot, Rdot])


def test_model(cmd_char, choose_ht, model, model_char, system_logger):
    # data for simulation that target satisfied insight condition
    hdot_flag = 0
    res_Y = np.zeros(((N, 7, total_sim)))
    hdot_change_count = []

    # test loop
    while True:
        insight = 0                 # check if target ship is in sight

        # player initial conditions
        hm0 = 1000                  # initial altitude
        Vm = 200                    # initial speed
        gamma0 = 0                  # initial flight path angle
        Pm_NED = np.array([0, 0, -hm0])                                             # initial NED position
        Vm_NED = np.array([Vm*np.cos(gamma0), 0, -Vm*np.sin(gamma0)])               # initial NED velocity

        X0 = np.array([g/np.cos(gamma0), 0, hm0, -Vm_NED[2], 0])                    # initial state vector

        # target initial conditions
        # initial altitude
        if choose_ht == 0:
            ht0 = 1000 + 10+abs(50*np.random.randn())
        elif choose_ht == 1:
            ht0 = 1000 - 10-abs(50*np.random.randn())
        elif choose_ht == 2:
            if (random.choice([True, False])):
                ht0 = 1000 + 120+10*np.random.randn()
            else:
                ht0 = 1000 - 120-10*np.random.randn()
        else:
            system_logger.info("Error, invalid value for choose ht")
        ##########################################################
        ################# Initialize environment #################
        Vt = 200                                                                    # initial velocity
        approach_angle = 50*np.pi/180*(2*np.random.rand()-1)                        # initial approach angle
        psi0 = np.pi + approach_angle + 2*np.random.randn()*np.pi/180
        psi0 = np.arctan2(np.sin(psi0), np.cos(psi0))

        Pt_N = 2000*(1+np.cos(approach_angle))
        Pt_E = 2000*np.sin(approach_angle)
        Pt_D = -ht0
        Pt_NED = np.array([Pt_N, Pt_E, Pt_D])                                       # initial NED position
        Vt_NED = np.array([Vt*np.cos(psi0), Vt*np.sin(psi0), 0])                    # initial NED velocity

        # initialize variables
        X = np.zeros((N, len(X0)))                                                  # state vector list
        X[0, :] = X0
        dotX_p = 0                                                                  # initial state variation
        theta0 = gamma0 + X0[0]*Acc2AoA + AoA0                                      # initial pitch angle

        # initial DCM NED-to-Body
        DCM = np.zeros((3, 3))
        DCM[0, 0] = np.cos(theta0)
        DCM[0, 2] = -np.sin(theta0)
        DCM[1, 1] = 1
        DCM[2, 0] = np.sin(theta0)
        DCM[2, 2] = np.cos(theta0)
        Pr_NED = Pt_NED - Pm_NED                                                    # relative NED position
        Vr_NED = Vt_NED - Vm_NED                                                    # relative NED velosity
        Pr_Body = np.dot(DCM, Pr_NED)                                               # relative position (Body frame)

        # radar outputs
        r = np.linalg.norm(Pr_Body)                                                 # range
        vc = -np.dot(Pr_NED, Vr_NED)/r                                              # closing velocity
        elev = np.arctan2(Pr_Body[2], Pr_Body[0])                                   # target vertical look angle (down +)
        azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta0))                    # target horizontal look angle (right +)
        los = theta0 - elev                                                         # line of sight angle
        dlos = 0
        daz = 0

        Y = np.zeros((N, 7))                                                        # print-out data
        Y[0, :] = np.array([*Pm_NED, *Pt_NED, r])

        # static variables
        los_p = los
        dlos_p = dlos
        azim_p = azim
        daz_p = daz
        hdot_cmd = 0
        count_change_hdot = 0
        vc0 = vc
        ##########################################################

        # main loop (simulation)
        for k in range(N-1):

            # update environment (adams-bashforth 2nd order integration)
            dotX = integral_model(X[k, :], t[k], hdot_cmd, Vm)
            X[k+1, :] = X[k, :] + 0.5*(3*dotX-dotX_p)*0.1
            dotX_p = dotX
            Pt_NED = Pt_NED + Vt_NED*0.1

            # get observation
            a, adot, h, hdot, R = X[k+1, :]
            gamma = np.arcsin(hdot/Vm)
            theta = gamma + a*Acc2AoA + AoA0

            DCM = np.zeros((3, 3))
            DCM[0, 0] = np.cos(theta)
            DCM[0, 2] = -np.sin(theta)
            DCM[1, 1] = 1
            DCM[2, 0] = np.sin(theta)
            DCM[2, 2] = np.cos(theta)

            Pm_NED = np.array([R, 0, -h])
            Vm_NED = np.array([Vm*np.cos(gamma), 0, -Vm*np.sin(gamma)])
            Pr_NED = Pt_NED - Pm_NED
            Vr_NED = Vt_NED - Vm_NED
            Pr_Body = np.dot(DCM, Pr_NED)

            r = np.linalg.norm(Pr_Body)
            vc = -np.dot(Pr_NED, Vr_NED)/r
            elev = np.arctan2(Pr_Body[2], Pr_Body[0])
            azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta))
            los = theta - elev

            # los rate and az rate estimation
            dlos = (30*(los-los_p) + 0*dlos_p) / 3
            daz = (30*(azim-azim_p) + 0*daz_p) / 3

            los_p = los
            dlos_p = dlos
            azim_p = azim
            daz_p = daz

            # compute action by model
            if k > 3 and r > dist_sep and abs(elev) < 40*np.pi/180 and abs(azim) < 40*np.pi/180:
                insight += 1
                data = torch.tensor(((np.array([r, vc, los, daz, dlos])
                                      - mean)/std).astype(np.float32)).cuda()
                output = model(data.view(-1, 5))
                _, predicted = torch.max(output, 1)
                if predicted[0] == 0:
                    hdot_cmd = 0
                if predicted[0] == 1:
                    if hdot_cmd != -20:
                        count_change_hdot += 1
                    hdot_cmd = -20
                if predicted[0] == 2:
                    if hdot_cmd != 20:
                        count_change_hdot += 1
                    hdot_cmd = 20
            elif k > 3:
                hdot_cmd = 0

            Y[k+1, :] = np.array([*Pm_NED, *Pt_NED, r])
        if insight > 0:
            hdot_change_count.append(count_change_hdot)
            res_Y[:, :, hdot_flag] = Y
            hdot_flag += 1
        if hdot_flag == total_sim:
            break

    # check test performance
    total_cor_mean = 0
    err = 0
    cor = 0
    cor_sum = 0
    disy = np.zeros(total_sim)
    for i in range(total_sim):
        disy[i] = min(res_Y[:, 6, i])
        if min(res_Y[:, 6, i]) < dist_sep:
            err += 1
        else:
            cor_sum += min(res_Y[:, 6, i])
            cor += 1
    cor_mean = cor_sum/cor
    total_cor_mean += cor_mean

    system_logger.info("error with test down sim {}: {}".format(total_sim, err))
    system_logger.info("Mean avoiding distance of correct avoidance with correction {}: {}".format(
        cor, cor_mean))

    plt.figure(figsize=(15, 15))
    sns.set(color_codes=True)
    sns.distplot(disy)
    plt.savefig("./res_img/"+cmd_char+model_char+".png", dpi=300)
    plt.close()
