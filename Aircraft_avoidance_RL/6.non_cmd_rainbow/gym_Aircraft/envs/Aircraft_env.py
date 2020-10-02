import gym
import numpy as np
import random
import warnings
from gym import error, spaces, utils
from gym.utils import seeding

Deg2Rad = np.pi/180
Rad2Deg = 1/Deg2Rad
g = 9.8
K_alt = .8*2          # hdot loop gain
RoC = 20              # maximum rate of climb (max. of hdot)
AoA0 = -1.71*Deg2Rad     # zero lift angle of attack
Acc2AoA = 0.308333*Deg2Rad  # 1m/s^2 ACC corresponds to 0.308333deg AOA
zeta_ap = 0.7         # pitch acceleration loop damping
omega_ap = 4          # pitch acceleration loop bandwidth

dist_sep = 100        # near mid-air collision range

dt = 0.1              # control frequency
tf = 30               # final time
t = np.arange(0, tf, dt)
N = len(t)


def model(z, t, hdot_cmd):  # computes state derivatives
    Vm = 200
    a, adot, h, hdot, R = z  # state vector: a (pitch acc), adot, h (alt), hdot, R (ground-track range)
    gamma = np.arcsin(hdot / Vm)  # fight path angle
    ac = K_alt * (hdot_cmd - hdot) + g / np.cos(gamma)  # pitch acceleration command
    ac = np.clip(ac, -30, 30)  # maneuver limit

    addot = omega_ap * omega_ap * (ac - a) - 2 * zeta_ap * omega_ap * adot
    hddot = a * np.cos(gamma) - g
    Rdot = Vm * np.cos(gamma)
    return np.array([adot, addot, hdot, hddot, Rdot])  # returns state derivatives

class AircraftEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,size=5):
        super(AircraftEnv,self).__init__()
        self.size=size
        self._state=np.zeros(size)

        self.h_cmd_count=0
        
        self.t_step=0
        self.hm0 = 1000
        self.Vm = 200
        self.gamma0 = 0*Deg2Rad
        self.Pm_NED = np.array([0, 0, -self.hm0])
        self.Vm_NED = np.array([self.Vm * np.cos(self.gamma0), 0, -self.Vm * np.sin(self.gamma0)])
        self.X0 = np.array([g / np.cos(self.gamma0), 0, self.hm0, -self.Vm_NED[2], 0])


        # target initial conditions
        self.ht0 = 1000 + 200 * np.random.randn()
        self.Vt = 200
        self.approach_angle = 50 * Deg2Rad * (2 * np.random.rand() - 1)
        self.psi0 = np.pi + self.approach_angle + 2 * np.random.randn() * Deg2Rad
        self.psi0 = np.arctan2(np.sin(self.psi0), np.cos(self.psi0))

        self.Pt_N = 2000 * (1 + np.cos(self.approach_angle))
        self.Pt_E = 2000 * np.sin(self.approach_angle)
        self.Pt_D = -self.ht0
        self.Pt_NED = np.array([self.Pt_N, self.Pt_E, self.Pt_D])  # initial NED position
        self.Vt_NED = np.array([self.Vt * np.cos(self.psi0), self.Vt * np.sin(self.psi0), 0])  # initial NED velocity

        # initialize variables
        self.X = np.zeros((N, len(self.X0)))
        self.X[0, :] = self.X0
        self.dotX_p = 0

        self.theta0 = self.gamma0 + self.X0[0] * Acc2AoA + AoA0  # initial pitch angle

        self.DCM = np.zeros((3, 3))  # initial DCM NED-to-Body
        self.DCM[0, 0] = np.cos(self.theta0)
        self.DCM[0, 2] = -np.sin(self.theta0)
        self.DCM[1, 1] = 1
        self.DCM[2, 0] = np.sin(self.theta0)
        self.DCM[2, 2] = np.cos(self.theta0)

        self.Pr_NED = self.Pt_NED - self.Pm_NED  # relative NED position
        self.Vr_NED = self.Vt_NED - self.Vm_NED  # relative NED velosity

        self.Pr_Body = np.dot(self.DCM, self.Pr_NED)  # relative position (Body frame)

        # radar outputs
        self.r = np.linalg.norm(self.Pr_Body)  # range
        self.vc = -np.dot(self.Pr_NED, self.Vr_NED) / self.r  # closing velocity
        self.elev = np.arctan2(self.Pr_Body[2], self.Pr_Body[0])  # target vertival look angle (down +)
        self.azim = np.arctan2(self.Pr_Body[1], self.Pr_Body[0] / np.cos(self.theta0))  # target horizontal look angle (right +)

        self.los = self.theta0 - self.elev  # line of sight angle
        self.dlos = 0
        self.daz = 0

        # static variables
        self.los_p = self.los
        self.dlos_p = self.dlos
        self.azim_p = self.azim
        self.daz_p = self.daz
        self.hdot_cmd = 0
        self.hdot = 0
        self.gamma = self.gamma0
        self.vc0 = self.vc



        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, -400, -np.pi, -2*np.pi, -2*np.pi]),
                                                        np.array([6000, 400, np.pi, 2*np.pi, 2*np.pi])) # r, vc, los, daz, dlos`

    def step(self, action):
        done = False
        reward = 0

        if self.t_step>len(t)-1:
            reward=0
            done=True
        if self.r>=5000:
            reward=0
            done=True
        if self.r<=dist_sep:
            reward=-1
            done=True
        if self.t_step>3 and self.r>dist_sep and abs(self.elev)>40*Deg2Rad and abs(self.azim)>40*Deg2Rad:
            reward=0
            done=True

        if not done:
            if action == 0:
                if self.hdot_cmd!=0:
                     self.h_cmd_count+=1
                self.hdot_cmd=0
            elif action == 1:
                if self.hdot_cmd!=-10:
                     self.h_cmd_count+=1
                self.hdot_cmd=-10
            elif action == 2:
                if self.hdot_cmd!=10:
                     self.h_cmd_count+=1
                self.hdot_cmd=10
            else:
                warnings.warn("The action should be 0 or 1 or 2 but other was detected.")
            
            
            self.dotX = model(self.X[self.t_step, :], t[self.t_step], self.hdot_cmd)
            self.X[self.t_step + 1, :] = self.X[self.t_step, :] + 0.5 * (3 * self.dotX - self.dotX_p) * dt
            self.dotX_p = self.dotX
            self.Pt_NED = self.Pt_NED + self.Vt_NED * dt


            self.a, self.adot, self.h, self.hdot, self.R = self.X[self.t_step+1,:]

            self.gamma = np.arcsin(self.hdot/self.Vm)
            self.theta = self.gamma + self.a*Acc2AoA + AoA0

            self.DCM = np.zeros((3,3))
            self.DCM[0,0] =  np.cos(self.theta)
            self.DCM[0,2] = -np.sin(self.theta)
            self.DCM[1,1] =  1
            self.DCM[2,0] =  np.sin(self.theta)
            self.DCM[2,2] =  np.cos(self.theta)

            self.Pm_NED = np.array([self.R, 0, -self.h])
            self.Vm_NED = np.array([self.Vm*np.cos(self.gamma), 0, -self.Vm*np.sin(self.gamma)])

            self.Pr_NED = self.Pt_NED - self.Pm_NED
            self.Vr_NED = self.Vt_NED - self.Vm_NED

            self.Pr_Body = np.dot(self.DCM, self.Pr_NED)

            self.r = np.linalg.norm(self.Pr_Body)
            self.vc = -np.dot(self.Pr_NED, self.Vr_NED)/self.r
            self.elev = np.arctan2(self.Pr_Body[2], self.Pr_Body[0])
            self.azim = np.arctan2(self.Pr_Body[1], self.Pr_Body[0]/np.cos(self.theta))

            psi = np.arctan2(self.Vt_NED[1], self.Vt_NED[0])

            # los rate and az rate estimation
            self.los = self.theta - self.elev

            self.dlos = ( 30*(self.los-self.los_p) + 0*self.dlos_p ) / 3 # filtered LOS rate, F(s)=20s/(s+20)
            self.daz = ( 30*(self.azim-self.azim_p) + 0*self.daz_p ) / 3 # filtered azim rate, F(s)=20s/(s+20)

            self.los_p = self.los
            self.dlos_p = self.dlos
            self.azim_p = self.azim
            self.daz_p = self.daz
            self._state=np.array([self.r,self.vc,self.los,self.daz,self.dlos])
            self.t_step+=1


        return self._state.flatten(),reward,done,[self.hdot_cmd,self.r,self.elev,self.azim,self.Pm_NED,self.Pt_NED,self.h]


    def reset(self):
        self.__init__(size=self.size)
        return self._state
    def render(self,mode='human',close=False):
        pass
