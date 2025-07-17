from DENNs import PINN2D_bimaterial
import torch
import torch.nn as nn
from utils.NN import stack_net,AxisScalar2D
from utils.Integral import trapz1D
import Embedding
import matplotlib.pyplot as plt
import numpy as np
from utils.Geometry import LineSegement
import utils.Geometry as Geometry
import utils.NodesGenerater as NodesGenerater

class Plate(PINN2D_bimaterial):
    def __init__(self, model: nn.Module,fy):
        super().__init__(model)
        self.fy = fy

    def hard_u(self, u, x, y):
        return u * x
        # return u * x * (1-x)
    
    def hard_v(self, v, x, y):
        return v * (y+2) / 4
    
    def add_BCPoints(self,num = [500]):
        x_up,y_up=NodesGenerater.genMeshNodes2D(1e-4,1-1e-4,num[0],2,2,1)
        self.x_up,self.y_up,self.xy_up = self._set_points(x_up ,y_up)
        self.up_zero = torch.zeros_like(self.x_up)


    def E_ext(self) -> torch.Tensor:
        u,v = self.pred_uv(self.xy_up)
        u,v=self.mm_to_m(u,v)
        return trapz1D(v * self.fy, self.x_up)

    
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可
'''平面应力'''
E2 = 1.0e3
gamma = 10.0
E1 = gamma * E2
v1=0.3;v2 = 0.3

a=0.5
fy=1.0

x_crackTip = a
y_crackTip = 0.0

x_crackCenter = 0.0
y_crackCenter = 0.0
beta = 0.0

local_axis = Embedding.LocalAxis(x0=x_crackTip,y0=0.0,beta=0.0)
crack_surface = LineSegement([0.0,0.0],[x_crackTip,0.0])
meaterial_surface = LineSegement([0.0,0.0],[1.0,0.0])




crack_embedding = Embedding.LineCrackEmbedding([x_crackCenter,y_crackCenter],
                                               [x_crackTip,y_crackTip],
                                               tip = 'right')


multiEmbedding = Embedding.multiEmbedding([crack_embedding,
                                        Embedding.InterfaceEmbedding(meaterial_surface)])


wid = 30

x_dense_num=100
y_dense_num=100
x_outer_num=100
y_outer_num=150
x_inteval=0.2
y_inteval=0.5

net = Embedding.extendAxisNet(
        net = AxisScalar2D(
            stack_net(input=4,output=2,activation=nn.Tanh,width=wid,depth=4),
            A=torch.tensor([2.0,0.5,1.0,0.5]),
            B=torch.tensor([-1.0,0.0,0.0,0.0])
            ),
        extendAxis= multiEmbedding)
pinn = Plate(net,fy=fy)


pinn.add_BCPoints()

pinn.setMaterial(E1=E1,nu1=v1,
                 E2=E2,nu2=v2)

pinn.set_LevelSet(meaterial_surface)



pinn.set_meshgrid_trapz_Tip_Dense(0.0001,0.9998,-1.9999,1.9998,
                                    x_crackTip,y_crackTip,
                                    x_dense_num,
                                    y_dense_num,
                                    x_outer_num,
                                    y_outer_num,
                                    x_inteval,y_inteval)
# NodesGenerater.plot_points(0,1,-2,2,pinn.XY.cpu().detach().numpy())

pinn.set_loss_func(losses=[pinn.Energy_loss],
                   weights=[1000.0])

model_name = 'examples/result/bimaterial_crack/gamma'+str(gamma)

pinn.train(path=model_name,epochs=20000,patience=10,lr=0.02,milestones=[5000,10000,15000])

pinn.load(path=model_name)

# pinn.evaluate(name=model_name)
# pinn.evaluate()
print(pinn.Energy_loss())


pinn.set_meshgrid_inner_points(0,1,50,-1,1,100)
print(pinn.Energy_loss().cpu().detach().numpy())
pinn.showPrediction(pinn.XY)



from SIF import DispExpolation_bimaterial


def get_kappa(v):
    return (3-v)/(1+v)

def get_mu(E,v):
    return (E/(2*(1+v)))

kappa_up = get_kappa(v1)
mu_up = get_mu(E1,v1)

kappa_low = get_kappa(v2)
mu_low = get_mu(E2,v2)


extrapolation_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.4),
                                      crack_surface.clamp(dist2=0.2))

K1,K2 = DispExpolation_bimaterial(pinn,
                          crack_embedding,
                          extrapolation_surface,5,
                          local_axis,
                          kappa_up,mu_up,kappa_low,mu_low)

# #gamma=10
# F1 = 1.136
# F2 = -0.182

sigma_pi_a = fy * np.sqrt(np.pi * a * 1000)

len = np.sqrt(1000)

print(K1/sigma_pi_a)
print(K2/sigma_pi_a)

print(K1)
print(K2)
