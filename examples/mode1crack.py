from DENNs import PINN2D
import torch
import torch.nn as nn
from utils.NodesGenerater import genMeshNodes2D
from utils.NN import stack_net,AxisScalar2D
from utils.Integral import trapz1D
import numpy as np
from Embedding import LineCrackEmbedding,extendAxisNet
import Embedding
import matplotlib.pyplot as plt
import utils.Geometry as Geometry



class Plate(PINN2D):
    def __init__(self, model: nn.Module,fy):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy

    def hard_u(self, u, x, y):
        return u * x
    
    def hard_v(self, v, x, y):
        return v * (y+1)/2
    
    def add_BCPoints(self,num = [128]):
        x_up,y_up=genMeshNodes2D(0,1,num[0],1,1,1)
        self.x_up,self.y_up,self.xy_up = self._set_points(x_up ,y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u_up,v_up = self.pred_uv(self.xy_up)
        u_up,v_up=self.mm_to_m(u_up,v_up)

        return trapz1D(v_up * self.fy, self.x_up)


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可



E=100.0e3 ; nu = 0.3
fy=10.0


y_crackTip = 0.0

x_crackCenter = 0.0
y_crackCenter = 0.0
# beta = torch.pi/4
beta = 0.0
x_crackTip = 0.5
a=x_crackTip

model_name = 'examples/result/test'
crack_embedding = LineCrackEmbedding([x_crackCenter,y_crackCenter],
                                            [x_crackTip,y_crackTip],
                                            tip = 'right')


net = extendAxisNet(
        net = AxisScalar2D(
            stack_net(input=3,output=2,activation=nn.Tanh,width=30,depth=4),
            A=torch.tensor([2.0,1.0,1.0]),
            B=torch.tensor([-1.0,0.0,0.0])
            ),
        extendAxis= crack_embedding)
pinn = Plate(net,fy=fy)


pinn = Plate(net,fy=fy)
pinn.add_BCPoints()

pinn.setMaterial(E=E , nu = nu,type='plane strain')

pinn.set_loss_func(losses=[pinn.Energy_loss,
                                    ],
                            weights=[1000.0]
                                    )


pinn.set_meshgrid_trapz_Tip_Dense(0.0001,0.9998,-0.9999,0.9998,
                                    x_crackTip,y_crackTip,
                                    50,50,30,50)

pinn.train(path=model_name,patience=10,epochs=15000,lr=0.02)

pinn.load(path=model_name)

# pinn.readData('reference/mode1crack.txt')

print(pinn.Energy_loss().cpu().detach().numpy())

# pinn.evaluate()
# pinn.showPrediction(pinn.labeled_xy)


from SIF import DispExpolation_homo

kappa = (3-4*nu)
mu = E/(2*(1+nu))

crack_surface = Geometry.LineSegement([0.0,0.0],[x_crackTip,0.0])


crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.35*a),
                                    crack_surface.clamp(dist2=0.3*a))



K1 , K2 = DispExpolation_homo(pinn,
                            crack_embedding,  
                            crack_surface,5,
                            Geometry.LocalAxis(a,0.0,0.0),
                            kappa,mu)

b = 1.0
a_b = a/b
a_mm = a * 1000
normalized_Param = fy * np.sqrt(np.pi * a_mm)

K1_true_coefficient = (1-0.025*a_b**2+0.06*a_b**4) * np.sqrt(1/np.cos(np.pi*a_b/2))

print(K1)
print(K1/normalized_Param)
print(normalized_Param * K1_true_coefficient)
print(K1_true_coefficient)


