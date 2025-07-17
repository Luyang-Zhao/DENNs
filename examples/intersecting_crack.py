from DENNs import PINN2D
import torch
import torch.nn as nn
from utils.NodesGenerater import genMeshNodes2D
from utils.NN import stack_net
from utils.Integral import trapz1D
import numpy as np
import Embedding
import utils.Geometry as Geometry
from Embedding import LineCrackEmbedding


class Plate(PINN2D):
    def __init__(self, model: nn.Module,fy):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy

    def hard_u(self, u, x, y):
        return u * (1 - x) * (1 + x)
    
    def hard_v(self, v, x, y):
        return v * (y+1)/2
    
    def add_BCPoints(self,num = [256]):
        x_up,y_up=genMeshNodes2D(-1,1,num[0],1,1,1)
        self.x_up,self.y_up,self.xy_up = self._set_points(x_up ,y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u_up,v_up = self.pred_uv(self.xy_up)
        u_up,v_up=self.mm_to_m(u_up,v_up)

        return trapz1D(v_up * self.fy, self.x_up) 
    

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
'''两条互相交叉的裂纹'''
'''平面应力'''
E=100.0e3 ; nu = 0.3
fy=10.0

embedding_1 = LineCrackEmbedding(xy0=[0.5,0.5],xy1=[-0.5,-0.5],tip='both')
embedding_2 = LineCrackEmbedding(xy0=[0.5,-0.5],xy1=[-0.5,0.5],tip='both')


multiEmbedding = Embedding.multiEmbedding([embedding_1,embedding_2])

net = Embedding.extendAxisNet(
        net = stack_net(input=4,output=2,activation=nn.Tanh,
                          width=30,depth=4),
        extendAxis= multiEmbedding)

pinn = Plate(net,fy=fy)
pinn.add_BCPoints()

pinn.setMaterial(E=E , nu = nu)

pinn.set_meshgrid_inner_points(-1+1e-5,1-1e-4,250,-1+1e-4,1-1e-5,250)

pinn.set_loss_func(losses=[pinn.Energy_loss,
                                      ],
                              weights=[1000.0]
                                       )


# # Evaluate
# pinn.readData('result/cross_crack/model_in_paper/cross_crack.txt')
# crack_line_1 = Geometry.LineSegement([0.49,0.49],[-0.49,-0.49])
# index_1 = crack_line_1.is_on_geometry(pinn.labeled_xy,eps=1e-4)
# crack_line_2 = Geometry.LineSegement([0.49,-0.49],[-0.49,0.49])
# index_2 = crack_line_2.is_on_geometry(pinn.labeled_xy,eps=1e-4)

# index = index_1 | index_2

# labeled_xy = pinn.labeled_xy[~index]
# labeled_u,labeled_v = pinn.labeled_u[~index] , pinn.labeled_v[~index]
# labeled_sx,labeled_sy,labeled_sxy = pinn.labeled_sx[~index] , pinn.labeled_sy[~index] , pinn.labeled_sxy[~index]


# u_disp_ref = pinn.displacement(labeled_u,labeled_v).cpu().detach().numpy()
# mises_ref = pinn.stressToMises(labeled_sx,labeled_sy,labeled_sxy).cpu().detach().numpy()

# def record_item():
#     u,v,sx,sy,sxy = pinn.infer(labeled_xy)
#     u_disp = pinn.displacement(u,v).cpu().detach().numpy()
#     mises = pinn.stressToMises(sx,sy,sxy).cpu().detach().numpy()
#     hist = [pinn.rmse(u_disp,u_disp_ref) , pinn.rmse(mises,mises_ref)]
#     print(hist)
#     return hist
# pinn.record_item = record_item

model_name = 'examples/result/intersecting_crack/DENNs'

pinn.train(path=model_name,patience=10,epochs=30000,lr=0.02)

pinn.load(path=model_name)
# pinn.evaluate(model_name,levels=100)
# pinn.evaluate(name=None,levels=100)
pinn.showPrediction(pinn.XY)
print(pinn.Energy_loss())

