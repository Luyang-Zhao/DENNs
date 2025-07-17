from DENNs import PINN2D
import torch
import torch.nn as nn
import utils.NodesGenerater as NodesGenerater
from utils.NN import stack_net
from utils.Integral import trapz1D
import Embedding
import matplotlib.pyplot as plt
import numpy as np
from utils.Geometry import LineSegement
import utils.Geometry as Geometry
from SIF import max_stress_theta,DispExpolation_homo

class Plate(PINN2D):
    def __init__(self, model: nn.Module,fy):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy

    def hard_u(self, u, x, y):
        return u * (y+1) /2
    
    def hard_v(self, v, x, y):
        return v * (1-y) * (y+1) * (x+1) * (1-x)
    
    def add_BCPoints(self,num = [500]):
        x_up,y_up=NodesGenerater.genMeshNodes2D(-1,1,num[0],1,1,1)
        self.x_up,self.y_up,self.xy_up = self._set_points(x_up ,y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u_up,v_up = self.pred_uv(self.xy_up)
        u_up,v_up=self.mm_to_m(u_up,v_up)

        return trapz1D((u_up) * self.fy, self.x_up)

    
    
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可



def is_crack_in_domain(points:list[list]):
    points_end = points[-1]
    if max(np.abs(points_end)) < 1 : return True
    else: return False


def get_RELUPSI_extension(crack_points,tip='right'):
    return Embedding.multiLineCrackEmbedding(crack_points,tip = tip)




def train(from_scratch = False):

    i = i_start
    lr = lr_init
    while is_crack_in_domain(points):
        print('step'+str(i+1))
        if i > 20: break
        embedding = get_embedding(points,tip = 'right')

        if from_scratch:
            pinn = init_model()
        else:
            # 将第一轮训练结果作为初始化参数
            if i>0:
                pinn.load(path=folder+model_name+str(0))

        pinn.model.set_extend_axis(embedding)

        pinn.set_meshgrid_trapz_Tip_Dense(-0.9999,1,-1,0.9999,
                                    points[-1][0],points[-1][1],
                                    120,120,130,130,x_inteval=0.25,y_inteval=0.25)

        pinn.train(path=folder+model_name+str(i),
                epochs=epochs,patience=10,lr=lr,eval_sep=100)
        
        # 读取最好的epoch
        pinn.load(path=folder+model_name+str(i))
        
        crack_surface = Geometry.LineSegement(points[-2],points[-1])
        '''计算裂纹张开位移时略微远离裂尖端'''
        crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.12),
                                                crack_surface.clamp(dist2=0.08))
        
        K1 , K2 = DispExpolation_homo(pinn,
                                embedding,
                                crack_surface,5,
                                Geometry.LocalAxis(points[-1][0],points[-1][1],
                                beta = crack_surface.tangent_theta),
                                kappa,mu)
        print(K1,K2)

        '''最大应力计算裂纹扩展方向'''
        ref_theta = max_stress_theta(K1,K2)
        open_theta = crack_surface.tangent_theta.numpy() + ref_theta
        print(open_theta * 180 / np.pi)

        '''裂纹扩展'''
        points.append([points[-1][0] + a_increment * np.cos(open_theta),
                    points[-1][1] + a_increment * np.sin(open_theta)])
        torch.save(points,folder+model_name+'points.pt')
        print(points)
        i+=1
        # lr=0.005
        lr=lr_else
        
    torch.save(points,folder+model_name+'points.pt')
    print(points)


def eval():

    points_set = torch.load(folder+model_name+'points.pt')
    pinn.set_meshgrid_inner_points(-1,1,100,-1,1,100)

    for i in range(len(points_set) - 2):
        print('step'+str(i+1))
        pinn.load(path=folder+model_name+str(i))
        embedding = get_embedding(points,tip='right')
        pinn.model.set_extend_axis(
            embedding
        )

        phi = embedding.getGamma(pinn.XY)
        # plt.figure
        pinn.plotContourf(pinn.X.cpu().detach().numpy(),
                        pinn.Y.cpu().detach().numpy(),
                        phi.cpu().detach().numpy(),
                        show=True,ax=plt.subplot(1,1,1)) 
        
        u,v = pinn.pred_uv(pinn.XY)


        plt.figure(figsize=(12, 4))
        plt.rcParams.update({'font.size': 12})

        pinn.plotContourf(pinn.X.cpu().detach().numpy(),
                        pinn.Y.cpu().detach().numpy(),
                        u.cpu().detach().numpy(),
                        show=False,ax=plt.subplot(1,2,1))    
        plt.plot(np.array(points)[...,0],np.array(points)[...,1],color='white',linewidth=2)
        # plt.axis('on')
        # plt.show()

        pinn.plotContourf(pinn.X.cpu().detach().numpy(),
                        pinn.Y.cpu().detach().numpy(),
                        v.cpu().detach().numpy(),
                        show=False,ax=plt.subplot(1,2,2))    
        plt.plot(np.array(points)[...,0],np.array(points)[...,1],color='white',linewidth=2)
        # plt.axis('on')
        plt.show()


        crack_surface = Geometry.LineSegement(points[-2],points[-1])
        '''计算裂纹张开位移时略微远离裂尖端'''
        crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.12),
                                                crack_surface.clamp(dist2=0.08))

        K1 , K2 = DispExpolation_homo(pinn,
                                embedding,
                                crack_surface,5,
                                Geometry.LocalAxis(points[-1][0],points[-1][1],
                                beta = crack_surface.tangent_theta),
                                kappa,mu)
        print(K1,K2)

        '''最大应力计算裂纹扩展方向'''
        ref_theta = max_stress_theta(K1,K2)
        print(ref_theta * 180 / np.pi)
        print(crack_surface.tangent_theta.numpy() * 180 / np.pi)
        open_theta = crack_surface.tangent_theta.numpy() + ref_theta
        # open_theta = ref_theta
        print(open_theta * 180 / np.pi)

        points.append(points_set[i+init_points_num])
        print(points)


def plot_animation(path=None):
    # import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(20, 4))
    num = 5

    plt.rcParams.update({'font.size': 12})
    axes = [plt.subplot(1,num,i+1) for i in range(num)]
    divs = [make_axes_locatable(ax) for ax in axes]
    caxes = [div.append_axes('right', '5%', '5%') for div in divs]


    points_set = torch.load(folder+model_name+'points.pt')
    pinn.set_meshgrid_inner_points(-1,1,100,-1,1,100)

    def plot(i):

        points_tmp = points_set[:i+init_points_num]
        pinn.load(path=folder+model_name+str(i))
        pinn.model.set_extend_axis(
            get_embedding(crack_points=points_tmp)
        )


        F = pinn.infer(pinn.XY)

        for j in range(num):

            plot = pinn.plotContourf(pinn.X.cpu().detach().numpy(),
                                pinn.Y.cpu().detach().numpy(),
                                F[j].cpu().detach().numpy(),
                                show=False,ax=axes[j],cbar=False,levels=200)    
            pinn.plot_cbar(ax=axes[j],plot=plot,cax=caxes[j])
            axes[j].plot(np.array(points_tmp)[...,0],np.array(points_tmp)[...,1],color='white',linewidth=1)
        
        return (axes[j] for j in range(num))

    ani = animation.FuncAnimation(fig, plot, frames=len(points_set)-init_points_num,interval=500) 
    
    if path == None:
        plt.show()
    else:
        ani.save(path+'.gif')


def plot(step=[0,2,4,8]):

    import matplotlib
    plt.rcParams.update({'font.size': 10})
    matplotlib.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    matplotlib.rcParams['mathtext.default'] = 'regular'

    pinn.set_meshgrid_inner_points(-0.99,0.99,100,-0.99,0.99,100)

    # F = []
    points_set = torch.load(folder+model_name+'points.pt')
    fig,axes = plt.subplots(5,len(step),figsize=(6.0,5.6),layout='constrained')
    for i,istep in enumerate(step):
        points_tmp = points_set[:istep+init_points_num]
        pinn.load(path=folder+model_name+str(istep))
        
        pinn.model.set_extend_axis(
            get_embedding(crack_points=points_tmp)
        )

        F = pinn.infer(pinn.XY)


        for j in range(5):
            ax=axes[j][i]
            pinn.plotContourf(pinn.X.cpu().detach().numpy(),
                                            pinn.Y.cpu().detach().numpy(),
                                            F[j].cpu().detach().numpy(),
                                            show=False,ax=ax,cbar=True,levels=100)  
            ax.plot(np.array(points_tmp)[...,0],np.array(points_tmp)[...,1],color='white',linewidth=0.5)
            ax.axis('off')
        

    plt.savefig('fig/'+'prop.svg', dpi=600,bbox_inches='tight')
        

get_embedding = get_RELUPSI_extension

x_crackTip = 0.0
y_crackTip = 0.0
x_crackCenter = -1.0
y_crackCenter = 0.0
# a=0.5

init_points_num = 3

a_increment = 0.15

folder = 'examples/result/shear_propagation/'
model_name = ''

points=[[x_crackCenter,y_crackCenter],
        [x_crackTip-a_increment,y_crackTip],
        [x_crackTip,y_crackTip]]


lr_init=0.02
lr_else = 0.02
epochs = 15000

i_start = 0

print(points)
E = 200.0e3
nu = 0.3

# kappa = (3-4*nu)
kappa = (3-nu)/(1+nu)
mu = E/(2*(1+nu))

def init_model()->Plate:
    net = Embedding.extendAxisNet(
            net = stack_net(input=3,output=2,activation=nn.Tanh,
                            width=30,depth=4),
            extendAxis= get_embedding(points,tip = 'right'))
    pinn = Plate(net,fy=5.0)

    pinn.add_BCPoints()
    pinn.setMaterial(E,nu)
    pinn.set_loss_func(losses=[pinn.Energy_loss],
                    weights=[1000.0])    
    return pinn


pinn = init_model()

train()

# model_name = 'left_shear_crack_polar_30_4_90000monto_0.15increment_newIntegral/'
# plot_animation(path='plate/edge_shear_crack/'+model_name+'result_full')
# plot_animation()
eval()
# plot()