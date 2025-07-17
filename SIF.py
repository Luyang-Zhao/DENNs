import torch
import DENNs
from Embedding import extendAxisNet,multiEmbedding,Embedding
from utils.Geometry import Geometry1D,LocalAxis
import numpy as np
from sklearn.linear_model import LinearRegression

def get_delta_u(model:DENNs.PINN2D,
                embedding:Embedding,
                crack_surface:Geometry1D,num,
                local_axis:LocalAxis):
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    '''下表面位移直接求出来'''
    embedding.set_ls(-1.0)
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    embedding.set_ls(1.0)
    u_up,v_up = model.pred_uv(xy)
    embedding.restore_ls()

    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)

    r = local_axis.getR(x,y).unsqueeze(-1) * 1000

    return r , delta_u.unsqueeze(-1) , delta_v.unsqueeze(-1)



def DispExpolation_homo(model:DENNs.PINN2D,
                        embedding:Embedding,
                    crack_surface:Geometry1D,num,
                    local_axis:LocalAxis,
                    kappa,mu):
    

    '''SIF单位:MPa*sqrt(mm)'''
    r , delta_u , delta_v = get_delta_u(model,embedding,crack_surface,num,local_axis)
    r_sqrt = torch.sqrt(r)

    material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)
    K1_bar = material_coefficient * delta_v / r_sqrt
    K2_bar = material_coefficient * delta_u / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    K1_model = LinearRegression() 
    K1_model.fit(r, K1_bar)  
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  
    K2_model.fit(r, K2_bar)  
    K2 = K2_model.intercept_
    return K1,K2
                                        

def DispExpolation_bimaterial(model:DENNs.PINN2D,
                    embedding:Embedding,
                    crack_surface:Geometry1D,num,
                    local_axis:LocalAxis,
                    kappa_up,mu_up,kappa_low,mu_low):
    

    '''SIF单位:MPa*sqrt(mm)'''
    r , delta_u , delta_v = get_delta_u(model,embedding,crack_surface,num,local_axis)
    r_sqrt = torch.sqrt(r)

 
    eps = np.log( (kappa_up / mu_up + 1 / mu_low) / (kappa_low/ mu_low + 1 / mu_up) ) / (2*np.pi)

    Q = eps * torch.log(r/1000) #Q=eps*ln(r/2a)

    C = 2 * np.cosh(eps * np.pi)  * np.sqrt(np.pi * 2) / (kappa_up / mu_up + 1 / mu_low + kappa_low/ mu_low + 1 / mu_up)

    cosQ = torch.cos(Q)
    sinQ = torch.sin(Q)

    e1 = cosQ + 2 * eps * sinQ
    e2 = sinQ - 2 * eps * cosQ

    K1_bar = C * (delta_v * e1 + delta_u * e2) / r_sqrt 
    K2_bar = C * (delta_u * e1 - delta_v * e2) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    K1_model = LinearRegression()  
    K1_model.fit(r, K1_bar)  
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  
    K2_model.fit(r, K2_bar) 
    K2 = K2_model.intercept_

    return K1,K2

def max_stress_theta(K1,K2):
    '''
    最大环向应力计算扩展角度
    '''
    
    '''环向应力极值方向,包含最大值与最小值'''
    theta = np.arctan(np.array([
                (K1 + np.sqrt(K1**2+8*K2**2)) / (4*K2) ,
                (K1 - np.sqrt(K1**2+8*K2**2)) / (4*K2) ,
            ]))
    theta = np.arccos(np.array([
        (3*K2**2 + np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2),
        (3*K2**2 - np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2)
    ]))
    theta = theta[theta<np.pi/2]
    theta = np.concatenate((theta,-theta),0)
    
    '''计算环向应力'''
    stress_theta = np.cos(theta/2) * (K1*(1+np.cos(theta)) - 3*K2*np.sin(theta))
    
    return theta[np.argmax(stress_theta)]
