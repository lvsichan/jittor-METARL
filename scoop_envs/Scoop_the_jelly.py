# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:32:57 2020

@author: yexiaohan
"""

import taichi as ti
import numpy as np
import math
import random
from scoop_envs.MPM_solver import MPMSolver

import jittor
from scoop_envs.AE_CNN import AutoEncoder
# from jittor.autograd import Variable

ti.init(arch=ti.gpu) # Try to run on GPU
#CUDA_VISIBLE_DEVICES=['0','1']

seed = 1024
jittor.set_global_seed(seed) 

random.seed(seed)    
np.random.seed(seed)  

class ScoopJellyEnvs(MPMSolver):
    def __init__(self,n_tasks,train_tasks,eval_tasks,path,gui):
        self.gui=gui
        self.n_tasks=n_tasks['n_tasks']
        self.cnt=0
        self.model = AutoEncoder()
        self.model.load_parameters(jittor.load(path))
        self.model.eval()
        scale=0.8
        self.summ=0
        self.epo=0
        #self.model = jittor.nn.DataParallel(self.model)
        '''
        #for meta test
        train_qualitys=np.random.randint(200,300,train_tasks)*0.01
        eval_qualitys=np.random.randint(160,161,eval_tasks)*0.025
        self.qualitys=np.append(train_qualitys,eval_qualitys)
        '''
        self.qualitys=np.random.randint(20,31,self.n_tasks)*0.1 #for meta train
        self.Es=np.random.randint(3,15,self.n_tasks)*100.0
        self.aas=np.random.randint(10,11,self.n_tasks)*0.1
        self.quality=float(self.qualitys[0])
        self.E=self.Es[0]
        
        srand=np.random.randint(1,4,size=(self.n_tasks,12))*0.4+1.0
        self.projections=np.random.randint(1,2,size=self.n_tasks)
        self.projection=self.projections[0]
        self.jelly_phos=srand*1.0
        self.jelly_pho=self.jelly_phos[0]
        #self.jelly_pho=np.array([23, 17,  3, 16,  8, 20, 10,  4, 22, 19,  4, 18])*0.1
        
        srand=np.random.randint(8,9,self.n_tasks)*40.0
        self.fluid_phos=srand*0.01
        self.fluid_pho=self.fluid_phos[0]
        
        srand=np.zeros([self.n_tasks,3])
        for i in range(self.n_tasks):
            srand[i][0]=scale*(np.random.randint(0,3)*0.04+0.08)#length
            srand[i][1]=scale*(np.random.randint(0,1)*0.06+0.00)#height
            srand[i][2]=scale*(np.random.randint(0,3)*0.04+0.12)#bottom

        self.shapes=srand
        self.shape=self.shapes[0]
        
        srand=np.random.randint(1,4,self.n_tasks)*20.0#-20.0
        self.gravitys=srand
        self.gravity=self.gravitys[0]
        
        srand=np.zeros([self.n_tasks,10])
        for i in range(self.n_tasks):
            srand[i]=random.sample(range(0, 10), 10)
        #srand=np.random.randint(0,2,size=(self.n_tasks,12))
        self.r_types=srand
        self.r_type=self.r_types[0]
        
        super().__init__(self.quality,self.E,self.jelly_pho,self.fluid_pho,self.gravity,self.projection,self.r_type,self.shape)
        self.observation_space=self.get_observation_space()
        self.action_space=np.zeros(3)
        
    def get_all_task_idx(self):
        return range(self.n_tasks)
    def reset_task(self,idx):
        
        ti.reset()
        ti.init(arch=ti.gpu) # Try to run on GPU
        self.quality=float(self.qualitys[idx])
        self.E=self.Es[idx]
        self.jelly_pho=self.jelly_phos[idx]
        self.fluid_pho=self.fluid_phos[idx]
        self.gravity=self.gravitys[idx]
        self.projection=self.projections[idx]
        self.r_type=self.r_types[idx]
        self.shape=self.shapes[idx]
        self.reset(self.quality,self.E,self.jelly_pho,self.fluid_pho,self.gravity,self.projection,self.r_type,self.shape)
        
        self.reset_env()

    def get_observation_space(self):
        alpha=0.1
        state=self.get_state().reshape([-1,2,128,128])

        #state[np.isnan(state)] = 0
        mask=np.zeros(state.shape)
        mask[:,:,4:124,4:124]=1
        state[mask==0]=0
        
        sta=jittor.Var(state)
        sta = jittor.Var(sta.float())
        #print(jittor.std(sta))
        #print(jittor.std(sta))
        sta=sta/(jittor.std(sta)+1e-7)
        
        encode=self.model.encode(sta)
        encode=encode.cpu().detach().numpy()
        
        #decode=decode.cpu().detach().numpy()
        #print(np.mean(np.square(decode-state)))
        
        #print(encode)
        
        a = np.append(self.get_jelly_state(), self.get_rigid_state())

        obs = np.append(a, alpha*encode)

        #return obs
        #print(a)
        return obs#a
        
        
    def step(self,action):
        
        for s in range(int(12.0e-3 / self.dt+0.1)):

            self.solve_windmill(action[0]*40,action[1]*40,action[2]*80)
            self.substep()
        
        obs=self.get_observation_space()
        reward=self.get_reward()
        done=0
        self.cnt+=1
        
        if self.cnt==118:
            self.epo+=1
            if self.epo%5!=1:
                self.summ+=self.maxx[None]
            #print(self.maxx[None],self.summ,self.epo)  
            
        #print(reward)
        #if self.bounded[None]==1:
        #    done=1
        return obs,reward,done,dict(reward=reward)
        
    def reset_env(self):
        self.o=random.sample(range(0, 5), 5)
        #print(self.maxx[None])
        self.initialize()
        self.cnt=0
        return self.get_observation_space()
    def render(self):
        
        #colors = np.array([0x008B8B, 0xFF6347, 0xEEEEF0], dtype=np.uint32)
        color_jelly=np.array([0xff7fff,
                              0xff99ff,
                              0xffb2ff,
                              0xffd1ff,
                              0xfff4ff,0x84bf96,0x000000],dtype=np.uint32)
        self.gui.circles(self.x.to_numpy(), radius=1.5, color=color_jelly[self.color.to_numpy()])
        self.gui.circles(self.windmill_x.to_numpy(), radius=1.5,color=0x845538)
        #self.gui.lines(self.begin.reshape([-1,2]),self.end.reshape([-1,2]), radius=1,color=0xd71345)
        self.gui.show()
        
        

'''
gui = ti.GUI("MPM SCOOP THE JELLY", res=512, background_color=0x112F41)

#model='./music_model/autoencoder16.pkl'
model=1
mpm=ScoopJellyEnvs({'n_tasks':200},200,0,model,gui)
velocity_256=np.zeros([1000,128,128,2])
flag=0
for epoch in range(200):
    print(epoch)
    itt=0
    mpm.reset_task(epoch)
    for frame in range(1000):    
        if frame%100==0:
            mpm.reset_env()
            flag=0
        x=mpm.jelly_pos[6]-0.176
        x_=mpm.jelly_pos[6]-0.226
        y=mpm.jelly_pos[7]-0.1
        x0=mpm.center[None][0]
        y0=mpm.center[None][1]
        dis=math.sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))
        dis_=math.sqrt((x_-x0)*(x_-x0)+(y-y0)*(y-y0))
        vx=(x-x0)/dis*2
        vy=(y-y0)/dis*2
        vx_=(x_-x0)/dis_*2
        vy_=(y-y0)/dis_*2
        #print(dis_)
        if dis_<0.02:
            flag==1
        if dis<0.02:
            flag=2
        if flag==0:
            action = np.array([vx_,vy_,0])
        elif flag==1:
            action = np.array([vx,vy,0])
        else:
            action=np.array([vx,2.0,0])
        mpm.step(action)
        velocity_256[itt]=mpm.get_state()
        itt=itt+1
        mpm.render()
    #np.save("./scoop_data/%d_velocity.npy"%epoch,velocity_256)
'''





