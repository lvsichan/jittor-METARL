    # -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:32:57 2020

@author: yexiaohan
"""

import taichi as ti
import numpy as np
import math
import random
@ti.data_oriented
class MPMSolver:
    def __init__(self,quality,E_,mass,rho,G,P,R,S):
        self.reset(quality,E_,mass,rho,G,P,R,S)
    def reset(self,quality,E_,mass,rho,G,P,R,S):
        self.P=P
        self.rho=rho
        self.G=G

        self.E_=E_
        self.quality = quality # Use a larger value for higher-res simulations
        self.n_particles, self.n_grid = int(6256 * self.quality ** 2), int(64 * self.quality)
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.sample_size=128
        self.sample_dx,self.sample_inv_dx=1/self.sample_size,self.sample_size
        self.dt = 2.0e-4 / self.quality
        self.p_vol = (self.dx * 0.5)**2
        self.p_rho = ti.field(dtype=ti.f32, shape=self.n_particles) # material id
        self.p_mass = ti.field(dtype=ti.f32, shape=self.n_particles) # material id
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles) # position
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles) # velocity
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_particles) # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_particles) # deformation gradient
        self.material = ti.field(dtype=ti.i32, shape=self.n_particles) # material id
        self.Jp = ti.field(dtype=ti.f32, shape=self.n_particles) # plastic deformation
        self.sample_v=ti.Vector.field(2, dtype=ti.f32, shape=(self.sample_size, self.sample_size))
        self.sample_m=ti.field(dtype=ti.f32, shape=(self.sample_size, self.sample_size))
        self.mask=ti.field(dtype=ti.f32,shape=(self.sample_size,self.sample_size))
        self.grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid)) # grid node momentum/velocity
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid)) # grid node mass
        self.fluid_m=ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.jelly_m=ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.a=ti.Vector.field(3,dtype=ti.f32, shape=())
        self.jelly_pos=ti.field(dtype=ti.f32, shape=10)
        self.jelly_vel=ti.field(dtype=ti.f32, shape=10)
        #initial jelly mass for each task
        self.jelly_mass=ti.field(dtype=ti.f32,shape=5)
        self.R=ti.field(dtype=ti.int32,shape=5)
        self.color=ti.field(dtype=ti.int32,shape=self.n_particles)
        self.order=ti.field(dtype=ti.i32,shape=5)
        self.o=np.zeros([5])
        self.S=ti.field(dtype=ti.f32,shape=4)
        self.maxx=ti.field(dtype=ti.i32,shape=())
        
        '''
        self.begin=ti.Vector.field(2, dtype=ti.f32, shape=(self.sample_size,self.sample_size))
        self.end=ti.Vector.field(2, dtype=ti.f32, shape=(self.sample_size,self.sample_size))
        self.begin1=np.zeros([self.sample_size,self.sample_size,2])
        self.end1=np.zeros([self.sample_size,self.sample_size,2])
        '''
        self.begin=np.zeros([128,128,2])
        self.end=np.zeros([128,128,2])
        # scoop

        pi = math.pi
        
        self.center = ti.Vector.field(2,dtype=ti.f32, shape=())
        self.angle = ti.field(dtype=ti.f32, shape=())
        self.omega = ti.field(dtype=ti.f32, shape=())
        self.vel = ti.Vector.field(2,dtype=ti.f32, shape=())
        self.l_down=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.l_up=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.r_down=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.r_up=ti.Vector.field(2,dtype=ti.f32, shape=())
        I = 1e-2
        self.grid_assit = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid)) # grid node momentum/velocity
        
        #windmill
        self.windmill_n_particles=int(384* quality ** 2)
        self.windmill_x=ti.Vector.field(2, dtype=ti.f32, shape=self.windmill_n_particles) # position
        self.windmill_mass = ti.field(dtype=ti.f32, shape=self.windmill_n_particles)
        self.rigid_size1=int(96*self.quality**2)
        self.rigid_size2=int(self.rigid_size1+96*self.quality**2)
        self.rigid_size3=int(self.rigid_size2+96*self.quality**2)
        self.rigid_size4=int(self.rigid_size3+96*self.quality**2)
        
        
        
        self.fluid_size = int(6000* self.quality ** 2)
        #self.jelly_size = int(self.n_particles//160)+1
        self.jelly_size = int((self.n_particles-self.fluid_size)/5)
        self.tmp_x = ti.Vector.field(2, dtype=ti.f32, shape=self.jelly_size)
        self.point=ti.Vector.field(2,dtype=ti.f32,shape=4)
        #print(R)
        for i in range(5):
            self.jelly_mass[i]=mass[i]
            self.R[i]=0
            
        self.o=random.sample(range(0, 5), 5)
        
        for i in range(3):
            self.S[i]=S[i]
        self.length = self.S[0]
        self.a_radius = 0.2*0.8
        self.b_radius = 0.16*0.8
        self.thick = 0.015*0.8
        '''
        cnt=0
        for n in range(1,10):
          for m in range(int(3*n*self.quality)):
            r=1.0*n/9
            alpha=math.pi*(m/(1.5*n*self.quality)-1)
            a=r*ti.cos(alpha)
            b=r*ti.sin(alpha)
            self.tmp_x[cnt]=ti.Vector.field([a*0.015,b*0.015])
            cnt+=1
            '''
        #self.initialize()
    
    @ti.kernel
    def solve_windmill(self,a:ti.f32,b:ti.f32,c:ti.f32):   #action:[ax,ay,aw]
      self.vel[None][0]+=self.dt*a
      self.vel[None][1]+=self.dt*b
      self.omega[None]+=self.dt*c
      
      self.vel[None][0]=ti.min(self.vel[None][0],3.0)
      self.vel[None][0]=ti.max(self.vel[None][0],-3.0)
      self.vel[None][1]=ti.min(self.vel[None][1],3.0)
      self.vel[None][1]=ti.max(self.vel[None][1],-3.0)
      self.omega[None]=ti.min(self.omega[None],6.0)
      self.omega[None]=ti.max(self.omega[None],-6.0)
     
      
      self.angle[None]+=self.dt*self.omega[None]
      #self.angle[None]=min(self.angle[None],3.5)
      #self.angle[None]=max(self.angle[None],-3.5)
      self.center[None]+=self.dt*self.vel[None]

      rot1 = ti.Vector([ti.cos(self.angle[None]), -ti.sin(self.angle[None])])
      rot2 = ti.Vector([ti.sin(self.angle[None]), ti.cos(self.angle[None])])

      wr = ti.Vector([self.center[None][1]-self.l_up[None][1], self.l_up[None][0]-self.center[None][0]])
      wr = wr*self.omega[None]
      self.l_up[None] += (self.vel[None]+wr)*self.dt
      wr = ti.Vector([self.center[None][1]-self.l_down[None][1], self.l_down[None][0]-self.center[None][0]])
      wr = wr*self.omega[None]
      self.l_down[None] += (self.vel[None]+wr)*self.dt
      wr = ti.Vector([self.center[None][1]-self.r_up[None][1], self.r_up[None][0]-self.center[None][0]])
      wr = wr*self.omega[None]
      self.r_up[None] += (self.vel[None]+wr)*self.dt
      wr = ti.Vector([self.center[None][1]-self.r_down[None][1], self.r_down[None][0]-self.center[None][0]])
      wr = wr*self.omega[None]
      self.r_down[None] += (self.vel[None]+wr)*self.dt
      #ti.block_dim(32)
      out=0
      for p in self.windmill_x:
        # vertex
        # p = transform[None]@ti.Vector.field([pos[i][0], pos[i][1], 1.])
        # pos[i] = [p[0], p[1]]

        if self.windmill_x[p][0]<0.02:
            out=1
        elif self.windmill_x[p][0]>0.98:
            out=2
        elif self.windmill_x[p][1]<0.01:
            out=3
        elif self.windmill_x[p][1]>0.96:
            out=4
      if out==1:
          self.vel[None][0]=0.5*abs(self.vel[None][0])
          self.omega[None]=-0.0*self.omega[None]
      elif out==2:
          self.vel[None][0]=-0.5*abs(self.vel[None][0])
          self.omega[None]=-0.0*self.omega[None]
      elif out==3:
          self.vel[None][1]=0.5*abs(self.vel[None][1])
          self.omega[None]=-0.0*self.omega[None]
      elif out==4:
          self.vel[None][1]=-0.5*abs(self.vel[None][1])
          self.omega[None]=-0.0*self.omega[None]
      for p in self.windmill_x:
        # vertex
        # p = transform[None]@ti.Vector.field([pos[i][0], pos[i][1], 1.])
        # pos[i] = [p[0], p[1]]

        wr = ti.Vector([self.center[None][1]-self.windmill_x[p][1], self.windmill_x[p][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.windmill_x[p]+=(self.vel[None]+wr)*self.dt
      self.point[0]=self.l_up[None]

      self.point[1] = self.r_up[None]

      self.point[2] = self.l_down[None]

      self.point[3] = self.r_down[None]

        
    @ti.kernel
    def substep(self):
      E, nu = self.E_, 0.2 # Young's modulus and Poisson's ratio
      mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
      for i, j in self.grid_m:
        if i<self.sample_size and j<self.sample_size:
          self.sample_v[i,j]=[0,0]
          self.sample_m[i,j]=0
        self.grid_v[i, j] = [0, 0]
        self.grid_m[i, j] = 0
        self.grid_assit[i, j] = [0, 0]
      ti.loop_config(block_dim=32)
      for p in self.x: # Particle state update and scatter to grid (P2G)
        base = (self.x[p] * self.inv_dx - 0.5).cast(int)
        fx = self.x[p] * self.inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        
        base_sample = (self.x[p] * self.sample_inv_dx - 0.5).cast(int)
        fx_sample = self.x[p] * self.sample_inv_dx - base_sample.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        sample_w = [0.5 * (1.5 - fx_sample) ** 2, 0.75 - (fx_sample - 1) ** 2, 0.5 * (fx_sample - 0.5) ** 2]
        
        self.F[p] = (ti.Matrix.identity(ti.f32, 2) + self.dt * self.C[p]) @ self.F[p] # deformation gradient update
        h = ti.exp(10 * (1.0 - self.Jp[p])) # Hardening coefficient: snow gets harder when compressed
        if self.material[p] == 1: # jelly, make it softer
          h = 1.0
        mu, la = mu_0 * h, lambda_0 * h
        if self.material[p] == 0: # liquid
          mu = 0.0
        U, sig, V = ti.svd(self.F[p])
        J = 1.0
        for d in ti.static(range(2)):
          new_sig = sig[d, d]
          self.Jp[p] *= sig[d, d] / new_sig
          sig[d, d] = new_sig
          J *= new_sig
        if self.material[p] == 0:  # Reset deformation gradient to avoid numerical instability
          self.F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
        elif self.material[p] == 2:
          self.F[p] = U @ sig @ V.transpose() # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
        affine = stress + self.p_mass[p] * self.C[p]
        
        if self.material[p] == 0 and self.P:
          self.Jp[p]=(1 + self.dt * self.C[p].trace()) * self.Jp[p]
          st = -self.dt * 4 * E * self.p_vol * (self.Jp[p] - 1) / self.dx**2
          affine = ti.Matrix([[st, 0], [0, st]]) + self.p_mass[p] * self.C[p]
        
        for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
          offset = ti.Vector([i, j])
          dpos = (offset.cast(float) - fx) * self.dx
          weight = w[i][0] * w[j][1]
          
          dpos_sample=(offset.cast(float) - fx_sample) * self.sample_dx
          weight_sample = sample_w[i][0] * sample_w[j][1]
          self.sample_v[base_sample + offset] += weight_sample * (self.p_mass[p] * self.v[p] + affine @ dpos_sample)
          self.sample_m[base_sample + offset] += weight_sample * self.p_mass[p]
          
          self.grid_assit[base + offset] += weight * (affine @ dpos)
          self.grid_v[base + offset] += weight * (self.p_mass[p] * self.v[p] + affine @ dpos)
          self.grid_m[base + offset] += weight * self.p_mass[p]

      
      ti.loop_config(block_dim=32)
      for p in self.windmill_x:
        windmill_C = ti.Matrix.zero(ti.f32, 2, 2)
        windmill_F = ti.Matrix.identity(ti.f32, 2)  #ti.Matrix([1., 0.],[0., 1.])
        base = (self.windmill_x[p] * self.inv_dx - 0.5).cast(int)
        fx = self.windmill_x[p] * self.inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        h = 1.0 # Hardening coefficient: snow gets harder when compressed
        mu, la = mu_0 * h, lambda_0 * h
        U, sig, V = ti.svd(windmill_F)
        J = 1.0 
        stress = 2 * mu * (windmill_F - U @ V.transpose()) @ windmill_F.transpose() + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
        affine = stress + self.windmill_mass[p] * windmill_C
        for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
          offset = ti.Vector([i, j])
          dpos = (offset.cast(float) - fx) * self.dx
          weight = w[i][0] * w[j][1]
          wr = ti.Vector([self.center[None][1]-self.windmill_x[p][1], self.windmill_x[p][0]-self.center[None][0]])
          wr = wr*self.omega[None]
          new_vel=self.vel[None]+wr
          self.grid_v[base + offset] += weight * (self.windmill_mass[p] * new_vel + affine @ dpos)
          self.grid_m[base + offset] += weight * self.windmill_mass[p]
    
      for i, j in self.grid_m:
        if self.grid_m[i, j] > 0: # No need for epsilon here
          self.grid_v[i, j] = (1 / self.grid_m[i, j]) * self.grid_v[i, j] # Momentum to velocity
          self.grid_v[i, j][1] -= self.dt * self.G # gravity
        self.grid_v[i,j][1]=ti.min(self.grid_v[i,j][1],10)
        self.grid_v[i,j][0]=ti.min(self.grid_v[i,j][0],10)
        self.grid_v[i,j][1]=ti.max(self.grid_v[i,j][1],-10)
        self.grid_v[i,j][0]=ti.max(self.grid_v[i,j][0],-10)
        
        if i<self.sample_size and j<self.sample_size:
          if self.sample_m[i, j] > 1e-8: # No need for epsilon here
            self.sample_v[i, j] = (1 / self.sample_m[i, j]) * self.sample_v[i, j] # Momentum to velocity
            self.sample_v[i, j][1] -= self.dt * self.G
          self.sample_v[i,j][1]=ti.min(self.sample_v[i,j][1],10)
          self.sample_v[i,j][0]=ti.min(self.sample_v[i,j][0],10)
          self.sample_v[i,j][1]=ti.max(self.sample_v[i,j][1],-10)
          self.sample_v[i,j][0]=ti.max(self.sample_v[i,j][0],-10)
          
        if i < self.quality and self.grid_v[i, j][0] < 0:
            n = ti.Vector([1, 0])
            nv = self.grid_v[i, j].dot(n)
            self.grid_v[i, j] = self.grid_v[i, j] - n * nv
        if i > self.n_grid - self.quality and self.grid_v[i, j][0] > 0:
            n = ti.Vector([-1, 0])
            nv = self.grid_v[i, j].dot(n)
            self.grid_v[i, j] = self.grid_v[i, j] - n * nv
        if j < self.quality and self.grid_v[i, j][1] < 0:
            n = ti.Vector([0, 1])
            nv = self.grid_v[i, j].dot(n)
            self.grid_v[i, j] = self.grid_v[i, j] - n * nv
        if j > self.n_grid - self.quality  and self.grid_v[i, j][1] > 0:
            n = ti.Vector([0, -1])
            nv = self.grid_v[i, j].dot(n)
            self.grid_v[i, j] = self.grid_v[i, j] - n * nv
      
      ti.loop_config(block_dim=32)
      for p in self.x: # grid to particle (G2P)
        base = (self.x[p] * self.inv_dx - 0.5).cast(int)
        fx = self.x[p] * self.inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
          dpos = ti.Vector([i, j]).cast(float) - fx
          g_v = self.grid_v[base + ti.Vector([i, j])]
          weight = w[i][0] * w[j][1]
          new_v += weight * g_v
          new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
        self.v[p], self.C[p] = new_v, new_C
        self.x[p] += self.dt * self.v[p] # advection
    
    
    
    @ti.func
    def random_point_in_unit_sphere(self):
            ret = ti.Vector.zero(dt=ti.f32, n=2)
            while True:
                for i in ti.static(range(2)):
                    ret[i] = ti.random(ti.f32) * 2 - 1
                if ret.norm_sqr() <= 1:
                    break
            return ret
        
    @ti.kernel
    def initialize(self):
      self.maxx[None]=0
      for i, j in self.grid_m:
        if i<self.sample_size and j<self.sample_size:
          self.sample_v[i,j]=[0,0]
        self.grid_v[i, j] = [0, 0]
        self.grid_m[i, j] = 0
        self.grid_assit[i, j] = [0, 0]
      k=int(ti.sqrt(self.fluid_size/(0.5/0.8)))
      unit=0.8/k
      a1=2-ti.random()*4
      a2=2-ti.random()*4
      for i in range(self.n_particles):
        if i<self.fluid_size:
          a=int(i/k)
          b=i%k
          if i<0.5*self.fluid_size: 
            self.x[i] = [0.05+ti.random()*0.9  , 0.05+0.25*ti.random() ]
            self.v[i]=ti.Matrix([a1,0])
          else:
            self.x[i] = [0.05+ti.random()*0.9  , 0.36+0.25*ti.random() ]
            self.v[i]=ti.Matrix([a2,0])
          self.p_rho[i]=self.rho
          self.p_mass[i]=self.p_rho[i]*self.p_vol
          self.material[i] = 0 # 0: fluid 1: jelly 
          self.color[i]=5
        else:
          idx=int((i-self.fluid_size)/self.jelly_size)#error
          t=(i-self.fluid_size)%self.jelly_size
          if idx>=4:
              idx=4
          #x[i]=[ti.random()*0.05+0.1+0.08*((i-fluid_size)//jelly_size),ti.random()*0.05+0.65]
          self.x[i]=ti.Vector([0.2+0.15*idx,0.33])+self.random_point_in_unit_sphere()*0.02
          self.material[i]=1
          
          self.p_rho[i]=self.jelly_mass[idx]
          self.p_mass[i]=self.p_rho[i]*self.p_vol
          #self.color[i]=int(5-self.p_rho[i]/0.4)
          self.color[i]=0
          if self.R[idx]==1:
            self.color[i]=6
        #self.v[i] = ti.Matrix([0,ti.random()*-8.0])
            self.v[i]=ti.Matrix([0,0])
        self.F[i] = ti.Matrix([[1, 0], [0, 1]])
        self.Jp[i] = 1
      
      # initialize windmill
      p=0.05+0.55*ti.random()
      self.center[None] = [p,0.94-self.S[1]]
      center_point=[p,0.94]
      #self.center[None] = [0.3,0.90]
      self.l_up[None]=center_point+ti.Vector([self.length+self.thick,0.0])
      self.l_down[None]=center_point+ti.Vector([self.length+self.thick+self.S[2]-self.b_radius,-self.a_radius-self.thick])
      self.r_down[None]=center_point+ti.Vector([self.length+self.thick+self.S[2],-self.a_radius-self.thick])
      self.r_up[None]=center_point+ti.Vector([self.length+self.thick+self.b_radius,0.0])
      self.angle[None] = 0.
      self.omega[None] = 0#-2.5
      self.vel[None]=ti.Vector([0,-3.0])
      for i in range(self.windmill_n_particles):
        self.windmill_mass[i]=self.p_vol*30
        if i<self.rigid_size1:
          self.windmill_x[i]=[ti.random() * (self.length) + self.center[None][0] , -ti.random() * self.thick + self.center[None][1] ]
        elif i<self.rigid_size2:
          a=ti.random()
          self.windmill_x[i]=[a*self.l_down[None][0]+(1-a)*self.l_up[None][0]-self.thick*ti.random() , a*self.l_down[None][1]+(1-a)*self.l_up[None][1] ]
        elif i<self.rigid_size3:
          a=ti.random()*(self.r_down[None][0]-self.l_down[None][0])+self.l_down[None][0]
          '''
          if self.S[2]>=0.1:
              if a-self.center[None][0]-self.length>0.02 and a-self.center[None][0]-self.length<0.055:
                  a=self.l_down[None][0]
          if self.S[2]>=0.2:
              if a-self.center[None][0]-self.length>0.135 and a-self.center[None][0]-self.length<0.17:
                  a=self.l_down[None][0]+0.15
          if self.S[2]>=0.3:
              if a-self.center[None][0]-self.length>0.215 and a-self.center[None][0]-self.length<0.25:
                  a=self.l_down[None][0]+0.27
                  '''
          self.windmill_x[i]=[a , -ti.random() * self.thick - self.a_radius + self.center[None][1]+self.S[1] ]
        else:
          a=ti.random()
          self.windmill_x[i]=[a*self.r_down[None][0]+(1-a)*self.r_up[None][0]+self.thick*ti.random() , a*self.r_down[None][1]+(1-a)*self.r_up[None][1] ]

      
    
    def getcross(self,p1,p2,p):
        return (p2[0]-p1[0])*(p[1]-p1[1])-(p[0]-p1[0])*(p2[1]-p1[1])
    def getdis(self,p1,p2,p):
        a=p2[1]-p1[1]
        b=p1[0]-p2[0]
        c=p2[0]*p1[1]-p1[0]*p2[1]
        return math.fabs(a*p[0]+b*p[1]+c)/pow(a*a+b*b,0.5)
    def isIn(self,p):
        '''
        p1=self.l_up.to_numpy()
        p2=self.l_down.to_numpy()
        p3=self.r_down.to_numpy()
        p4=self.r_up.to_numpy()
        '''
        
        p1=np.array([self.l_up[None][0],self.l_up[None][1]])
        p2=np.array([self.l_down[None][0],self.l_down[None][1]])
        p3=np.array([self.r_down[None][0],self.r_down[None][1]])
        p4=np.array([self.r_up[None][0],self.r_up[None][1]])

        ispointIn=self.getcross(p1, p2, p)*self.getcross(p3, p4, p)>=0 and self.getcross(p2, p3, p)*self.getcross(p4, p1, p)>=0
        return ispointIn
    def get_state(self):
     
        state=self.sample_v.to_numpy()
        return state
    def get_rigid_state(self):
        II=np.array([self.center[None][0],self.center[None][1],self.angle[None]*0.1,self.vel[None][0]*0.1,self.vel[None][1]*0.1,self.omega[None]*0.1])
        return II
    @ti.kernel
    def cal_jelly_state(self):
        for i in range(10):
            self.jelly_pos[i]=0.0
            self.jelly_vel[i]=0.0
        for i in range(self.n_particles):
            if i>=self.fluid_size:
                idx=(i-self.fluid_size)//self.jelly_size
                if idx>=5:
                    continue
                self.jelly_pos[idx*2]+=self.x[i][0]/self.jelly_size
                self.jelly_pos[idx*2+1]+=self.x[i][1]/self.jelly_size
                self.jelly_vel[idx*2]+=self.v[i][0]/self.jelly_size*0.1
                self.jelly_vel[idx*2+1]+=self.v[i][1]/self.jelly_size*0.1
        #self.jelly_pos[20]=0.025

    def get_jelly_state(self):
        '''
        self.cal_jelly_state()
        poss=np.zeros(21)
        vels=np.zeros(20)
        for i in range(20):
            poss[i]=self.jelly_pos[i]
            vels[i]=self.jelly_vel[i]
        poss[20]=0.025
        return np.append(poss,vels)
        '''
        self.cal_jelly_state()
        poss=np.zeros(10)
        #phos=np.zeros(10)
        vels=np.zeros(10)
        Rs=np.zeros(5)
        for i in range(5):
            k=i#self.o[i]
            poss[2*i]=self.jelly_pos[2*k]
            poss[2*i+1]=self.jelly_pos[2*k+1]
            vels[2*i]=self.jelly_vel[2*k]
            vels[2*i+1]=self.jelly_vel[2*k+1]
        for i in range(5):
            Rs[i]=self.R[i]
        #poss[20]=0.025
        #print(Rs)
        a=np.append(poss,vels)
        #return np.append(a,Rs)
        return a
    def get_reward(self):#0.01 0.25
        goal=[0.4,0.8]
        bad=0
        good=0
        reward=0
        for i in range(5):
            pos=np.array([self.jelly_pos[i*2],self.jelly_pos[2*i+1]])
            if self.isIn(pos):
                if self.R[i]==1:
                    bad=bad+1
                else:
                    good=good+1
        self.maxx[None]=good#max(self.maxx[None],good)
        dist=np.sqrt((self.center[None][0]-goal[0])*(self.center[None][0]-goal[0])+(self.center[None][1]-goal[1])*(self.center[None][1]-goal[1]))
        alpha=pow(max(0.0,good-0.0),1.5)
        v=self.vel[None][0]*self.vel[None][0]+self.vel[None][1]*self.vel[None][1]
        angle=abs(self.angle[None])
        
        reward=2*alpha*(math.exp(-dist))
        
        if self.center[None][0]<0.6 and self.center[None][1]>0.5 and good>=2:
            reward+=0.25*(alpha)*math.exp(-0.5*v)+0.25*(alpha)*math.exp(-1.0*angle)
        
        return reward
    
        
