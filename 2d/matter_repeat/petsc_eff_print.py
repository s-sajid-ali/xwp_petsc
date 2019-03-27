
# coding: utf-8

# In[1]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
from multislice import prop,prop_utils
from skimage.restoration import unwrap_phase


# In[2]:


def get_first_lobe(foc_spot,loc,width,plot=0):
    cx,cy = np.round(np.where(foc_spot==np.max(foc_spot)))
    nr,nc = np.shape(foc_spot)
    r = np.arange(nr)-cx 
    c = np.arange(nc)-cy 
    [R,C] = np.meshgrid(r,c)
    index = np.round(np.sqrt(R**2+C**2))+1 
    primary_lobe = 0
    temp = np.max(foc_spot)

    for i in np.arange(-width,width):
        j = i + loc
        primary_lobe+=np.sum(foc_spot[np.where(index==j)])
        
    if plot==1 :
        plt.imshow(np.log10(foc_spot),alpha=0.5,cmap='jet')
        plt.colorbar()
        for i in np.arange(-width,width):
            j = i + loc
            R1,C1 = np.where(index==j)
            plt.scatter(C1,R1,s=5,alpha = 0.25)
        plt.show()
    return primary_lobe


# In[3]:


f = h5py.File('solution.h5', 'r')
dset = f['sol_vec']
dim_x,dim_y = int(np.sqrt(dset.shape[0])),int(np.sqrt(dset.shape[0]))
slices = 1


# In[4]:


data = np.zeros(dset.shape)
dset.read_direct(data, source_sel=np.s_[:,:], dest_sel=np.s_[:,:])
dset.shape
f.close()


# In[5]:


wave_exit = (data[:,0]+1j*data[:,1]).reshape(dim_x,dim_y)
wave_far_field = np.fft.fftshift(np.fft.fft2(wave_exit,norm="ortho"))


# In[6]:


N = 2500
n = 250
foc_spot = np.abs(wave_far_field**2)[N-n:N+n,N-n:N+n]
first_lobe = get_first_lobe(foc_spot,124,40)
eff = 100 * (first_lobe/np.sum(np.abs(wave_far_field**2)))*(5e-6)**2/(np.pi*(100*20e-9)**2)


# In[7]:


print(eff)


# In[ ]:




