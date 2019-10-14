##################################
# optics with PyTorch
##################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import ifftshift
import fractions
# U_in
class Propagation():
    def __init__(self,
                 lamda,
                 focal_length,
                 #focus_distance,
                 #do, #物到透镜的距离，即物瞳距
                 #di, #透镜到观察屏的距离
                 wave_lengths):
        
        self.lamda = lamda
        self.focal_length = focal_length
        self.wave_lengths = wave_lengths
        self.k = 2. * np.pi / wave_lengths # k是波矢

    def Uin(self, img, pysical_size, d1):
        """
        img: 图片输入
        Lambda: wave length
        pysical_size:物面的尺寸
        d1: 物瞳距, 物到透镜的距离
        """
        [c,r] = img.shape
        x0 = np.linspace(-pysical_size/2,pysical_size/2,r)
        y0 = np.linspace(-pysical_size/2,pysical_size/2,c)
        [x0,y0] = np.meshgrid(x0,y0)

        #衍射光在透镜前表面上的尺寸
        L = r * lamda * d1 / L0

        #透镜前表面坐标
        p = np.linspace(-L/2,L/2,r)
        q = np.linspace(-L/2,L/2,c)
        [p,q] = np.meshgrid(p,q)
 



        
    

def get_spherical_wavefront_phase(resolution,
                                  physical_size,
                                  wave_lengths,
                                  source_distance):
    source_distance = tf.cast(source_distance, tf.float64)
    physical_size = tf.cast(physical_size, tf.float64)
    wave_lengths = tf.cast(wave_lengths, tf.float64)
    # pysical_size 赋值物面的尺寸
    N, M = resolution
    [x, y] = np.mgrid[-N//2:N//2,
                      -M//2:M//2].astype(np.float64)

    x = x/N * physical_size
    y = y/M * physical_size

    # Assume distance to source is approx. constant over wave
    curvature = tf.sqrt(x**2 + y**2 + source_distance**2)
    wave_nos = 2. * np.pi / wave_lengths

    phase_shifts = compl_exp_tf(wave_nos * curvature)
    phase_shifts = tf.expand_dims(tf.expand_dims(phase_shifts, 0), -1)
    return phase_shifts