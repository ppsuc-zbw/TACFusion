from setuptools import setup 
from torch.utils.cpp_extension import BuildExtension, CUDAExtension 
#####编译可形变卷积底层cuda算子####
setup( 
      name='deform_conv_ext', 
       
      ext_modules=[ CUDAExtension(define_macros=[('WITH_CUDA', None)],name= 'deform_conv_ext', 
                                  sources=["deform_conv_ext.cpp", "deform_conv_cuda.cpp",'deform_conv_cuda_kernel.cu'], ) ], 
      cmdclass={ "build_ext": BuildExtension }
      )