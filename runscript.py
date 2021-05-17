#!/usr/bin/python

import numpy as np
import os
# import lfsr_mnist_GXNOR as gxnor_lfsr
# import mnist_GXNOR_fixedrand as gxnor_rand
# import tanh_mnist_GXNOR as gxnor_tanh
# import mnist_GXNOR as gxnor_base
# import fc_mnist_GXNOR as gxnor_fc

num_epochs = 500
discrete = True
batch_size = 10000

# gxnor_tanh.main(num_epochs, discrete, batch_size, 7, 16, 0.01, 0.01, "tanh", "srng", 125)
# gxnor_tanh.main(num_epochs, discrete, batch_size, 7, 16, 0.01, 0.01, "tanh", "srng", 875)
# gxnor_tanh.main(num_epochs, discrete, batch_size, 7, 16, 0.01, 0.01, "tanh", "srng", 34257)
# for i in range(5):
#     os.system("python tanh_mnist_GXNOR.py" + 
#         " -e " + str(num_epochs) + " -d " + " -b " + str(batch_size) + 
#         " --nl 7" + " --nt 16" + " --lrs 0.01" + " --lrf 0.01" + 
#         " -t pwl" + " -r srng" + " -s " + str(i))

# for nt in reversed([4, 8, 16, 24]):
# # nt = 32
#     for i in range(5):
#         os.system("python tanh_mnist_GXNOR.py" + 
#             " -e " + str(num_epochs) + " -d " + " -b " + str(batch_size) + 
#             " --nl 7" + " --nt " + str(nt) + " --lrs 0.01" + " --lrf 0.01" + 
#             " -t lut" + " -r srng" + " -s " + str(i))
nl = 7
nt = 0


# # tanh + lfsr7
# for i in range(5):
#     os.system("python tanh_mnist_GXNOR.py" + 
#         " -e " + str(num_epochs) + " -d " + " -b " + str(batch_size) + 
#         " --nl " + str(nl) + " --nt " + str(nt) + " --lrs 0.01" + " --lrf 0.01" + 
#         " -t tanh" + " -r lfsr" + " -s " + str(i))

# # lut16 + lfsr7
# nt = 16
# for i in range(5):
#     os.system("python tanh_mnist_GXNOR.py" + 
#         " -e " + str(num_epochs) + " -d " + " -b " + str(batch_size) + 
#         " --nl " + str(nl) + " --nt " + str(nt) + " --lrs 0.01" + " --lrf 0.01" + 
#         " -t lut" + " -r lfsr" + " -s " + str(i))
# pwl + lfsr
# nt = 0
# for nl in [3, 5, 15, 32]:
#     for i in range(5):
#         os.system("python tanh_mnist_GXNOR.py" + 
#             " -e " + str(num_epochs) + " -d " + " -b " + str(batch_size) + 
#             " --nl " + str(nl) + " --nt " + str(nt) + " --lrs 0.01" + " --lrf 0.01" + 
#             " -t pwl" + " -r lfsr" + " -s " + str(i))

# # lut16 + lfsr
nt = 16
for nl in [3, 5]:
    for i in range(5):
        os.system("python tanh_mnist_GXNOR.py" + 
            " -e " + str(num_epochs) + " -d " + " -b " + str(batch_size) + 
            " --nl " + str(nl) + " --nt " + str(nt) + " --lrs 0.01" + " --lrf 0.01" + 
            " -t lut" + " -r lfsr" + " -s " + str(i))

# # bn + pwl + lfsr7
# for i in range(5):
#     os.system("python tanh_mnist_GXNOR.py" + " -e " + str(num_epochs) + " -d " + 
#     " -b " + str(batch_size) + " --nl " + str(7) + " --nt " + str(nt) + 
#     " --lrs 0.1" + " --lrf 0.0001" + " -t pwl" + " -r lfsr" + " -s " + str(i) + " --bn")
# # bn + lut16 + lfsr7
# for i in range(5):
#     os.system("python tanh_mnist_GXNOR.py" + " -e " + str(num_epochs) + " -d " + 
#     " -b " + str(batch_size) + " --nl " + str(7) + " --nt " + str(16) + 
#     " --lrs 0.1" + " --lrf 0.0001" + " -t tanh" + " -r lfsr" + " -s " + str(i) + " --bn")
# # bn + hard + lfsr7
# for i in range(5):
#     os.system("python tanh_mnist_GXNOR.py" + " -e " + str(num_epochs) + " -d " + 
#     " -b " + str(batch_size) + " --nl " + str(7) + " --nt " + str(nt) + 
#     " --lrs 0.1" + " --lrf 0.0001" + " -t none" + " -r lfsr" + " -s " + str(i) + " --bn")

# os.system("python tanh_mnist_GXNOR.py" + 
#             " -e " + str(num_epochs) + " -d " + " -b " + str(batch_size) + 
#             " --nl 7" + " --nt 32 --lrs 0.01" + " --lrf 0.01" + 
#             " -t lut" + " -r srng" + " -s 0")

# gxnor_tanh.main(2000, discrete, batch_size, 7, 16, 0.01, 0.01)
# gxnor_base.main()

# lfsr
# for n in range(3, 33):
#    print("Starting for LFSR with "+str(n)+" bits...")
#    gxnor_lfsr.main(num_epochs, discrete, batch_size, n)
# gxnor_lfsr.main(num_epochs, discrete, 100, 10)


# fixed_rand
# for nbits in range(3, 9):
#     for n in range(1, 14, 2):
#         gxnor_rand.main(num_epochs, discrete, batch_size, nbits, n)
# gxnor_rand.main(num_epochs, discrete, batch_size, 7, 1)
    

# approx_tanh   (old with LR 0.1 to 0.0000001)
# for n in range(8, 17, 2):
#    gxnor_tanh.main(num_epochs, discrete, batch_size, 1, n)
# for n in range(20, 33, 4):
#    gxnor_tanh.main(num_epochs, discrete, batch_size, 1, n)

# gxnor_tanh.main(num_epochs, discrete, batch_size, 7, 16, 0.01, 0.01)
# gxnor_tanh.main(num_epochs, discrete, batch_size, 7, 20)
        

# base
# gxnor_base.main()


#small test learning rate
# res = np.zeros((10, 10))
# lr_starts = np.float32(np.linspace(0.0001, 0.1, 10))
# lr_ends = np.float32(np.linspace(0.01, 0.00001, 10))
# for i, lr_start in np.ndenumerate(lr_starts):
#     for j, lr_end in np.ndenumerate(lr_ends):
#         if lr_start > lr_end:
#             print("start: "+str(lr_start)+", end: "+str(lr_end))    
#             # best = gxnor_fc.main(lr_start, lr_end)
#             best = gxnor_tanh.main(num_epochs, discrete, batch_size, 7, 20, lr_start, lr_end)
#             res[i][j] = best

# print(res)