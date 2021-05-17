import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import izip

colmap_bluegreen = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["slategrey", "steelblue", "darkcyan", "lightseagreen", "seagreen", "darkgreen"])
colmap_blue = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["lightsteelblue", "cornflowerblue", "steelblue", "mediumblue", "navy"])
colmap_green = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["yellowgreen", "darkseagreen", "mediumseagreen", "seagreen", "darkgreen"])

fig, ax = plt.subplots()

def plot_mean(fname_pref, col, lab, eps=500, num=5):
    mean = np.zeros((eps-1))
    for n in range(num):
        filename = fname_pref+str(n)+".txt"
        f = open(filename, "r")
        epochs, verr, tloss = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=2)
        f.close()
        verr = 100 - verr
        mean += verr
        ax.plot(epochs, verr, color=col, alpha=0.3)
    mean /= 5
    ax.plot(epochs, mean, label=lab, color=col)


plot_mean("tanh-nobn/verr_loss_tanh16_srng7_", "black", "Base")

plot_mean("hardtanh-nobn/verr_loss_none16_srng7_run", "forestgreen", r"tanh$_\mathrm{hard}$")

plot_mean("pwl-nobn/verr_loss_pwl16_srng7_", "darkred", r"tanh$_\mathrm{PWL}$")

# for nt, col in izip([4, 8, 16, 24, 32], ["palegreen", "yellowgreen", "limegreen", "forestgreen", "darkgreen"]):
#     plot_mean("lutnew-nobn/verr_loss_lut"+str(nt)+"_srng7_", col, "LUT"+str(nt))
# nt = 16
plot_mean("lutnew-nobn/verr_loss_lut16_srng7_", "steelblue", r"tanh$_\mathrm{LUT16}$")

colors_blue = [colmap_blue(x) for x in np.linspace(0, 0.6, 4)]
colors_green = [colmap_green(x) for x in np.linspace(0, 1, 4)]

# for nt, col in izip([8, 16, 24, 32], ["mediumturquoise", "darkturquoise", "cadetblue", "darkcyan"]):
# for nt, col_b, col_g in izip([8, 16, 24, 32], colors_blue, colors_green):
    # plot_mean("lutold-nobn/verr_loss_lut"+str(nt)+"_srng7_", col_b, "LUT"+str(nt))
    # plot_mean("lutnew-nobn/verr_loss_lut"+str(nt)+"_srng7_", col_b, "LUT"+str(nt))



handles, labels = plt.gca().get_legend_handles_labels()
newLabels, newHandles = [], []
for handle, label in zip(handles, labels):
    if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
legend = ax.legend(newHandles, newLabels, 
    # loc='best', bbox_to_anchor=(0.65, 0.39), 
    loc='lower right', 
    shadow=True, fontsize='x-large')

ax.set(xlabel='Epochs', ylabel='Accuracy')
# legend = ax.legend(
#     loc='lower right', 
#     # loc='best', 
#     shadow=True, 
#     fontsize='x-large')
fig.savefig("tanh_comp.png", bbox_extra_artists=(legend,), bbox_inches='tight')
plt.show()







# mean = np.zeros((499))
# for n in range(1, 6):
#     filename = "verr_loss_tanh16_srng7_run"+str(n)+".txt"
#     f = open(filename, "r")
#     epochs, verr, tloss = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=2)
#     f.close()
#     mean += tloss
#     ax.plot(epochs, tloss, label="base", color="dimgray")
# mean /= 5
# ax.plot(epochs, mean, label="base(mean)", linestyle='--', color="black")

# for nt in [32, ]:#16, 24, 32]:
#     mean = np.zeros((499))
#     for n in range(0, 5):
#         filename = "verr_loss_lut"+str(nt)+"_srng7_"+str(n)+".txt"
#         f = open(filename, "r")
#         epochs, verr, tloss = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=2)
#         f.close()
#         mean += tloss
#         ax.plot(epochs, tloss, label="LUT"+str(nt)+"-tanh", color="lightsteelblue")
#     mean /= 5
#     ax.plot(epochs, mean, label="LUT"+str(nt)+"-tanh(mean)", linestyle='--', color="steelblue")

# f = open("verr_loss_lut32_srng7_0_new.txt", "r")
# epochs, verr, tloss = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=2)
# f.close()
# ax.plot(epochs, tloss, label="LUT32-new", color="darkorange")

# f = open("verr_loss_lut64_srng7_0_new.txt", "r")
# epochs, verr, tloss = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=2)
# f.close()
# ax.plot(epochs, tloss, label="LUT64-new", color="green")

# f = open("verr_loss_lut32_srng7_0_bn.txt", "r")
# epochs, verr, tloss = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=2)
# f.close()
# ax.plot(epochs, tloss, label="LUT32-BN", color="purple")

# f = open("verr_loss_lut256_srng7_0.txt", "r")
# epochs, verr, tloss = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True, skiprows=2)
# f.close()
# ax.plot(epochs, tloss, label="LUT256", color="fuchsia")


# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# legend = ax.legend(newHandles, newLabels, loc='upper right', shadow=True, fontsize='x-large')
# ax.set(xlabel='Epochs', ylabel='Loss')
# fig.savefig("tanh_lut8_loss.png", bbox_extra_artists=(legend,), bbox_inches='tight')
# plt.show()