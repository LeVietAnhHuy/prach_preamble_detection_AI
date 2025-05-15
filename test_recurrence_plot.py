from pyts.image import RecurrencePlot
import numpy as np
import matplotlib.pyplot as plt
import os

image_dir = 'image'

data = np.load('generated_dataset/corr_dataset/corr_15dB.npy')
pre_idx = 160000
x_corr = data[pre_idx, :1024]
x_recurrence_plot = x_corr.reshape(1, -1)
print(data.shape)
print(data)

transformer = RecurrencePlot()
X_new = transformer.transform(x_recurrence_plot)

image_name_recurrence_plot = 'recurrence_plot_' + str(data[pre_idx, -1]) + '.png'
image_name_corr = 'corr_' + str(data[pre_idx, -1]) + '.png'

full_image_path_recurrence_plot = os.path.join(image_dir, image_name_recurrence_plot)
full_image_path_corr = os.path.join(image_dir, image_name_corr)

plt.imshow(X_new[0], cmap='binary', origin='lower')
plt.title("Recurrence Plot")
plt.colorbar()
plt.savefig(full_image_path_recurrence_plot, dpi=300)
print(f"Plot saved as '{full_image_path_recurrence_plot}'")
plt.close()


plt.plot(x_corr)
plt.savefig(full_image_path_corr, dpi=300)
print(f"Plot saved as '{full_image_path_corr}'")
plt.close()




# from pyts.datasets import load_gunpoint
# from pyts.image import RecurrencePlot
# X, _, _, _ = load_gunpoint(return_X_y=True)
# transformer = RecurrencePlot()
# X_new = transformer.transform(X)
#
# print("")

# from pyunicorn.timeseries import RecurrencePlot
# import numpy as np
#
# x = np.sin(np.linspace(0, 2 * np.pi, 100))
#
# rp = RecurrencePlot(x, dimension=1, time_delay=1, threshold=0.1)
# rp.plot()

# from pyts.image import RecurrencePlot
# import matplotlib.pyplot as plt
# # import matplotlib
# # matplotlib.use('TkAgg')
# import numpy as np
#
# # Example time series
# x = np.sin(np.linspace(0, 4 * np.pi, 128))
# x = x.reshape(1, -1)
# # Recurrence plot transformation
# rp = RecurrencePlot(threshold='point', percentage=10)
# X_rp = rp.fit_transform(x)
#
# # Plot
#
#
# plt.imshow(X_rp[0], cmap='binary', origin='lower')
# plt.title("Recurrence Plot")
# plt.colorbar()
# plt.savefig('recurrence_plot.png', dpi=300)
# print("Plot saved as 'recurrence_plot.png'")
# plt.show()


