
import matplotlib.pyplot as plt


def plot_grid(mat, row=4, col=5,):
    """ mat : (b, w, h, c)
    """
    plt.figure(figsize = (row * 1.8, col * 1.8), dpi=250)
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            plt.subplot(row, col, idx + 1)
            plt.imshow(mat[idx])
            plt.axis('off')
    plt.show()
