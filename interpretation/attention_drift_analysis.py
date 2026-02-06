import matplotlib.pyplot as plt

def plot_attention(attn):
    avg = attn.squeeze().detach().numpy()
    plt.plot(avg)
    plt.xlabel("Time")
    plt.ylabel("Attention")
    plt.title("Silent Adaptation Attention Drift")
    plt.show()
