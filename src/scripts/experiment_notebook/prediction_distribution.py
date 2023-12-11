
def plot_distribution(ade_distribution):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.set_title("Distribution of ADE")

    ax.set_xlabel("ADE")

    ax.set_ylabel("Frequency")

    ax.hist(ade_distribution, bins=100)

    plt.show()
