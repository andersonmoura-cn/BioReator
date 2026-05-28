import matplotlib.pyplot as plt

def plot_results(time, X_history, S_history, D_history, Xsp=None):
    # ======= X =======
    plt.figure()
    plt.plot(time, X_history, label="X")

    if Xsp is not None:
        plt.axhline(Xsp, linestyle="--", label="Xsp")

    plt.xlabel("Tempo (h)")
    plt.ylabel("Biomassa X (g/L)")
    plt.legend()
    plt.grid()
    plt.show()

    # ======= S =======
    plt.figure()
    plt.plot(time, S_history, label="S")
    plt.xlabel("Tempo (h)")
    plt.ylabel("Substrato S (g/L)")
    plt.legend()
    plt.grid()
    plt.show()

    # ======= D =======
    plt.figure()
    plt.plot(time, D_history, label="D")

    # Desativa notação científica com offset
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)

    plt.xlabel("Tempo (h)")
    plt.ylabel("Taxa de diluição D (h⁻¹)")
    plt.legend()
    plt.grid()
    plt.show()