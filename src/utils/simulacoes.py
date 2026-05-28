import numpy as np
from scipy.integrate import solve_ivp

from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# parametros
from config.config import values

########################################################################
## PID
########################################################################

class PID:
    def __init__(self, Kp, Ki, Kd, u0=0.0, u_min=None, u_max=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.u0 = u0
        self.u_min = u_min
        self.u_max = u_max

        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error, dt):
        """
        Calcula a ação de controle.
        """
        self.integral += error * dt

        derivative = (error - self.previous_error) / dt

        u = (
            self.u0 # bias
            + self.Kp * error
            + self.Ki * self.integral
            + self.Kd * derivative
        )

        if self.u_min is not None and self.u_max is not None:
            u = np.clip(u, self.u_min, self.u_max)

        self.previous_error = error

        return u


########################################################################
## MODELO NÃO LINEAR
########################################################################

def mu(S, params):
    # Tx. específica de crescimento
    mu = (params["Mi_m"]*S)/(params["Ks"] + S)
    return mu


def bioreactor_model(t, states, D, params):
    # D = tx. de diluição calculada pelo controlador

    X, S = states

    # eqs.
    dXdt = mu(S, params) * X - D * X
    dSdt = D * (params["Sf"] - S) - (mu(S, params) * X)/params["Y"]

    return [dXdt, dSdt]


########################################################################
## SIMULAÇÃO PID
########################################################################

def simular_PID(Kc, Ti, Td=0, Xsp=0.3066, D0=0.35, states=np.array([0.3066, 0.2333]), pert=False, t_pert: int = None, Sf_pert: int = None, tempo_final=100, dt=0.01, u_min=0, u_max=1, show: bool = True):
    X_history = []
    S_history = []
    D_history = []
    error_history = []
    values["Sf"] = 1.0 # recuperar estado.... talvez nem der mais problmea

    tempo = np.arange(0, tempo_final + dt, dt)

    pid = PID(
        Kp=Kc,
        Ki=Kc/Ti,
        Kd=Kc*Td,
        u0=D0,
        u_min=u_min,
        u_max=u_max
    )

    for t in tempo:
        X, S = states

        error = Xsp - X

        D = pid.update(error, dt)

        # pertubação
        if pert is True and t_pert is None:
            print("defina tempo em que a pertubação ocorre, pertubado")
            return
        elif pert is True and t > t_pert:
            values["Sf"] = Sf_pert
            pass

        sol = solve_ivp(
            fun=lambda tau, y: bioreactor_model(tau, y, D, values),
            t_span=(t, t + dt),
            y0=states,
            method="RK45"
        )

        states = sol.y[:, -1]
        # print(f't:{t:.2f} -> D:{D:.4f} com error: {error:.4f}')

        X_history.append(states[0])
        S_history.append(states[1])
        D_history.append(D)
        error_history.append(error)

    # metricas
    e = np.array(Xsp) - np.array(X_history)

    IAE = np.trapezoid(np.abs(e), tempo)
    ISE = np.trapezoid(e**2, tempo)

    if show:
        ####### X #######
        plt.figure()
        plt.plot(tempo, X_history, label="X")
        plt.axhline(Xsp, linestyle="--", label="Xsp")
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        # plt.ylim(
        #     Xsp - 3e-5,
        #     Xsp + 3e-5
        # )
        plt.xlabel("Tempo (h)")
        plt.ylabel("Biomassa X (g/L)")
        plt.legend()
        plt.grid()
        plt.show()

        ####### erro #######
        plt.figure()
        plt.plot(tempo, error_history, label="error")
        # plt.axhline(Xsp, linestyle="--", label="Xsp")
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        # plt.ylim(
        #     Xsp - 3e-5,
        #     Xsp + 3e-5
        # )
        plt.xlabel("Tempo (h)")
        plt.ylabel("Erro X (g/L)")
        plt.legend()
        plt.grid()
        plt.show()

        ####### Sf #######

        plt.figure()
        plt.plot(tempo, S_history, label="S")
        plt.xlabel("Tempo (h)")
        plt.ylabel("Substrato S (g/L)")
        plt.legend()
        plt.grid()
        plt.show()

        ####### D #######

        plt.figure()
        plt.plot(tempo, D_history, label="D")

        # desativa a notação científica com offset
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)

        plt.xlabel("Tempo (h)")
        plt.ylabel("Taxa de diluição D (h⁻¹)")
        plt.legend()
        plt.grid()
        plt.show()
    
    return tempo, X_history, S_history, D_history, IAE, ISE,error_history


########################################################################
## SIMULAÇÃO STEP TEST
########################################################################

def simular_step_test(
    D0=0.35,
    step_percent=0.05,
    step_time=10,
    states=np.array([0.3066, 0.2333]),
    tempo_final=50,
    dt=0.01
):

    X_history = []
    S_history = []
    D_history = []

    tempo = np.arange(0, tempo_final + dt, dt)

    for t in tempo:

        if t < step_time:
            D = D0
        else:
            D = D0 * (1 + step_percent)

        sol = solve_ivp(
            fun=lambda tau, y: bioreactor_model(
                tau, y, D, values
            ),
            t_span=(t, t + dt),
            y0=states,
            method="RK45"
        )

        states = sol.y[:, -1]

        X_history.append(states[0])
        S_history.append(states[1])
        D_history.append(D)

    return tempo, X_history, S_history, D_history


#################################
## SIMULAÇÃO RELAY
####################################
def simular_relay_autotuning(
    Xsp=0.3066,
    D0=0.35,
    d=0.01,                 # amplitude do relé
    dead_zone=0.0005,       # zona morta -> evitar chaveamento excessivo
    states=np.array([0.3066, 0.2333]),
    tempo_final=100,
    dt=0.01,
    u_min=0,
    u_max=1
):
    X_history = []
    S_history = []
    D_history = []
    error_history = []

    tempo = np.arange(0, tempo_final + dt, dt)

    # relé em torno do ponto nominal D0.
    # começa com um empurrão inicial, já que X inicia exatamente em Xsp.
    D = D0 + d

    for t in tempo:
        X, S = states
        error = Xsp - X

        # liga/desliga do relé: Se erro > dead_zone, então X < Xsp -> aumento X.
        # aumentar X exige reduzir D -> D0 - d.
        if error > dead_zone:
            D = D0 - d

        # Se erro < -dead_zone, então X > Xsp -> reduzo X.
        # reduzir X exige aumentar D -> D0 + d.
        elif error < -dead_zone:
            D = D0 + d

        # na zona morta -> mantém o último valor de D.

        # Saturação da manipulada
        D = np.clip(D, u_min, u_max)

        sol = solve_ivp(
            fun=lambda tau, y: bioreactor_model(tau, y, D, values),
            t_span=(t, t + dt),
            y0=states,
            method="RK45"
        )

        states = sol.y[:, -1]

        X_history.append(states[0])
        S_history.append(states[1])
        D_history.append(D)
        error_history.append(error)

    return tempo, X_history, S_history, D_history


def estimar_parametros_relay(t, X, d, tempo_descartar=20):
    t = np.array(t)
    X = np.array(X)

    # descarta o começo -> pegar só a oscilação em regime
    mask = t > tempo_descartar
    t_ss = t[mask]
    X_ss = X[mask]

    # máximos e mínimos
    peaks, _ = find_peaks(X_ss)
    troughs, _ = find_peaks(-X_ss)

    X_max = X_ss[peaks]
    X_min = X_ss[troughs]
    t_peaks = t_ss[peaks]

    # amplitude média da saída
    a = (np.mean(X_max) - np.mean(X_min)) / 2

    # período médio da saída
    Pu = np.mean(np.diff(t_peaks))

    # ganho último aproximado pelo relay
    Kcu = (4*d) / (np.pi*a)

    # Ziegler-Nichols para PID
    Kc = 0.45*Kcu
    Ti = Pu/1.2
    Td = 0

    return a, Pu, Kcu, Kc, Ti, Td