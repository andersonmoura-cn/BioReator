import numpy as np
from scipy.integrate import solve_ivp

from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# metricas
from utils.metrics import metricsControl
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
    metrics = metricsControl(Xsp, X_history, tempo)

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
    
    return tempo, X_history, S_history, D_history, metrics['IAE'], metrics['ISE'], error_history


########################################################################
## SIMULAÇÃO STEP TEST
########################################################################

def simular_step_test(
    D0=0.35,
    step_percent=0.05,
    step_time=10,
    states=np.array([0.3066, 0.2333]),
    tempo_final=50,
    dt=0.01,
    Xsp = 0.3066,
    show=True
):

    X_history = []
    S_history = []
    D_history = []
    error_history = []

    tempo = np.arange(0, tempo_final + dt, dt)

    for t in tempo:
        X, S = states
        error = Xsp - X
        
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
        error_history.append(error)

    if show:
        ####### X #######
        plt.figure()
        plt.plot(tempo, X_history, label="X")
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.xlabel("Tempo (h)")
        plt.ylabel("Biomassa X (g/L)")
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


###################################
## fuzzy
##################################

def simular_fuzzy(
    sim,
    D0=0.35,
    D_min=0.22,
    D_max=0.43,
    states=np.array([0.3066, 0.2333]),
    tempo_final=50,
    dt=0.01,
    Xsp=0.3066,
    show=True
):
    X_history = []
    S_history = []
    D_history = []
    error_history = []
    de_history = []
    deltaD_history = []

    tempo = np.arange(0, tempo_final + dt, dt)

    D = D0
    e_ant = Xsp - states[0]

    for t in tempo:
        X, S = states

        # erro atual
        e = Xsp - X

        # variação do erro
        de = (e - e_ant)/dt

        # evita sair do universo de discurso do fuzzy
        e_fuzzy = np.clip(e, -0.05, 0.05)
        de_fuzzy = np.clip(de, -0.02, 0.02)

        # entrada crisp no controlador fuzzy
        sim.input['erro'] = e_fuzzy
        sim.input['delta_erro'] = de_fuzzy

        # calcula saída fuzzy defuzzificada
        sim.compute()

        deltaD = sim.output['delta_D']

        # atualiza D
        D = D + deltaD

        # saturação D
        D = np.clip(D, D_min, D_max)

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
        error_history.append(e)
        de_history.append(de)
        deltaD_history.append(deltaD)

        e_ant = e

    # metricas
    metrics = metricsControl(Xsp, X_history, tempo)

    if show:
        ####### X #######
        plt.figure()
        plt.plot(tempo, X_history, label="X")
        plt.axhline(Xsp, linestyle="--", label="Xsp")
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.xlabel("Tempo (h)")
        plt.ylabel("Biomassa X (g/L)")
        plt.legend()
        plt.grid()
        plt.show()

        ####### S #######
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
        plt.axhline(D_min, linestyle="--", label="D_min")
        plt.axhline(D_max, linestyle="--", label="D_max")
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.xlabel("Tempo (h)")
        plt.ylabel("Taxa de diluição D (h⁻¹)")
        plt.legend()
        plt.grid()
        plt.show()

        ####### erro #######
        plt.figure()
        plt.plot(tempo, error_history, label="erro")
        plt.axhline(0, linestyle="--")
        plt.xlabel("Tempo (h)")
        plt.ylabel("Erro e = Xsp - X")
        plt.legend()
        plt.grid()
        plt.show()

        ####### delta D #######
        plt.figure()
        plt.plot(tempo, deltaD_history, label="ΔD")
        plt.axhline(0, linestyle="--")
        plt.xlabel("Tempo (h)")
        plt.ylabel("Correção fuzzy ΔD")
        plt.legend()
        plt.grid()
        plt.show()

    return (
        tempo,
        X_history,
        S_history,
        D_history,
        error_history,
        de_history,
        deltaD_history,
        metrics
    )
    
###################################
## Continuous Cycling 
##################################
def simular_CC(
    Xsp=0.3066,
    D0=0.35,
    states=np.array([0.3066, 0.2333]),
    tempo_final=100,
    dt=0.01,
    u_min=0,
    u_max=1,
    iter_max=100
):

    tempo = np.arange(0, tempo_final + dt, dt)

    # Pequena mudança no setpoint de 5%
    Xsp_novo = Xsp * 1.05
    
    Kcu = -0.5
    
    while iter_max > 0:
        t, X, S, D, _, _, erro = simular_PID(Kc=Kcu, Ti=np.inf, Xsp=Xsp_novo, D0=D0, states=states, tempo_final=tempo_final)
         
        # detecta os picos da resposta
        indices_picos, _ = find_peaks(X)
         
        tempos_picos = t[indices_picos]
        valores_picos = X[indices_picos]

        amplitudes = np.abs(valores_picos - Xsp_novo)

        n_picos = 5
        
        if len(amplitudes) >= n_picos:

            ultimas_amplitudes = amplitudes[-n_picos:]

            amp_max = np.max(ultimas_amplitudes)
            amp_min = np.min(ultimas_amplitudes)

            variacao_relativa = (amp_max - amp_min) / amp_max

            print("Últimas amplitudes:", ultimas_amplitudes)
            print(f"Variação relativa: {variacao_relativa:.4f}")

            # Critério de oscilação sustentada
            tolerancia = 0.05  # 5%

            if variacao_relativa < tolerancia:
                print("Oscilação sustentada detectada.")

                # Calcula Pu usando os últimos picos
                ultimos_tempos = tempos_picos[-n_picos:]

                periodos = np.diff(ultimos_tempos)

                Pu = np.mean(periodos)

                print(f"Pu = {Pu:.4f}")
                break
        
        Kcu -= 0.1
        iter_max -= 1

    return t, X, S, D, erro, Kcu, Pu

def estimar_parametros_CC(Kcu, Pu, controlador: str = "PI"):
    
    if controlador == "P":
        Kc = 0.5 * Kcu
        
        return Kc
    
    elif controlador == "PI":
        Kc = 0.45 * Kcu
        Ti = Pu / 1.2
        
        return Kc, Ti
    
    elif controlador == "PID":
        Kc = 0.6 * Kcu
        Ti = Pu / 2
        Td = Pu / 8
        
        return Kc, Ti, Td
    
    else:
        print("Controlador indisponível. Possíveis: P, PI e PID")
        
        return -1
