import numpy as np
from scipy.integrate import solve_ivp

from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# metricas
from utils.metrics import metricsControl
# parametros
from config.config import values

from utils.geneticos_auxiliares import mutacao, cruzamento, ordenar_populacao

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
    X_history = [states[0]]
    S_history = [states[1]]
    D_history = [D0]
    error_history = []
    values["Sf"] = 1.0 # recuperar estado.... talvez nem der mais problmea

    tempo = np.arange(dt, tempo_final + dt, dt)

    Ki = 0
    if Ti != 0:
        Ki = Kc/Ti
    
    pid = PID(
        Kp=Kc,
        Ki=Ki,
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

    tempo = np.concatenate(([0], tempo))
    error_history.append(Xsp - states[0])
    
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
    pert=False, t_pert: int = None, Sf_pert: int = None,
    show=True
):
    X_history = []
    S_history = []
    D_history = []
    error_history = []
    de_history = []
    deltaD_history = []
    values["Sf"] = 1.0 # recuperar estado.... talvez nem der mais problmea


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

        # pertubação
        if pert is True and t_pert is None:
            print("defina tempo em que a pertubação ocorre, pertubado")
            return
        elif pert is True and t > t_pert:
            values["Sf"] = Sf_pert
            pass


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

        ####### delta erro #######
        plt.figure()
        plt.plot(tempo, de_history, label="erro")
        plt.axhline(0, linestyle="--")
        plt.xlabel("Tempo (h)")
        plt.ylabel(r"$\Delta$ Erro")
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
    Kcu= -0.5,
    passo= 0.1,
    Xsp=0.3066,
    D0=0.35,
    states=np.array([0.3066, 0.2333]),
    tempo_final=100,
    dt=0.01,
    u_min=0,
    u_max=1,
    iter_max=100
):

    # Pequena mudança no setpoint de 3%
    Xsp_novo = Xsp * 1.03
    
    Pu = 0
    
    while iter_max > 0:
        t, X, S, D, _, _, erro = simular_PID(Kc=Kcu, Ti=np.inf, Xsp=Xsp_novo, D0=D0, states=states, tempo_final=tempo_final, show= False)
        
        t = np.array(t)
        X = np.array(X)
         
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
        
        Kcu -= passo
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
    
def selecao(erro_individuo_lim, Xsp, D0, states, gen, tf, dt, pert, t_pert, Sf_pert):
    erro_min = np.inf
    fortes = []
    fracos = []
    erros_fortes = []
    erros_fracos = []

    for individuo in gen:
        erro_individuo = np.inf
        # só pode ser selecionado como forte se Ti != 0
        if individuo[1] != 0:
            t1, X1, S1, D1, IAE, ISE, erro = simular_PID(Kc=individuo[0], Ti=individuo[1], Xsp=Xsp, D0=D0, states=states, tempo_final=tf, dt=dt, show=False, pert=pert, t_pert=t_pert, Sf_pert=Sf_pert)
            erro_individuo = IAE

        if erro_individuo <= erro_individuo_lim: 
            fortes.append(individuo)
            erros_fortes.append(erro_individuo)
        else:
            fracos.append(individuo)
            erros_fracos.append(erro_individuo)
        
        # menor erro dentro dessa geracao
        if np.abs(erro_min) > np.abs(erro_individuo):
            erro_min = erro_individuo
                
    return fortes, fracos, erros_fortes, erros_fracos, erro_min

def simular_genetico(iter_max, erro_desejado,erro_individuo_lim, D0, states, gen, Xsp, tf, dt, pert=False, t_pert: int = None, Sf_pert: int = None, flag_fracos: bool = True):
    df = None
    historico_erro = []
    while iter_max > 0:
        # seleção da geração
        fortes, fracos, erros_fortes, erros_fracos, erro_min = selecao(erro_individuo_lim= erro_individuo_lim, Xsp=Xsp, D0=D0, states=states, gen= gen, tf=tf, dt=dt, pert=pert, t_pert=t_pert, Sf_pert= Sf_pert)
             
        # salva erro minimo de cada geração
        historico_erro.append(erro_min)
        
        dict_populacao = {
                "candidatos": fortes,
                "erros": erros_fortes
            }
        
        # o menor custo da geração foi top, para aqui
        if abs(erro_min) < erro_desejado:
            print("ola")
            break
        
        if iter_max > 1:
            filhos = np.empty((0, 2))
            if len(fortes) >= 2:
                filhos = cruzamento(np.array(fortes))
                filhos = mutacao(filhos) 
            elif len(fortes) == 1:
                filhos = fortes

            # nova geracao
            gen = np.concatenate([mutacao(np.array(fracos), individuo= "Fraco"), filhos]) if len(fracos) > 0 else filhos

        if iter_max == 1 and len(fortes) == 0:
            # Se na ultima geração só houver fracos, o escolhido será o melhor entre eles
            dict_populacao = {
                "candidatos": fracos,
                "erros": erros_fracos
            }
            
        iter_max-=1
        
    povo = ordenar_populacao(dict_populacao)
    Kp = povo["candidatos"][0][0]
    Ti = povo["candidatos"][0][1]
    
    return Kp, Ti, historico_erro
