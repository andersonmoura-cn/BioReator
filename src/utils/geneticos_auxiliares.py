import numpy as np
import matplotlib.pyplot as plt

def ajuste_string(individuo: list = None, tipo_param: str = "Kp"):
    if tipo_param == "Ti":
        individuo = ['+'] + individuo
        
    digitos_completar = 6 - len(individuo)
    while digitos_completar > 0:
        individuo.append('0')
        digitos_completar-=1
    return individuo

def gerar_mask_cruzamento():
    mask = np.array([
                    0,
                    np.random.randint(0, 2),
                    0,
                    np.random.randint(0, 2),
                    np.random.randint(0, 2),
                    np.random.randint(0, 2)
                ])
    
    return mask

def gerar_populacao_inicial(tamanho_povo):
    # g init
    gen = np.column_stack([
        -np.round(np.random.uniform(0, 5, tamanho_povo), 3),  # sempre negativo
        np.round(np.random.uniform(0, 10, tamanho_povo), 3),
    ])

    for tuning_parameter_indice in range(len(gen)):
        # Checa casos em que Ti = 0
        while gen[tuning_parameter_indice, 1] == 0:
            # re-gera aleatoriamente 
            gen[tuning_parameter_indice, 1] = np.round(np.random.uniform(0, 10), 3)
    
    return gen

def ordenar_populacao(povo):
    # Ordena pelo erro (usando valor absoluto)
    # Agora apenas os fortes têm erros correspondentes
    indices_ordenados = np.argsort(np.abs(povo["erros"]))

    povo_ordenado = {
        "candidatos": [povo["candidatos"][i] for i in indices_ordenados],
        "erros": [povo["erros"][i] for i in indices_ordenados]
    }

    return povo_ordenado

def grafico_convergencia(historico_erro):
    tempo = np.arange(len(historico_erro))

    plt.figure(figsize=(8,4))

    plt.plot(
        tempo,
        historico_erro,
        color='lightblue',
        linewidth=1.5
    )

    plt.scatter(
        tempo,
        historico_erro,
        color='blue',
        s=20
    )

    plt.xlabel("Geração")
    plt.ylabel("Erro")
    plt.grid(True, alpha=0.3)

    plt.show()

def cruzamento(fortes, pc: int = 1):
    N = len(fortes)

    filhos = np.zeros(shape=(N, 2))
    
    for i in range(0, N, 2):
        # escolha dos pais
        idx_pai, idx_mae = np.random.choice(len(fortes),size=2,replace=False)
        
        # sorteia se esse par vai copular
        r = np.random.uniform(0, 1)
        if r <= pc:
            # crossover para o Kp
            pai = fortes[idx_pai, 0]
            mae = fortes[idx_mae, 0]
                
            arr_p = np.array(ajuste_string(list(str(pai)), tipo_param="Kp"))
            arr_m = np.array(ajuste_string(list(str(mae)), tipo_param="Kp"))

            # crossover para o Ti
            pai = fortes[idx_pai, 1]
            mae = fortes[idx_mae, 1]
                
            arr_p = np.array(ajuste_string(list(str(pai)), tipo_param="Ti"))
            arr_m = np.array(ajuste_string(list(str(mae)), tipo_param="Ti"))

            # mascaras
            mask_Kp = gerar_mask_cruzamento()
            mask_Ti = gerar_mask_cruzamento()
             
            # gera filho 1
            filhos[i, 0] = float(''.join(np.where(mask_Kp, arr_m, arr_p)))
            filhos[i, 1] = float(''.join(np.where(mask_Ti, arr_m, arr_p)))

            if i != N-1:
                # gera filho 2
                filhos[i + 1, 1] = float(''.join(np.where(mask_Ti, arr_p, arr_m)))
                filhos[i + 1, 0] = float(''.join(np.where(mask_Kp, arr_p, arr_m)))
        else:
            # caso não possam acasalar, os pais são sujeitos direto a mutação
            filhos[i] = fortes[idx_mae]
            filhos[i + 1] = fortes[idx_pai]

    return filhos

def mutacao_gene(cromossomo, pm):
    for gene in range(len(cromossomo)):
            if (gene != 0 and gene != 2):
                r = np.random.uniform(0, 1)
                if r < pm:
                    cromossomo[gene] = str(np.random.randint(0, 10))
    
    return cromossomo
    
def mutacao(populacao, pm: float = 0.1):
    for i in range(len(populacao)):
        Kp = populacao[i, 0]
        Ti = populacao[i, 1]
        
        cromossomo_Kp = np.array(ajuste_string(list(str(Kp)), tipo_param="Kp"))
        cromossomo_Ti = np.array(ajuste_string(list(str(Ti)), tipo_param="Ti"))
        
        cromossomo_Kp = mutacao_gene(cromossomo_Kp, pm)
        cromossomo_Ti = mutacao_gene(cromossomo_Ti, pm)
        
        populacao[i, 0] = float(''.join(cromossomo_Kp))
        populacao[i, 1] = float(''.join(cromossomo_Ti))
        
    return populacao
