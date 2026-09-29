import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Importações dos módulos do seu repositório
from modelo_microrrede_pv_ess import (
    PainelFotovoltaicoVirtual,
    BateriaVirtual,
    MicrorredeVirtual,
)
from mpc_economico_microrrede import MPC_Economico

# ----------------------------------------------------------------- #
# 1. FUNÇÃO PARA LER OS CSVs E EXPANDIR PARA 7 DIAS (168 HORAS)
# ----------------------------------------------------------------- #
def ler_e_limpar_csv(caminho):
    dados_horarios = {}
    with open(caminho, mode='r', encoding='utf-8') as ficheiro:
        leitor = csv.DictReader(ficheiro)
        for linha in leitor:
            dt = datetime.strptime(linha['Time'], '%Y-%m-%d %H:%M:%S')
            chave_hora = dt.replace(minute=0, second=0)
            valor = float(linha['Potência ativa'].replace(' kW', '').replace(',', '.'))
            
            if chave_hora not in dados_horarios:
                dados_horarios[chave_hora] = {'soma': 0.0, 'contagem': 0}
            dados_horarios[chave_hora]['soma'] += valor
            dados_horarios[chave_hora]['contagem'] += 1
            
    return {hora: dados_horarios[hora]['soma'] / dados_horarios[hora]['contagem'] for hora in sorted(dados_horarios.keys())}

def carregar_perfil_carga_csv_168h():
    caminho_al4 = "Potência Ativa-data-2026-09-17 15_31_43_al4_dia15_09_26.csv"
    caminho_ru = "Potência Ativa-data-2026-09-17 15_32_17_RU_dia15_09_26.csv"
    medias_al4 = ler_e_limpar_csv(caminho_al4)
    medias_ru = ler_e_limpar_csv(caminho_ru)
    
    carga_48h_w = []
    for hora in sorted(medias_al4.keys()):
        if hora in medias_ru:
            carga_48h_w.append((medias_al4[hora] + medias_ru[hora]) * 1000.0)
            
    # Repete o padrão de 48h para preencher 168h (7 dias)
    carga_168h_w = np.tile(carga_48h_w, int(np.ceil(168 / len(carga_48h_w))))[:168]
    return np.array(carga_168h_w)

def obter_previsao_tarifa(hora_atual, horizonte):
    hh = (hora_atual + np.arange(horizonte)) % 24
    buy = np.full(horizonte, 3.0) 
    sell = np.where((hh >= 18) & (hh <= 21), 1.98, 0.58)
    return buy, sell

def obter_previsao_ciclica(perfil, hora_atual, horizonte):
    indices = (hora_atual + np.arange(horizonte)) % len(perfil)
    return perfil[indices]

# ----------------------------------------------------------------- #
# 2. SIMULADOR DOS CENÁRIOS DO ARTIGO (C0, C1, C2)
# ----------------------------------------------------------------- #
def run_case(caso, horas, irr_pred, irr_real, carga_pred, carga_real, buy_real, sell_real):
    HORIZONTE_PREDICAO = 24
    HORIZONTE_CONTROLE = 12
    PESO_LAMBDA = 5e-9

    # PARÂMETROS DA PLANTA REAL (Baseado no modelo C&I de 4 Horas)
    # E. & J. Gallo Winery Energy Storage
    PLANTA_PV = 2000e3   # 2.0 MW de PV (garante excedente diurno maciço)
    PLANTA_CAP = 4000e3  # 4.0 MWh de Bateria (banco de 4 horas)
    PLANTA_PCS = 1000e3  # Inversor da bateria de 1.0 MW (1000 kW)
    PLANTA_EFF = 0.90
    
    painel = PainelFotovoltaicoVirtual(potencia_nominal_w=PLANTA_PV)
    bateria = BateriaVirtual(
        capacidade_maxima_wh=PLANTA_CAP, eficiencia_carga=PLANTA_EFF, eficiencia_descarga=PLANTA_EFF,
        potencia_maxima_carga_w=PLANTA_PCS, potencia_maxima_descarga_w=PLANTA_PCS,
        soc_minimo=0.1, soc_maximo=0.9, soc_inicial=0.48, passo_tempo_segundos=3600.0,
    )
    microrrede = MicrorredeVirtual(painel, bateria, passo_tempo_segundos=3600.0)

    # PARÂMETROS DO MPC
    if caso == "C1":
        # C1 (Ideal): Parâmetros cravados com a realidade
        params_mpc = dict(
            capacidade_maxima_bateria_wh=PLANTA_CAP, eficiencia_carga=PLANTA_EFF, eficiencia_descarga=PLANTA_EFF,
            potencia_nominal_pv_w=PLANTA_PV, potencia_bateria_min_w=-PLANTA_PCS, potencia_bateria_max_w=PLANTA_PCS
        )
    else:
        # C2 (Mismatch): MPC acha que tem 2.2 MW de PV e 4.5 MWh de bateria
        params_mpc = dict(
            capacidade_maxima_bateria_wh=4500e3, eficiencia_carga=0.86, eficiencia_descarga=0.98,
            potencia_nominal_pv_w=2200e3, potencia_bateria_min_w=-PLANTA_PCS, potencia_bateria_max_w=PLANTA_PCS
        )

    controlador = MPC_Economico(
        horizonte_predicao=HORIZONTE_PREDICAO, horizonte_controle=HORIZONTE_CONTROLE,
        peso_lambda=PESO_LAMBDA, delta_potencia_bateria_max_w=PLANTA_PCS,
        irradiancia_referencia_w_m2=1000.0, passo_tempo_segundos=3600.0,
        soc_minimo=0.1, soc_maximo=0.9, peso_penalidade_soc=1e7, **params_mpc
    )

    hist = {k: [] for k in ("bat", "grid", "soc", "cumcost", "import_cost", "export_rev")}
    pot_anterior = 0.0

    for h in horas:
        soc_atual = bateria.get_soc()
        if caso == "C0":
            pref = 0.0
        elif caso == "C3":
            # Heurística Clássica (Rule-based BESS)
            hh = h % 24
            
            # Estima a geração fotovoltaica atual
            pot_pv_estimada_w = (irr_real[h] / 1000.0) * PLANTA_PV
            excedente_w = pot_pv_estimada_w - carga_real[h]

            if 18 <= hh <= 21:
                pref = PLANTA_PCS  # Descarrega tudo no pico para lucro máximo
            elif excedente_w > 0:
                pref = -excedente_w  # Bateria absorve o excedente solar (carga é negativa)
            else:
                pref = 0.0  # Em repouso
        else:
            ih = obter_previsao_ciclica(irr_real if caso == "C1" else irr_pred, h, HORIZONTE_PREDICAO)
            lh = obter_previsao_ciclica(carga_real if caso == "C1" else carga_pred, h, HORIZONTE_PREDICAO)
            b, s = obter_previsao_tarifa(h, HORIZONTE_PREDICAO)
            pref = controlador.calcular_u(soc_atual, pot_anterior, ih, lh, b, s)

        res = microrrede.executar_passo_simulacao(irr_real[h], carga_real[h], pref, buy_real[h], sell_real[h])
        
        pot_anterior = res["potencia_bateria_w"]
        hist["bat"].append(pot_anterior)
        hist["grid"].append(res["potencia_rede_w"])
        hist["soc"].append(res["soc_bateria"])
        hist["cumcost"].append(res["custo_acumulado_reais"])
        
        # Custos separados
        p_grid = res["potencia_rede_w"] / 1000.0 # kW
        if p_grid > 0:
            hist["import_cost"].append(p_grid * buy_real[h])
            hist["export_rev"].append(0.0)
        else:
            hist["import_cost"].append(0.0)
            hist["export_rev"].append(abs(p_grid) * sell_real[h])

    return {k: np.array(v) for k, v in hist.items()}, microrrede

if __name__ == "__main__":
    np.random.seed(42)
    NUM_HORAS = 168
    horas = np.arange(NUM_HORAS)
    
    # 1. PERFIL DE CARGA (7 dias)
    carga_predicao = carregar_perfil_carga_csv_168h()
    ruido_carga = np.random.normal(0, 5000, NUM_HORAS)
    carga_real = np.clip(carga_predicao + ruido_carga, 0, None)

    # 2. PERFIL DE IRRADIÂNCIA (7 dias)
    dias = horas // 24
    base_por_dia = 900 * np.random.normal(1.0, 0.2, dias[-1] + 1)
    base = base_por_dia[dias]
    irradiancia_predicao = np.clip(base * np.sin(np.pi * (horas - 6) / 12), 0, None)
    ruido_irrad = np.random.normal(1.0, 0.15, NUM_HORAS)
    irradiancia_real = np.clip(irradiancia_predicao * ruido_irrad, 0, None)
    irradiancia_real[irradiancia_predicao == 0] = 0
    
    buy_real, sell_real = obter_previsao_tarifa(0, NUM_HORAS)

    # 3. EXECUÇÃO
    print("="*60)
    print("MÉTRICAS PARA A TABELA II DO ARTIGO (7 Dias / 168h)")
    print("="*60)
    
    resultados = {}
    for c in ("C0", "C1", "C2", "C3"):
        res_hist, micro = run_case(c, horas, irradiancia_predicao, irradiancia_real, 
                                   carga_predicao, carga_real, buy_real, sell_real)
        resultados[c] = res_hist
        
        # Cálculos para a Tabela
        e_imp = micro.get_energia_comprada_kwh_acumulada()
        e_exp = micro.get_energia_vendida_kwh_acumulada()
        import_cost = np.sum(res_hist["import_cost"])
        export_rev = np.sum(res_hist["export_rev"])
        net_cost = import_cost - export_rev
        p_imp_max = np.max(res_hist["grid"]) / 1000.0
        p_exp_max = abs(np.min(res_hist["grid"])) / 1000.0
        
        # Métricas de Bateria (apenas C1 e C2)
        if c != "C0":
            e_thr = np.sum(abs(res_hist["bat"])) / 1000.0
            eq_cycles = e_thr / (2 * 900.0) # 900 kWh é a planta real
            soc_min, soc_max = np.min(res_hist["soc"]), np.max(res_hist["soc"])
        else:
            e_thr, eq_cycles, soc_min, soc_max = 0, 0, 0, 0

        print(f"[{c}]")
        print(f"  E_imp (kWh):      {e_imp:.2f}")
        print(f"  E_exp (kWh):      {e_exp:.2f}")
        print(f"  Import Cost (R$): {import_cost:.2f}")
        print(f"  Export Rev (R$):  {export_rev:.2f}")
        print(f"  Net Cost (R$):    {net_cost:.2f}")
        print(f"  P_imp^max (kW):   {p_imp_max:.2f}")
        print(f"  P_exp^max (kW):   {p_exp_max:.2f}")
        if c != "C0":
            print(f"  E_thr (kWh):      {e_thr:.2f} ({eq_cycles:.2f} ciclos eq.)")
            print(f"  SOC Interval:     {soc_min*100:.2f}% a {soc_max*100:.2f}%\n")
        else:
            print("")

    # =========================================================================
    # 4. FIGURAS NO ESTILO DO ARTIGO (Figs. 2 a 5)
    # =========================================================================
    PV_PLANTA_KW = 2000.0   # 2.0 MW (2000 kW)
    PV_MPC_KW    = 2200.0
    FIGSIZE      = (6.4, 4.0)
    TICKS_168    = np.arange(0, 176, 25)

    plt.rcParams.update({"font.size": 10, "axes.grid": False})

    # ---- Fig. 2: net load previsto x realizado -------------------------------
    net_pred_kw = carga_predicao / 1000.0 - irradiancia_predicao * PV_MPC_KW / 1000.0
    net_real_kw = carga_real / 1000.0 - irradiancia_real * PV_PLANTA_KW / 1000.0
    plt.figure(figsize=FIGSIZE)
    plt.plot(horas, net_pred_kw, color='tab:blue',   label='MPC predicted net load')
    plt.plot(horas, net_real_kw, color='tab:orange', label='Realized plant net load')
    plt.xlabel('Time (h)')
    plt.ylabel('Net load (kW)')
    plt.xticks(TICKS_168)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('fig_net_load_forecast.png')
    plt.close()

    # ---- Fig. 3: SOC da planta (C2), em % ------------------------------------
    plt.figure(figsize=FIGSIZE)
    plt.plot(horas, resultados["C2"]["soc"] * 100, color='tab:blue', label='Plant SOC')
    plt.axhline(10, color='tab:blue', label='MPC lower target')
    plt.axhline(90, color='tab:blue', label='MPC upper target')
    plt.xlabel('Time (h)')
    plt.ylabel('State of charge (%)')
    plt.xticks(TICKS_168)
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('fig_soc_closed_loop.png')
    plt.close()

    # ---- Fig. 4: Balanço de potências (C2), 72 h, barras sobrepostas ------------
    h72 = horas[:72]
    
    # 1. Separar o que é positivo (Fonte) e negativo (Sumidouro/Consumo)
    pv_kw = irradiancia_real[:72] * PV_PLANTA_KW / 1000.0
    load_kw = -carga_real[:72] / 1000.0  
    
    bat_kw = resultados["C2"]["bat"][:72] / 1000.0
    bat_descarga = np.maximum(bat_kw, 0)  
    bat_carga = np.minimum(bat_kw, 0)     
    
    grid_kw = resultados["C2"]["grid"][:72] / 1000.0
    grid_importacao = np.maximum(grid_kw, 0) 
    grid_exportacao = np.minimum(grid_kw, 0) 

    # 3. Criar a figura
    fig, ax = plt.subplots(figsize=(8, 4.5)) # Proporção clássica 16:9 compacta
    
    largura_barra = 1.0
    estilo_borda = {'edgecolor': 'white', 'linewidth': 0.4, 'zorder': 3}

    cor_pv = '#E6A01D'       # Ouro/Amarelo Muted
    cor_bat_desc = '#7FB97A' # Verde suave
    cor_bat_carg = '#2A7A3E' # Verde escuro
    cor_grid_imp = '#D9534F' # Vermelho/Coral (comprando, custo)
    cor_grid_exp = '#5BC0DE' # Azul claro (vendendo, receita)
    cor_carga = '#333333'    # Cinza escuro/Quase preto (Carga local base)
    
    # --- Empilhamento Positivo (Fontes) ---
    ax.bar(h72, pv_kw, color=cor_pv, label='PV Generation', width=largura_barra, **estilo_borda)
    ax.bar(h72, bat_descarga, bottom=pv_kw, color=cor_bat_desc, label='BESS Discharge', width=largura_barra, **estilo_borda)
    ax.bar(h72, grid_importacao, bottom=pv_kw + bat_descarga, color=cor_grid_imp, label='Grid Import', width=largura_barra, **estilo_borda)
    
    # --- Empilhamento Negativo (Sumidouros) ---
    ax.bar(h72, load_kw, color=cor_carga, label='Local Load', width=largura_barra, **estilo_borda)
    ax.bar(h72, bat_carga, bottom=load_kw, color=cor_bat_carg, label='BESS Charge', width=largura_barra, **estilo_borda)
    ax.bar(h72, grid_exportacao, bottom=load_kw + bat_carga, color=cor_grid_exp, label='Grid Export', width=largura_barra, **estilo_borda)
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Power (kW)')
    ax.axhline(0, color='black', linewidth=1.2, zorder=4)
    ax.set_xlim(-0.5, 71.5)
    ax.set_xticks(np.arange(0, 73, 12))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    
    plt.tight_layout()
    plt.savefig('fig_power_72h.png', bbox_inches='tight') # Usar bbox_inches='tight' evita cortar a legenda
    plt.close()

    # ---- Fig. 5: custo acumulado (a) + economia acumulada vs C0 (b) ------------
    c0 = resultados["C0"]["cumcost"]
    c1 = resultados["C1"]["cumcost"]
    c2 = resultados["C2"]["cumcost"]
    c3 = resultados["C3"]["cumcost"] # Extraindo dados do C3
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1]})
                                   
    ax1.plot(horas, c0, color='tab:blue',   label='C0: no BESS')
    ax1.plot(horas, c3, color='tab:purple', label='C3: rule-based (Classic BESS)')
    ax1.plot(horas, c2, color='tab:orange', label='C2: uncertain/mismatched MPC')
    ax1.plot(horas, c1, color='tab:green',  label='C1: perfect/matched MPC')
    ax1.set_ylabel('Cumulative net cost (R$)')
    ax1.legend(loc='best')
    ax1.set_title('(a) Cumulative net cost', fontsize=10)

    # Economia acumulada em relação ao C0 (positivo = BESS reduz o custo)
    ax2.plot(horas, c0 - c3, color='tab:purple', label='C3: rule-based (Classic BESS)')
    ax2.plot(horas, c0 - c2, color='tab:orange', label='C2: uncertain/mismatched MPC')
    ax2.plot(horas, c0 - c1, color='tab:green',  label='C1: perfect/matched MPC')
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.set_ylabel('Cumulative savings vs. C0 (R$)')
    ax2.set_xlabel('Time (h)')
    ax2.set_xticks(TICKS_168)
    ax2.legend(loc='best')
    ax2.set_title('(b) Savings relative to no BESS', fontsize=10)
    
    fig.tight_layout()
    fig.savefig('fig_cumulative_cost_comparison.png')
    plt.close(fig)

    # Economia acumulada em relação ao C0 (positivo = BESS reduz o custo)
    ax2.plot(horas, c0 - c2, color='tab:orange', label='C2: uncertain/mismatched MPC')
    ax2.plot(horas, c0 - c1, color='tab:green',  label='C1: perfect/matched MPC')
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.set_ylabel('Cumulative savings vs. C0 (R$)')
    ax2.set_xlabel('Time (h)')
    ax2.set_xticks(TICKS_168)
    ax2.legend(loc='best')
    ax2.set_title('(b) Savings relative to no BESS', fontsize=10)
    fig.tight_layout()
    fig.savefig('fig_cumulative_cost_comparison.png')
    plt.close(fig)

    print("As 4 figuras PDF (estilo do artigo) foram geradas no diretório.")