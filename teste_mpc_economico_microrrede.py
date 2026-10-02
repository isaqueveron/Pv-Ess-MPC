import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import csv
from datetime import datetime

from modelo_microrrede_pv_ess import (
    PainelFotovoltaicoVirtual,
    BateriaVirtual,
    MicrorredeVirtual,
)
from mpc_economico_microrrede import MPC_Economico

W1, W2 = 3.5, 7.16

PV_PLANTA_KW = 2000.0
PV_MPC_KW    = 2200.0
TICKS_168    = np.arange(0, 169, 24)      # múltiplos de 24 h (dias)
 
# Paleta Okabe-Ito (daltônicos) + estilos de linha distintos
# (legível também em impressão P&B). Mesma cor por cenário em todas as figuras.
COR = {"C0": "#000000", "C1": "#009E73", "C2": "#D55E00", "C3": "#CC79A7"}
LS  = {"C0": (0, (4, 2)), "C1": "-", "C2": "-", "C3": "-."}
LBL = {"C0": "C0: no BESS",
       "C1": "C1: perfect/matched MPC",
       "C2": "C2: uncertain/mismatched MPC",
       "C3": "C3: rule-based (classic BESS)"}
 
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "0.85", "grid.linewidth": 0.4, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7", "legend.fancybox": False,
    "legend.borderpad": 0.3, "legend.labelspacing": 0.25,
    "legend.handlelength": 1.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # fontes embutidas (exigência IEEE)
    "savefig.dpi": 600,
})
 
 
def finalizar(fig, nome):
    fig.savefig(f"{nome}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{nome}_v2.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
 
 
def eixo_tempo(ax, xmax=168):
    ax.set_xlim(0, xmax)
    ax.set_xticks(TICKS_168[TICKS_168 <= xmax])
    ax.xaxis.set_minor_locator(MultipleLocator(12))
 
 
def gerar_graficos(horas, carga_real, carga_predicao,
                   irradiancia_real, irradiancia_predicao, resultados):
 
    # ---- Fig. 2: Net load previsto x realizado (1 coluna) ----
    net_pred_kw = carga_predicao / 1000.0 - irradiancia_predicao * PV_MPC_KW / 1000.0
    net_real_kw = carga_real / 1000.0 - irradiancia_real * PV_PLANTA_KW / 1000.0
 
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(horas, net_real_kw, color="#E69F00", lw=0.9, label="Realized plant net load")
    ax.plot(horas, net_pred_kw, color="#0072B2", lw=0.9, ls="--", label="MPC predicted net load")
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Net load (kW)")
    eixo_tempo(ax)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2,
              frameon=False, columnspacing=1.5)
    finalizar(fig, "fig_net_load_forecast")
 
    # ---- Fig. 3: SOC da planta (C2) (1 coluna) ----
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    ax.axhspan(10, 90, color="#0072B2", alpha=0.07, lw=0)          # faixa operacional
    ax.axhline(10, color="0.35", lw=0.7, ls="--")
    ax.axhline(90, color="0.35", lw=0.7, ls="--", label="MPC SOC targets (10 % / 90 %)")
    ax.plot(horas, resultados["C2"]["soc"] * 100, color="#0072B2", lw=1.0, label="Plant SOC")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("State of charge (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    eixo_tempo(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2,
              frameon=False, columnspacing=1.5)
    finalizar(fig, "fig_soc_closed_loop")
 
        # ---- Fig. 4: Balanço de potências (C2), 72 h (1 coluna) ----
    N = 72
    pv_kw   = irradiancia_real[:N] * PV_PLANTA_KW / 1000.0
    load_kw = -carga_real[:N] / 1000.0
    bat_kw  = resultados["C2"]["bat"][:N] / 1000.0
    grid_kw = resultados["C2"]["grid"][:N] / 1000.0
    bat_desc, bat_carg = np.maximum(bat_kw, 0), np.minimum(bat_kw, 0)
    grid_imp, grid_exp = np.maximum(grid_kw, 0), np.minimum(grid_kw, 0)

    # Mesma paleta Okabe-Ito das demais figuras
    C_PV, C_BD, C_GI = "#E69F00", "#009E73", "#D55E00"
    C_LD, C_BC, C_GE = "#6E6E6E", "#0072B2", "#56B4E9"

    x = np.arange(N + 1)                       # degraus horários (step="post")
    def st(y): return np.append(y, y[-1])

    # pilhas positiva (fontes) e negativa (cargas/sumidouros)
    p0 = np.zeros(N)
    p1 = pv_kw
    p2 = p1 + bat_desc
    p3 = p2 + grid_imp
    n1 = load_kw
    n2 = n1 + bat_carg
    n3 = n2 + grid_exp

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    fk = dict(step="post", linewidth=0, zorder=3)
    ax.fill_between(x, st(p0), st(p1), color=C_PV, label="PV generation",  **fk)
    ax.fill_between(x, st(p1), st(p2), color=C_BD, label="BESS discharge", **fk)
    ax.fill_between(x, st(p2), st(p3), color=C_GI, label="Grid import",    **fk)
    ax.fill_between(x, st(p0), st(n1), color=C_LD, label="Local load",     **fk)
    ax.fill_between(x, st(n1), st(n2), color=C_BC, label="BESS charge",    **fk)
    ax.fill_between(x, st(n2), st(n3), color=C_GE, label="Grid export",    **fk)

    ax.axhline(0, color="black", lw=0.6, zorder=4)
    ax.set_xlim(0, N)
    ax.set_xticks(np.arange(0, N + 1, 12))
    ax.xaxis.set_minor_locator(MultipleLocator(6))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (kW)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, columnspacing=1.0, handlelength=1.0,
              handletextpad=0.4, fontsize=6.5)
    finalizar(fig, "fig_power_72h")
 
    # ---- Fig. 5: Custo acumulado (a) + economia vs C0 (b) (1 coluna) ----
    cum = {k: resultados[k]["cumcost"] for k in ("C0", "C1", "C2", "C3")}
    ordem_plot = ["C0", "C3", "C2", "C1"]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W1, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12})
    for k in ordem_plot:
        ax1.plot(horas, cum[k], color=COR[k], ls=LS[k], label=LBL[k])
    ax1.set_ylabel("Cumulative net cost (R$)")
    ax1.legend(loc="upper left", fontsize=6.5)
    ax1.text(0.98, 0.04, "(a)", transform=ax1.transAxes, ha="right", va="bottom",
             fontweight="bold", fontsize=8)
 
    for k in ("C3", "C2", "C1"):
        ax2.plot(horas, cum["C0"] - cum[k], color=COR[k], ls=LS[k], label=LBL[k])
    ax2.axhline(0, color="0.4", lw=0.6)
    ax2.set_ylabel("Savings vs. C0 (R$)")
    ax2.set_xlabel("Time (h)")
    ax2.text(0.98, 0.04, "(b)", transform=ax2.transAxes, ha="right", va="bottom",
             fontweight="bold", fontsize=8)
    eixo_tempo(ax2)
    for a in (ax1, ax2):
        a.yaxis.set_minor_locator(AutoMinorLocator(2))
    finalizar(fig, "fig_cumulative_cost_comparison")

# ----------------------------------------------------------------- #
# 1. PREPARAÇÃO DE DADOS
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
# 2. SIMULAÇÃO DOS CENÁRIOS
# ----------------------------------------------------------------- #
def run_case(caso, horas, irr_pred, irr_real, carga_pred, carga_real, buy_real, sell_real):
    HORIZONTE_PREDICAO = 24
    HORIZONTE_CONTROLE = 12
    PESO_LAMBDA = 5e-9

    # Parâmetros da planta real (Baseado em BESS C&I de 4 horas)
    PLANTA_PV = 2000e3
    PLANTA_CAP = 4000e3
    PLANTA_PCS = 1000e3
    PLANTA_EFF = 0.90
    
    painel = PainelFotovoltaicoVirtual(potencia_nominal_w=PLANTA_PV)
    bateria = BateriaVirtual(
        capacidade_maxima_wh=PLANTA_CAP, eficiencia_carga=PLANTA_EFF, eficiencia_descarga=PLANTA_EFF,
        potencia_maxima_carga_w=PLANTA_PCS, potencia_maxima_descarga_w=PLANTA_PCS,
        soc_minimo=0.1, soc_maximo=0.9, soc_inicial=0.48, passo_tempo_segundos=3600.0,
    )
    microrrede = MicrorredeVirtual(painel, bateria, passo_tempo_segundos=3600.0)

    if caso == "C1":
        # C1 (Matched MPC): Parâmetros perfeitamente alinhados com a planta
        params_mpc = dict(
            capacidade_maxima_bateria_wh=PLANTA_CAP, eficiencia_carga=PLANTA_EFF, eficiencia_descarga=PLANTA_EFF,
            potencia_nominal_pv_w=PLANTA_PV, potencia_bateria_min_w=-PLANTA_PCS, potencia_bateria_max_w=PLANTA_PCS
        )
    else:
        # C2 (Mismatched MPC): Parâmetros com incerteza na modelagem
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
            # C3: Rule-based Heuristic BESS
            hh = h % 24
            pot_pv_estimada_w = (irr_real[h] / 1000.0) * PLANTA_PV
            excedente_w = pot_pv_estimada_w - carga_real[h]

            if 18 <= hh <= 21:
                pref = PLANTA_PCS 
            elif excedente_w > 0:
                pref = -excedente_w 
            else:
                pref = 0.0 
                
        else:
            # MPC (C1 e C2)
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
        
        p_grid = res["potencia_rede_w"] / 1000.0 
        if p_grid > 0:
            hist["import_cost"].append(p_grid * buy_real[h])
            hist["export_rev"].append(0.0)
        else:
            hist["import_cost"].append(0.0)
            hist["export_rev"].append(abs(p_grid) * sell_real[h])

    return {k: np.array(v) for k, v in hist.items()}, microrrede


# ----------------------------------------------------------------- #
# 3. EXECUÇÃO PRINCIPAL
# ----------------------------------------------------------------- #
if __name__ == "__main__":
    np.random.seed(42)
    NUM_HORAS = 168
    horas = np.arange(NUM_HORAS)
    
    carga_predicao = carregar_perfil_carga_csv_168h()
    ruido_carga = np.random.normal(0, 5000, NUM_HORAS)
    carga_real = np.clip(carga_predicao + ruido_carga, 0, None)

    dias = horas // 24
    base_por_dia = 900 * np.random.normal(1.0, 0.2, dias[-1] + 1)
    base = base_por_dia[dias]
    irradiancia_predicao = np.clip(base * np.sin(np.pi * (horas - 6) / 12), 0, None)
    ruido_irrad = np.random.normal(1.0, 0.15, NUM_HORAS)
    irradiancia_real = np.clip(irradiancia_predicao * ruido_irrad, 0, None)
    irradiancia_real[irradiancia_predicao == 0] = 0
    
    buy_real, sell_real = obter_previsao_tarifa(0, NUM_HORAS)

    print("\n" + "="*70)
    print("MÉTRICAS PARA A TABELA II DO ARTIGO (7 Dias / 168h)")
    print("="*70)
    
    resultados = {}
    for c in ("C0", "C1", "C2", "C3"):
        res_hist, micro = run_case(c, horas, irradiancia_predicao, irradiancia_real, 
                                   carga_predicao, carga_real, buy_real, sell_real)
        resultados[c] = res_hist
        
        e_imp = micro.get_energia_comprada_kwh_acumulada()
        e_exp = micro.get_energia_vendida_kwh_acumulada()
        import_cost = np.sum(res_hist["import_cost"])
        export_rev = np.sum(res_hist["export_rev"])
        net_cost = import_cost - export_rev
        p_imp_max = np.max(res_hist["grid"]) / 1000.0
        p_exp_max = abs(np.min(res_hist["grid"])) / 1000.0
        
        if c != "C0":
            e_thr = np.sum(abs(res_hist["bat"])) / 1000.0
            eq_cycles = e_thr / (2 * 4000.0) 
            soc_min, soc_max = np.min(res_hist["soc"]), np.max(res_hist["soc"])
        else:
            e_thr, eq_cycles, soc_min, soc_max = 0, 0, 0, 0

        print(f"[{c}]")
        print(f"  E_imp (kWh):      {e_imp:>10.2f}")
        print(f"  E_exp (kWh):      {e_exp:>10.2f}")
        print(f"  Import Cost (R$): {import_cost:>10.2f}")
        print(f"  Export Rev (R$):  {export_rev:>10.2f}")
        print(f"  Net Cost (R$):    {net_cost:>10.2f}")
        print(f"  P_imp^max (kW):   {p_imp_max:>10.2f}")
        print(f"  P_exp^max (kW):   {p_exp_max:>10.2f}")
        if c != "C0":
            print(f"  E_thr (kWh):      {e_thr:>10.2f} ({eq_cycles:.2f} eq. cycles)")
            print(f"  SOC Interval:     {soc_min*100:>10.2f}% a {soc_max*100:.2f}%\n")
        else:
            print("")

    
    gerar_graficos(horas, carga_real, carga_predicao,
                   irradiancia_real, irradiancia_predicao, resultados)

    print("\nExecução concluída. Figuras PDF/PNG geradas no diretório.")
