import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import random as rd

# Planta REAL (não-linear, com eficiências de carga/descarga distintas)
from modelo_microrrede_pv_ess import (
    PainelFotovoltaicoVirtual,
    BateriaVirtual,
    MicrorredeVirtual,
)

from mpc_economico_microrrede import MPC_Economico

# ----------------------------------------------------------------- #
# FUNÇÃO PARA LER E PROCESSAR OS FICHEIROS CSV (SEM PANDAS)
# ----------------------------------------------------------------- #
def ler_e_limpar_csv(caminho):
    dados_horarios = {}
    
    with open(caminho, mode='r', encoding='utf-8') as ficheiro:
        leitor = csv.DictReader(ficheiro)
        for linha in leitor:
            tempo_str = linha['Time']
            potencia_str = linha['Potência ativa']
            
            # Converter a string de tempo num objeto datetime
            dt = datetime.strptime(tempo_str, '%Y-%m-%d %H:%M:%S')
            # Truncar para a hora certa (ex: 15:32:00 -> 15:00:00)
            chave_hora = dt.replace(minute=0, second=0)
            
            # Limpar a string da potência e converter para float
            valor = float(potencia_str.replace(' kW', '').replace(',', '.'))
            
            # Agrupar os valores para calcular a média posteriormente
            if chave_hora not in dados_horarios:
                dados_horarios[chave_hora] = {'soma': 0.0, 'contagem': 0}
            
            dados_horarios[chave_hora]['soma'] += valor
            dados_horarios[chave_hora]['contagem'] += 1
            
    # Calcular a média horária
    medias = {}
    for hora in sorted(dados_horarios.keys()):
        medias[hora] = dados_horarios[hora]['soma'] / dados_horarios[hora]['contagem']
        
    return medias

def carregar_perfil_carga_csv():
    caminho_al4 = "Potência Ativa-data-2026-09-17 15_31_43_al4_dia15_09_26.csv"
    caminho_ru = "Potência Ativa-data-2026-09-17 15_32_17_RU_dia15_09_26.csv"
    
    medias_al4 = ler_e_limpar_csv(caminho_al4)
    medias_ru = ler_e_limpar_csv(caminho_ru)
    
    carga_total_w = []
    
    # Sincronizar as horas e somar as potências (AL4 + RU)
    for hora in sorted(medias_al4.keys()):
        if hora in medias_ru:
            soma_kw = medias_al4[hora] + medias_ru[hora]
            carga_total_w.append(soma_kw * 1000.0) # Converter para Watts
            
    return np.array(carga_total_w)


if __name__ == "__main__":

    # ----------------------------------------------------------------- #
    # 1. PARÂMETROS DO PROJETO UFSC (CMD01 + ESCENÁRIO BASE BESS)
    # ----------------------------------------------------------------- #
    POTENCIA_PV_W = 495e3        # 495 kW AC (CMD01: RU + Cultura + etc.)
    CAPACIDADE_BESS_WH = 900e3   # 900 kWh (Escenario Base Recomendado)
    POTENCIA_PCS_W = 300e3       # 300 kW (Potência do Inversor da Bateria)
    EFICIENCIA = 0.90            # Eficiência realista de 90%
    
    HORIZONTE_PREDICAO = 24
    HORIZONTE_CONTROLE = 12
    PESO_LAMBDA = 5e-9

    # ----------------------------------------------------------------- #
    # 2. CARREGAMENTO DOS DADOS REAIS DO CSV
    # ----------------------------------------------------------------- #
    carga_predicao = carregar_perfil_carga_csv()
    NUM_HORAS_SIMULACAO = len(carga_predicao) # Ajusta automaticamente (ex: 48h)
    
    # Ruído menor, pois o perfil base (CSV) já incorpora as flutuações reais
    ruido_carga = np.random.normal(0, 5000, NUM_HORAS_SIMULACAO) 
    carga_real = np.clip(carga_predicao + ruido_carga, 0, None)

    # ----------------------------------------------------------------- #
    # Instanciamento da planta onde o controlador vai atuar (Realidade)
    # ----------------------------------------------------------------- #
    painel = PainelFotovoltaicoVirtual(potencia_nominal_w=POTENCIA_PV_W)
    bateria = BateriaVirtual(
        capacidade_maxima_wh=CAPACIDADE_BESS_WH,
        eficiencia_carga=EFICIENCIA,
        eficiencia_descarga=EFICIENCIA,
        potencia_maxima_carga_w=POTENCIA_PCS_W,
        potencia_maxima_descarga_w=POTENCIA_PCS_W,
        soc_minimo=0.1,
        soc_maximo=0.9,
        soc_inicial=0.48,
        passo_tempo_segundos=3600.0,
    )
    microrrede = MicrorredeVirtual(painel, bateria, passo_tempo_segundos=3600.0)

    # ----------------------------------------------------------------- #
    # Projeto do controlador GPC econômico
    # ----------------------------------------------------------------- #
    controlador_mpc_economico = MPC_Economico(
        capacidade_maxima_bateria_wh=CAPACIDADE_BESS_WH,
        eficiencia_carga=EFICIENCIA,
        eficiencia_descarga=EFICIENCIA,
        potencia_nominal_pv_w=POTENCIA_PV_W,
        horizonte_predicao=HORIZONTE_PREDICAO,
        horizonte_controle=HORIZONTE_CONTROLE,
        peso_lambda=PESO_LAMBDA,
        potencia_bateria_min_w=-POTENCIA_PCS_W,
        potencia_bateria_max_w=POTENCIA_PCS_W,
        delta_potencia_bateria_max_w=POTENCIA_PCS_W,
        irradiancia_referencia_w_m2=1000.0,
        passo_tempo_segundos=3600.0,
        soc_minimo=0.1,
        soc_maximo=0.9,
        peso_penalidade_soc=1e7
    )

    # ----------------------------------------------------------------- #
    # Perfis de Irradiância e Tarifas
    # ----------------------------------------------------------------- #
    horas = np.arange(NUM_HORAS_SIMULACAO)
    dias = horas // 24
    num_dias = dias[-1] + 1
    
    # Pico médio 900 W/m²
    base_por_dia = 900 * np.random.normal(1.0, 0.2, num_dias) 
    base = base_por_dia[dias]

    irradiancia_predicao = np.clip(base * np.sin(np.pi * (horas - 6) / 12), 0, None)
    ruido_irrad = np.random.normal(1.0, 0.15, NUM_HORAS_SIMULACAO)
    irradiancia_real = np.clip(irradiancia_predicao * ruido_irrad, 0, None)
    irradiancia_real[irradiancia_predicao == 0] = 0

    def obter_previsao_tarifa(hora_atual, horizonte):
        horas_futuras = (hora_atual + np.arange(horizonte)) % 24
        preco_compra = np.where((horas_futuras >= 18) & (horas_futuras <= 21), 1.98, 0.58)
        preco_venda = 0.6 * preco_compra
        return preco_compra, preco_venda
        
    preco_compra_perfil, preco_venda_perfil = obter_previsao_tarifa(0, NUM_HORAS_SIMULACAO)
    
    def obter_previsao_ciclica(perfil, hora_atual, horizonte):
        indices = (hora_atual + np.arange(horizonte)) % len(perfil)
        return perfil[indices]

    # ----------------------------------------------------------------- #
    # Execução da simulação em malha fechada
    # ----------------------------------------------------------------- #
    hist_pv, hist_bat, hist_grid, hist_soc = [], [], [], []
    hist_custo_instantaneo, hist_custo_acumulado = [], []

    potencia_bateria_anterior = 0.0
    soc_atual = bateria.get_soc() 

    for h in horas:
        irradiancia_futura_mpc = obter_previsao_ciclica(irradiancia_predicao, h, HORIZONTE_PREDICAO)
        carga_futura_mpc = obter_previsao_ciclica(carga_predicao, h, HORIZONTE_PREDICAO)
        preco_compra_futuro, preco_venda_futuro = obter_previsao_tarifa(h, HORIZONTE_PREDICAO)

        potencia_bateria_referencia_w = controlador_mpc_economico.calcular_u(
            soc_atual=soc_atual,
            potencia_bateria_anterior=potencia_bateria_anterior,
            irradiancia_futura_w_m2=irradiancia_futura_mpc,
            carga_futura_w=carga_futura_mpc,
            preco_compra_futuro_reais_kwh=preco_compra_futuro,
            preco_venda_futuro_reais_kwh=preco_venda_futuro,
        )

        resultado = microrrede.executar_passo_simulacao(
            irradiancia_w_m2=irradiancia_real[h],
            potencia_carga_consumidora_w=carga_real[h],
            potencia_bateria_referencia_w=potencia_bateria_referencia_w,
            preco_compra_reais_kwh=preco_compra_perfil[h],
            preco_venda_reais_kwh=preco_venda_perfil[h],
        )

        potencia_bateria_anterior = resultado['potencia_bateria_w']
        soc_atual = resultado['soc_bateria'] 

        hist_pv.append(resultado['potencia_pv_w'])
        hist_bat.append(resultado['potencia_bateria_w'])
        hist_grid.append(resultado['potencia_rede_w'])
        hist_soc.append(resultado['soc_bateria'])
        hist_custo_instantaneo.append(resultado['custo_instantaneo_reais'])
        hist_custo_acumulado.append(resultado['custo_acumulado_reais'])

    hist_pv_arr = np.array(hist_pv)
    hist_bat_arr = np.array(hist_bat)
    hist_grid_arr = np.array(hist_grid)
    hist_soc_arr = np.array(hist_soc)
    hist_custo_instantaneo_arr = np.array(hist_custo_instantaneo)
    hist_custo_acumulado_arr = np.array(hist_custo_acumulado)

    # ----------------------------------------------------------------- #
    # RESULTADOS
    # ----------------------------------------------------------------- #
    print(f"\n--- RESULTADOS UFSC (CMD01) - {NUM_HORAS_SIMULACAO} Horas ---")
    print(f"Custo total do período: R$ {microrrede.get_custo_acumulado_reais():.2f}")
    print(f"Energia comprada da rede: {microrrede.get_energia_comprada_kwh_acumulada():.2f} kWh")
    print(f"Energia vendida à rede:   {microrrede.get_energia_vendida_kwh_acumulada():.2f} kWh")

    # =========================================================================
    # FIGURA 1: SOC e Sinal de Controle
    # =========================================================================
    fig1, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(horas, hist_soc_arr, label='SOC Real', color='teal', marker='o', linestyle='-', markersize=3)
    axs[0].axhline(controlador_mpc_economico.soc_minimo, color='gray', linestyle=':', linewidth=1)
    axs[0].axhline(controlador_mpc_economico.soc_maximo, color='gray', linestyle=':', linewidth=1)
    axs[0].set_title('SOC Resultante na Planta Real')
    axs[0].set_ylim(0, 1.0)
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    axs[1].plot(horas, hist_bat_arr, label='Potência Bateria (W)', color='blue', marker='o', linestyle='None', markersize=3)
    axs[1].axhline(0, color='black', linewidth=1.0)
    axs[1].set_title('Sinal de Controle Aplicado (Pbat)')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    axs[2].plot(horas, preco_compra_perfil, label='Compra (R$/kWh)', color='crimson', marker='o', linestyle='None', markersize=3)
    axs[2].plot(horas, preco_venda_perfil, label='Venda (R$/kWh)', color='seagreen', marker='s', linestyle='None', markersize=3)
    axs[2].set_title('Tarifas')
    axs[2].set_xlabel('Horas (h)')
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Soc_Pot_bat_tarifas.svg',format='svg')

    # =========================================================================
    # FIGURA 2: Balanço Energético da Microrrede
    # =========================================================================
    pv_geracao = hist_pv_arr
    bat_descarga = np.maximum(0, hist_bat_arr)
    grid_importacao = np.maximum(0, hist_grid_arr)

    carga_consumo = -carga_real
    bat_carga = np.minimum(0, hist_bat_arr)
    grid_exportacao = np.minimum(0, hist_grid_arr)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.bar(horas, pv_geracao, label='Geração PV (Real)', color='gold', edgecolor='black')
    ax2.bar(horas, bat_descarga, bottom=pv_geracao, label='Bat (Descarga)', color='purple', edgecolor='black')
    ax2.bar(horas, grid_importacao, bottom=pv_geracao + bat_descarga, label='Rede (Importação)', color='gray', edgecolor='black')

    ax2.bar(horas, carga_consumo, label='Carga (Real)', color='red', edgecolor='black')
    ax2.bar(horas, bat_carga, bottom=carga_consumo, label='Bat (Carga)', color='blue', edgecolor='black')
    ax2.bar(horas, grid_exportacao, bottom=carga_consumo + bat_carga, label='Rede (Exportação)', color='green', edgecolor='black')

    ax2.axhline(0, color='black', linewidth=1.5)
    ax2.set_title('Balanço Energético da Microrrede Real', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Horas (h)')
    ax2.set_ylabel('Potência (W)')
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Balanco_energetico.svg',format='svg')

    # =========================================================================
    # FIGURA 3: Custos
    # =========================================================================
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    cores_custo = np.where(hist_custo_instantaneo_arr >= 0, 'firebrick', 'forestgreen')
    axs3[0].bar(horas, hist_custo_instantaneo_arr, color=cores_custo, edgecolor='black')
    axs3[0].axhline(0, color='black', linewidth=1.0)
    axs3[0].set_title('Custo Instantâneo Real (R$)')
    axs3[0].grid(True, axis='y', linestyle=':', alpha=0.7)

    axs3[1].plot(horas, hist_custo_acumulado_arr, label='Custo Acumulado (R$)', color='navy', marker='o', linestyle='-', markersize=3)
    axs3[1].axhline(0, color='black', linewidth=1.0)
    axs3[1].set_title('Custo Acumulado Real da Microrrede')
    axs3[1].set_xlabel('Horas (h)')
    axs3[1].legend()
    axs3[1].grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Custo.svg',format='svg')

    # =========================================================================
    # FIGURA 4: PREDIÇÃO DO MPC vs REALIDADE DA PLANTA
    # =========================================================================
    fig4, axs4 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs4[0].plot(horas, irradiancia_predicao, label='Predição do MPC (Base limpa)', color='orange', linewidth=2)
    axs4[0].plot(horas, irradiancia_real, label='Planta Real (Com variação)', color='red', linestyle='--', alpha=0.8)
    axs4[0].set_title('Irradiância: Expectativa do MPC vs O que a Planta Sentiu')
    axs4[0].set_ylabel('W/m²')
    axs4[0].legend()
    axs4[0].grid(True, linestyle=':', alpha=0.7)

    axs4[1].plot(horas, carga_predicao, label='Predição do MPC (Base limpa)', color='blue', linewidth=2)
    axs4[1].plot(horas, carga_real, label='Planta Real (Com variação)', color='purple', linestyle='--', alpha=0.8)
    axs4[1].set_title('Carga (Demanda): Expectativa do MPC vs O que a Planta Sentiu')
    axs4[1].set_ylabel('W')
    axs4[1].set_xlabel('Horas da Simulação (h)')
    axs4[1].legend()
    axs4[1].grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('modelo_planta_expectativa.svg',format='svg')
    
    plt.show()
