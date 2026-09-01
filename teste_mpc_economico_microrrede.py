import numpy as np
import matplotlib.pyplot as plt
import random as rd

from modelo_microrrede_pv_ess import (
    PainelFotovoltaicoVirtual,
    BateriaVirtual,
    MicrorredeVirtual,
)
from mpc_economico_microrrede import MPC_Economico

# --------------------------------------------------------------------------- #
# EXEMPLO DE USO EM MALHA FECHADA: MPC ECONÔMICO (MINIMIZA CUSTO)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # ----------------------------------------------------------------- #
    # Instanciamento da planta onde o controlador vai atuar (Realidade)
    # ----------------------------------------------------------------- #
    painel = PainelFotovoltaicoVirtual(potencia_nominal_w=400e3)
    bateria = BateriaVirtual(
        capacidade_maxima_wh=874e3,
        eficiencia_carga=0.88,
        eficiencia_descarga=0.94,
        potencia_maxima_carga_w=294e3,
        potencia_maxima_descarga_w=314e3,
        soc_minimo=0.0,
        soc_maximo=1.0,
        soc_inicial=0.48,
        passo_tempo_segundos=3600.0,   # Ts = 1h
    )
    microrrede = MicrorredeVirtual(painel, bateria, passo_tempo_segundos=3600.0)

    # ----------------------------------------------------------------- #
    # Instanciamento da planta para o mpc (Predição - ligeiramente diferente)
    # ----------------------------------------------------------------- #
    painel_mpc = PainelFotovoltaicoVirtual(potencia_nominal_w=425e3)
    bateria_mpc = BateriaVirtual(
        capacidade_maxima_wh=900e3,
        eficiencia_carga=0.95,
        eficiencia_descarga=0.95,
        potencia_maxima_carga_w=300e3,
        potencia_maxima_descarga_w=300e3,
        soc_minimo=0.0,
        soc_maximo=1.0,
        soc_inicial=0.5,
        passo_tempo_segundos=3600.0,   # Ts = 1h
    )
    microrrede_mpc = MicrorredeVirtual(painel_mpc, bateria_mpc, passo_tempo_segundos=3600.0)
    
    # ----------------------------------------------------------------- #
    # Projeto do controlador MPC
    # ----------------------------------------------------------------- #
    NUM_HORAS_SIMULACAO = 24 * 7
    HORIZONTE_PREDICAO = 24
    HORIZONTE_CONTROLE = 12
    PESO_LAMBDA = 5e-9

    controlador_mpc_economico = MPC_Economico(
        microrrede=microrrede_mpc,
        horizonte_predicao=HORIZONTE_PREDICAO,
        horizonte_controle=HORIZONTE_CONTROLE,
        peso_lambda=PESO_LAMBDA,
        potencia_bateria_min_w=-300e3,
        potencia_bateria_max_w=300e3,
        delta_potencia_bateria_max_w=100e3,
        soc_minimo=0.1,
        soc_maximo=0.9,
        peso_penalidade_soc=1e7
    )

    # ----------------------------------------------------------------- #
    # Perfis: Criação da Base (Predição) e da Realidade (Planta)
    # ----------------------------------------------------------------- #
    horas = np.arange(NUM_HORAS_SIMULACAO)
    
    # 1. Irradiância: Base vs Real
    irradiancia_predicao = np.clip(800 * np.sin(np.pi * (horas - 6) / 12), 0, None)
    
    # Adicionando ruído para a realidade (simulando nuvens/variações)
    ruido_irrad = np.random.normal(1.0, 0.2, NUM_HORAS_SIMULACAO)
    irradiancia_real = np.clip(irradiancia_predicao*ruido_irrad, 0, None)
    # Zera a irradiância real à noite (onde a predição também é zero)
    irradiancia_real[irradiancia_predicao == 0] = 0

    # 2. Carga: Base vs Real
    def gerar_perfil_carga_base(num_horas_simulacao):
        perfil_base_kw = np.array([
            150, 140, 130, 125, 125, 130,   # 00h-05h
            150, 200, 260, 290, 310, 380,   # 06h-11h
            420, 400, 340, 280, 260, 280,   # 12h-17h
            340, 370, 360, 320, 220, 180,   # 18h-23h
        ])
        perfil_base_w = perfil_base_kw * 1000.0
        indices = np.arange(num_horas_simulacao) % len(perfil_base_w)
        return perfil_base_w[indices]
        
    carga_predicao = gerar_perfil_carga_base(NUM_HORAS_SIMULACAO)
    
    # Adicionando ruído aleatório para a carga real (máquinas ligando/desligando)
    ruido_carga = np.random.normal(0, 20000, NUM_HORAS_SIMULACAO)
    carga_real = carga_predicao + ruido_carga

    # 3. Tarifas
    def obter_previsao_tarifa(hora_atual, horizonte):
        horas_futuras = (hora_atual + np.arange(horizonte)) % 24
        preco_compra = np.where(
            (horas_futuras >= 18) & (horas_futuras <= 21), 1.98,
            np.where((horas_futuras >= 0) & (horas_futuras <= 5), 0.58, 0.58)
        )
        preco_venda = preco_compra * 0.0
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

    for h in horas:
        # MPC ENXERGA APENAS A PREDIÇÃO (O que ele ACHA que vai acontecer)
        irradiancia_futura_mpc = obter_previsao_ciclica(irradiancia_predicao, h, HORIZONTE_PREDICAO)
        carga_futura_mpc = obter_previsao_ciclica(carga_predicao, h, HORIZONTE_PREDICAO)
        preco_compra_futuro, preco_venda_futuro = obter_previsao_tarifa(h, HORIZONTE_PREDICAO)

        # 1) Controlador calcula a referência baseado nas predições perfeitas base
        potencia_bateria_referencia_w = controlador_mpc_economico.calcular_u(
            potencia_bateria_anterior=potencia_bateria_anterior,
            irradiancia_futura_w_m2=irradiancia_futura_mpc,
            carga_futura_w=carga_futura_mpc,
            preco_compra_futuro_reais_kwh=preco_compra_futuro,
            preco_venda_futuro_reais_kwh=preco_venda_futuro,
        )

        # 2) Planta real SENTE AS VARIÁVEIS REAIS (com ruído) e executa a ação do MPC
        resultado = microrrede.executar_passo_simulacao(
            irradiancia_w_m2=irradiancia_real[h],
            potencia_carga_consumidora_w=carga_real[h],
            potencia_bateria_referencia_w=potencia_bateria_referencia_w,
            preco_compra_reais_kwh=preco_compra_perfil[h],
            preco_venda_reais_kwh=preco_venda_perfil[h],
        )

        potencia_bateria_anterior = resultado['potencia_bateria_w']

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

    # =========================================================================
    # FIGURA 4: PREDIÇÃO DO MPC vs REALIDADE DA PLANTA
    # =========================================================================
    fig4, axs4 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Plot 1: Irradiância (Predição vs Real)
    axs4[0].plot(horas, irradiancia_predicao, label='Predição do MPC (Base limpa)', color='orange', linewidth=2)
    axs4[0].plot(horas, irradiancia_real, label='Planta Real (Com variação)', color='red', linestyle='--', alpha=0.8)
    axs4[0].set_title('Irradiância: Expectativa do MPC vs O que a Planta Sentiu')
    axs4[0].set_ylabel('W/m²')
    axs4[0].legend()
    axs4[0].grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Carga (Predição vs Real)
    axs4[1].plot(horas, carga_predicao, label='Predição do MPC (Base limpa)', color='blue', linewidth=2)
    axs4[1].plot(horas, carga_real, label='Planta Real (Com variação)', color='purple', linestyle='--', alpha=0.8)
    axs4[1].set_title('Carga (Demanda): Expectativa do MPC vs O que a Planta Sentiu')
    axs4[1].set_ylabel('W')
    axs4[1].set_xlabel('Horas da Simulação (h)')
    axs4[1].legend()
    axs4[1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    # Mostra todos os gráficos juntos
    plt.show()

    # Resumo econômico
    print(f"\nCusto total do dia: R$ {microrrede.get_custo_acumulado_reais():.2f}")
    print(f"Energia comprada da rede: {microrrede.get_energia_comprada_kwh_acumulada():.2f} kWh")
    print(f"Energia vendida à rede:   {microrrede.get_energia_vendida_kwh_acumulada():.2f} kWh")
