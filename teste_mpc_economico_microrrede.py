import numpy as np
import matplotlib.pyplot as plt

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
    # Instanciamento da planta
    # ----------------------------------------------------------------- #
    painel = PainelFotovoltaicoVirtual(potencia_nominal_w=425e3)
    bateria = BateriaVirtual(
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
    microrrede = MicrorredeVirtual(painel, bateria, passo_tempo_segundos=3600.0)

    # ----------------------------------------------------------------- #
    # Projeto do controlador: a própria MicrorredeVirtual (painel +
    # bateria + contabilidade de custo) é passada como "gerador de
    # predições" -- o controlador roda cópias dela para simular o futuro
    # econômico.
    # ----------------------------------------------------------------- #.
    NUM_HORAS_SIMULACAO = 24 * 7
    HORIZONTE_PREDICAO = 24
    HORIZONTE_CONTROLE = 2
    PESO_LAMBDA = 1e-10

    controlador_mpc_economico = MPC_Economico(
        microrrede=microrrede,
        horizonte_predicao=HORIZONTE_PREDICAO,
        horizonte_controle=HORIZONTE_CONTROLE,
        peso_lambda=PESO_LAMBDA,
        potencia_bateria_min_w=-300e3,
        potencia_bateria_max_w=300e3,
        delta_potencia_bateria_max_w=100e3,
        soc_minimo=0.1,
        soc_maximo=0.9
    )

    # ----------------------------------------------------------------- #
    # Perfis sintéticos de 24h
    # ----------------------------------------------------------------- #
    horas = np.arange(NUM_HORAS_SIMULACAO)
    irradiancia_perfil = np.clip(800 * np.sin(np.pi * (horas - 6) / 12), 0, None)
    
    def gerar_perfil_carga_cmd01(num_horas_simulacao):
        # Perfil-base de 24h [kW] -- ver tabela de justificativa
        perfil_base_kw = np.array([
            150, 140, 130, 125, 125, 130,   # 00h-05h (madrugada/vale)
            150, 200, 260, 290, 310, 380,   # 06h-11h (rampa + pico almoço)
            420, 400, 340, 280, 260, 280,   # 12h-17h (almoço + tarde)
            340, 370, 360, 320, 220, 180,   # 18h-23h (ponta/jantar + declínio)
        ])

        perfil_base_w = perfil_base_kw * 1000.0

        indices = np.arange(num_horas_simulacao) % len(perfil_base_w)
        carga_perfil_w = perfil_base_w[indices]

        return carga_perfil_w
        
    carga_perfil = gerar_perfil_carga_cmd01(NUM_HORAS_SIMULACAO)

    def obter_previsao_ciclica(perfil, hora_atual, horizonte):
        indices = (hora_atual + np.arange(horizonte)) % len(perfil)
        return perfil[indices]


    def obter_previsao_tarifa(hora_atual, horizonte):
        """
        Previsão de TARIFA (preço de compra/venda), calculada diretamente
        pela REGRA tarifária.
        """
        horas_futuras = (hora_atual + np.arange(horizonte)) % 24

        preco_compra = np.where(
            (horas_futuras >= 18) & (horas_futuras <= 21), 1.98,
            np.where((horas_futuras >= 0) & (horas_futuras <= 5), 0.58, 0.58)
        )
        preco_venda = preco_compra * 0.0

        return preco_compra, preco_venda
        
    preco_compra_perfil, preco_venda_perfil = obter_previsao_tarifa(0, NUM_HORAS_SIMULACAO)
    
    # ----------------------------------------------------------------- #
    # Execução da simulação em malha fechada
    # ----------------------------------------------------------------- #
    hist_pv, hist_bat, hist_grid, hist_soc = [], [], [], []
    hist_custo_instantaneo, hist_custo_acumulado = [], []

    potencia_bateria_anterior = 0.0

    for h in horas:
        irradiancia_futura = obter_previsao_ciclica(irradiancia_perfil, h, HORIZONTE_PREDICAO)
        carga_futura = obter_previsao_ciclica(carga_perfil, h, HORIZONTE_PREDICAO)
        preco_compra_futuro, preco_venda_futuro = obter_previsao_tarifa(h, HORIZONTE_PREDICAO)

        # 1) Controlador econômico: simula o próprio modelo da microrrede
        #    internamente e calcula a potência de referência que minimiza
        #    o custo previsto
        potencia_bateria_referencia_w = controlador_mpc_economico.calcular_u(
            potencia_bateria_anterior=potencia_bateria_anterior,
            irradiancia_futura_w_m2=irradiancia_futura,
            carga_futura_w=carga_futura,
            preco_compra_futuro_reais_kwh=preco_compra_futuro,
            preco_venda_futuro_reais_kwh=preco_venda_futuro,
        )

        # 2) Planta real: aplica a referência calculada
        resultado = microrrede.executar_passo_simulacao(
            irradiancia_w_m2=irradiancia_perfil[h],
            potencia_carga_consumidora_w=carga_perfil[h],
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
    # FIGURA 1: SOC e Sinal de Controle (Pbat) resultantes da decisão econômica
    # =========================================================================
    fig1, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(horas, hist_soc_arr, label='SOC Real (MPC Econômico)', color='teal',
                marker='o', linestyle='-')
    axs[0].axhline(controlador_mpc_economico.soc_minimo, color='gray', linestyle=':', linewidth=1)
    axs[0].axhline(controlador_mpc_economico.soc_maximo, color='gray', linestyle=':', linewidth=1)
    axs[0].set_title('SOC Resultante da Decisão Econômica do MPC (sem setpoint de SOC)')
    axs[0].set_ylim(0, 1.0)
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    axs[1].plot(horas, hist_bat_arr, label='Potência Bateria Aplicada (W)', color='blue',
                marker='o', linestyle='None')
    axs[1].axhline(0, color='black', linewidth=1.0)
    axs[1].set_title('Sinal de Controle Calculado pelo MPC Econômico: Potência da Bateria')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    axs[2].plot(horas, preco_compra_perfil, label='Tarifa de Compra (R$/kWh)', color='crimson',
                marker='o', linestyle='None')
    axs[2].plot(horas, preco_venda_perfil, label='Tarifa de Venda (R$/kWh)', color='seagreen',
                marker='s', linestyle='None')
    axs[2].set_title('Tarifas (o controlador decide quando carregar/descarregar em função delas)')
    axs[2].set_xlabel('Horas da Simulacao (h)')
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()

    # =========================================================================
    # FIGURA 2: Balanço Energético da Microrrede em Malha Fechada
    # =========================================================================
    pv_geracao = hist_pv_arr
    bat_descarga = np.maximum(0, hist_bat_arr)
    grid_importacao = np.maximum(0, hist_grid_arr)

    carga_consumo = -carga_perfil
    bat_carga = np.minimum(0, hist_bat_arr)
    grid_exportacao = np.minimum(0, hist_grid_arr)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.bar(horas, pv_geracao, label='Geração PV', color='gold', edgecolor='black')
    ax2.bar(horas, bat_descarga, bottom=pv_geracao, label='Bateria (Descarga)', color='purple', edgecolor='black')
    ax2.bar(horas, grid_importacao, bottom=pv_geracao + bat_descarga, label='Rede (Importação)', color='gray', edgecolor='black')

    ax2.bar(horas, carga_consumo, label='Demanda (Carga)', color='red', edgecolor='black')
    ax2.bar(horas, bat_carga, bottom=carga_consumo, label='Bateria (Carga)', color='blue', edgecolor='black')
    base_exportacao = carga_consumo + bat_carga
    ax2.bar(horas, grid_exportacao, bottom=base_exportacao, label='Rede (Exportação)', color='green', edgecolor='black')

    ax2.axhline(0, color='black', linewidth=1.5)
    ax2.set_title('Balanço Energético da Microrrede (MPC Econômico em Malha Fechada)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Horas da Simulacao (h)')
    ax2.set_ylabel('Potência (W)')
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()

    # =========================================================================
    # FIGURA 3: Custo Econômico (Custo Instantâneo e Acumulado)
    # =========================================================================
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    cores_custo = np.where(hist_custo_instantaneo_arr >= 0, 'firebrick', 'forestgreen')
    axs3[0].bar(horas, hist_custo_instantaneo_arr, color=cores_custo, edgecolor='black')
    axs3[0].axhline(0, color='black', linewidth=1.0)
    axs3[0].set_title('Custo do Intervalo (R$) — vermelho: compra, verde: venda')
    axs3[0].grid(True, axis='y', linestyle=':', alpha=0.7)

    axs3[1].plot(horas, hist_custo_acumulado_arr, label='Custo Acumulado (R$)', color='navy',
                 marker='o', linestyle='-')
    axs3[1].axhline(0, color='black', linewidth=1.0)
    axs3[1].set_title('Custo Acumulado da Microrrede (objetivo minimizado pelo MPC)')
    axs3[1].set_xlabel('Horas da Simulacao (h)')
    axs3[1].legend()
    axs3[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Resumo econômico do dia
    print(f"\nCusto total do dia: R$ {microrrede.get_custo_acumulado_reais():.2f}")
    print(f"Energia comprada da rede: {microrrede.get_energia_comprada_kwh_acumulada():.2f} kWh")
    print(f"Energia vendida à rede:   {microrrede.get_energia_vendida_kwh_acumulada():.2f} kWh")
