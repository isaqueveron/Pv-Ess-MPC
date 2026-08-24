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
    # Instanciamento da planta (mesmos parâmetros de test.py)
    # ----------------------------------------------------------------- #
    painel = PainelFotovoltaicoVirtual(potencia_nominal_w=5000.0)
    bateria = BateriaVirtual(
        capacidade_maxima_wh=10000.0,
        eficiencia_carga=0.95,
        eficiencia_descarga=0.95,
        potencia_maxima_carga_w=3000.0,
        potencia_maxima_descarga_w=3000.0,
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
    # econômico, sem qualquer setpoint de SOC
    # ----------------------------------------------------------------- #
    # O horizonte de predição precisa ser longo o suficiente para o
    # controlador "enxergar" o pico tarifário da noite (18h-21h) já
    # quando decide carregar de madrugada -- com um horizonte curto
    # (ex.: 10h) o MPC gasta a bateria cedo demais, sem saber que o
    # preço vai subir mais à frente. Por isso aqui cobre-se o dia
    # inteiro (24h).
    HORIZONTE_PREDICAO = 24
    HORIZONTE_CONTROLE = 2
    PESO_LAMBDA = 1e-8   # regularização leve, só para suavizar Delta_Pbat

    controlador_mpc_economico = MPC_Economico(
        microrrede=microrrede,
        horizonte_predicao=HORIZONTE_PREDICAO,
        horizonte_controle=HORIZONTE_CONTROLE,
        peso_lambda=PESO_LAMBDA,
        potencia_bateria_min_w=-3000.0,
        potencia_bateria_max_w=3000.0,
        delta_potencia_bateria_max_w=1000.0,
        soc_minimo=0.1,
        soc_maximo=0.9,
    )

    # ----------------------------------------------------------------- #
    # Perfis sintéticos de 24h (perturbações e tarifas, iguais a test.py)
    # ----------------------------------------------------------------- #
    horas = np.arange(24)
    irradiancia_perfil = np.clip(800 * np.sin(np.pi * (horas - 6) / 12), 0, None)
    carga_perfil = 1500 + 500 * np.sin(np.pi * horas / 12)

    preco_compra_perfil = np.where(
        (horas >= 18) & (horas <= 21), 1.20,
        np.where((horas >= 0) & (horas <= 5), 0.45, 0.75)
    )
    preco_venda_perfil = preco_compra_perfil * 0.6

    def obter_previsao(perfil, hora_atual, horizonte):
        """
        Extrai a janela de previsão de comprimento 'horizonte' a partir da
        hora atual, assumindo que o perfil diário se repete (previsão
        perfeita) -- em uma aplicação real, viria de um módulo de
        forecast de irradiância/carga/tarifa.
        """
        indices = (hora_atual + np.arange(horizonte)) % len(perfil)
        return perfil[indices]

    # ----------------------------------------------------------------- #
    # Execução da simulação em malha fechada
    # ----------------------------------------------------------------- #
    hist_pv, hist_bat, hist_grid, hist_soc = [], [], [], []
    hist_custo_instantaneo, hist_custo_acumulado = [], []

    potencia_bateria_anterior = 0.0

    for h in horas:
        irradiancia_futura = obter_previsao(irradiancia_perfil, h, HORIZONTE_PREDICAO)
        carga_futura = obter_previsao(carga_perfil, h, HORIZONTE_PREDICAO)
        preco_compra_futuro = obter_previsao(preco_compra_perfil, h, HORIZONTE_PREDICAO)
        preco_venda_futuro = obter_previsao(preco_venda_perfil, h, HORIZONTE_PREDICAO)

        # 1) Controlador econômico: simula o próprio modelo da microrrede
        #    internamente e calcula a potência de referência que minimiza
        #    o custo previsto (sem nenhum setpoint de SOC)
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
    axs[2].set_xlabel('Horas do Dia (h)')
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
    ax2.set_xlabel('Horas do Dia (h)')
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
    axs3[1].set_xlabel('Horas do Dia (h)')
    axs3[1].legend()
    axs3[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Resumo econômico do dia
    print(f"\nCusto total do dia: R$ {microrrede.get_custo_acumulado_reais():.2f}")
    print(f"Energia comprada da rede: {microrrede.get_energia_comprada_kwh_acumulada():.2f} kWh")
    print(f"Energia vendida à rede:   {microrrede.get_energia_vendida_kwh_acumulada():.2f} kWh")