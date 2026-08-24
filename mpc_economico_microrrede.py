import copy

import numpy as np
from scipy.optimize import minimize

from modelo_microrrede_pv_ess import MicrorredeVirtual


# --------------------------------------------------------------------------- #
# MPC ECONÔMICO -- MINIMIZA CUSTO, NÃO RASTREIA REFERÊNCIA DE SOC
# --------------------------------------------------------------------------- #
class MPC_Economico:
    """
    Controlador Preditivo Econômico (Economic MPC) para o subsistema
    ESS+PV da microrrede.

    Diferença central em relação à versão anterior (MPC_ModeloNaoLinear):
    aquele controlador definia a função de custo como o ERRO DE
    RASTREAMENTO entre o SOC previsto e um setpoint de SOC externo. Este
    controlador NÃO possui setpoint de SOC algum -- a função objetivo é
    diretamente o CUSTO ECONÔMICO acumulado da microrrede ao longo do
    horizonte de predição:

        min_{Delta_Pbat}  sum_{j=1}^{N2} J_grid(k+j) + lambda * ||Delta_Pbat||^2

    onde J_grid(t) = Gamma_pur(t)*P_grid_compra(t) - Gamma_sale(t)*P_grid_venda(t)
    é exatamente o mesmo termo de custo já implementado em
    MicrorredeVirtual.calcular_custo_intervalo (Eq. de custo do controle
    terciário). O termo lambda*||Delta_Pbat||^2 é apenas uma pequena
    regularização para evitar chaveamentos bruscos/"bang-bang" de
    potência, e não representa nenhum objetivo de rastreamento.

    A predição continua sendo feita SIMULANDO diretamente o próprio
    modelo não-linear da planta (agora a MicrorredeVirtual completa --
    painel + bateria + contabilidade de custo --, e não apenas a
    bateria), da mesma forma que no MPC_ModeloNaoLinear: uma cópia da
    microrrede é rodada internamente, passo a passo, para cada candidata
    de sequência de incrementos de controle avaliada pelo otimizador
    SLSQP.

    Como o objetivo agora depende de previsões de irradiância, carga e
    tarifas ao longo do horizonte (e não apenas do estado interno da
    bateria), calcular_u passa a exigir essas previsões como argumento --
    em uma aplicação real, viriam de um módulo de previsão (forecast);
    aqui, no teste em malha fechada, assume-se previsão perfeita a partir
    dos próprios perfis sintéticos do dia.
    """

    def __init__(self,
                 microrrede: MicrorredeVirtual,
                 horizonte_predicao,
                 horizonte_controle,
                 peso_lambda,
                 potencia_bateria_min_w,
                 potencia_bateria_max_w,
                 delta_potencia_bateria_max_w,
                 soc_minimo=0.0,
                 soc_maximo=1.0):
        """
        Args:
            microrrede (MicrorredeVirtual): instância do modelo completo
                da microrrede (painel + bateria + contabilidade
                econômica), usada como "gerador de predições". Assim como
                no MPC_ModeloNaoLinear, o controlador nunca modifica esta
                instância diretamente: a cada chamada de calcular_u uma
                CÓPIA é criada internamente (copy.deepcopy) para simular
                o futuro sem afetar o estado real da planta nem a
                contabilidade de custo acumulado real.
            horizonte_predicao (int): N2.
            horizonte_controle (int): Nu.
            peso_lambda (float): pondera (levemente) o esforço de
                controle -- apenas para suavizar a solução, não é um
                objetivo de rastreamento.
            potencia_bateria_min_w / potencia_bateria_max_w (float):
                limites físicos de potência do inversor/bateria [W].
            delta_potencia_bateria_max_w (float): variação máxima de
                potência da bateria permitida por passo [W].
            soc_minimo / soc_maximo (float): limites operacionais de SOC,
                usados apenas como RESTRIÇÕES (para proteger a bateria),
                nunca como referência a ser seguida.
        """
        self._microrrede_referencia = microrrede
        self.horizonte_predicao = horizonte_predicao
        self.horizonte_controle = horizonte_controle
        self.peso_lambda = peso_lambda

        self.potencia_bateria_min_w = potencia_bateria_min_w
        self.potencia_bateria_max_w = potencia_bateria_max_w
        self.delta_potencia_bateria_max_w = delta_potencia_bateria_max_w
        self.soc_minimo = soc_minimo
        self.soc_maximo = soc_maximo

    # ------------------------------------------------------------------ #
    # Núcleo da predição: roda o próprio modelo da microrrede "para frente"
    # ------------------------------------------------------------------ #
    def simular_trajetoria(self,
                            potencia_bateria_anterior,
                            delta_u,
                            irradiancia_futura_w_m2,
                            carga_futura_w,
                            preco_compra_futuro_reais_kwh,
                            preco_venda_futuro_reais_kwh):
        """
        Simula o modelo NÃO-LINEAR completo da microrrede (painel + bateria
        + custo) ao longo do horizonte de predição, para uma dada
        sequência de incrementos de controle Delta_Pbat e um conjunto de
        previsões de perturbação/tarifa, retornando o custo total previsto
        e as trajetórias de SOC e potência de bateria (para uso nas
        restrições).

        A microrrede de referência é copiada com seu estado atual (SOC
        real da bateria já embutido nela), garantindo que a simulação
        parta exatamente da condição presente da planta real.
        """
        N2, Nu = self.horizonte_predicao, self.horizonte_controle

        microrrede_simulada = copy.deepcopy(self._microrrede_referencia)

        matriz_soma_cumulativa = np.tril(np.ones((Nu, Nu)))
        potencia_planejada_ate_nu = potencia_bateria_anterior + \
            np.dot(matriz_soma_cumulativa, delta_u)

        custo_total_previsto = 0.0
        soc_predito = np.zeros(N2)
        potencia_predita = np.zeros(N2)

        for m in range(N2):
            potencia_referencia_m = potencia_planejada_ate_nu[m] if m < Nu \
                else potencia_planejada_ate_nu[-1]

            resultado = microrrede_simulada.executar_passo_simulacao(
                irradiancia_w_m2=irradiancia_futura_w_m2[m],
                potencia_carga_consumidora_w=carga_futura_w[m],
                potencia_bateria_referencia_w=potencia_referencia_m,
                preco_compra_reais_kwh=preco_compra_futuro_reais_kwh[m],
                preco_venda_reais_kwh=preco_venda_futuro_reais_kwh[m],
            )

            custo_total_previsto += resultado['custo_instantaneo_reais']
            soc_predito[m] = resultado['soc_bateria']
            potencia_predita[m] = resultado['potencia_bateria_w']

        return custo_total_previsto, soc_predito, potencia_predita

    # ------------------------------------------------------------------ #
    # Otimização restrita (mesma estrutura das versões anteriores)
    # ------------------------------------------------------------------ #
    def calcular_u(self,
                    potencia_bateria_anterior,
                    irradiancia_futura_w_m2,
                    carga_futura_w,
                    preco_compra_futuro_reais_kwh,
                    preco_venda_futuro_reais_kwh):
        """
        Resolve o problema de otimização restrita e retorna a potência de
        bateria a ser aplicada no próximo passo, minimizando o CUSTO
        ECONÔMICO previsto da microrrede ao longo do horizonte (não há
        setpoint de SOC nesta função).

        Args:
            potencia_bateria_anterior (float): última potência de bateria
                efetivamente aplicada [W].
            irradiancia_futura_w_m2, carga_futura_w,
            preco_compra_futuro_reais_kwh, preco_venda_futuro_reais_kwh:
                arrays de comprimento horizonte_predicao com as previsões
                de irradiância, carga e tarifas para os próximos passos.
        """
        Nu = self.horizonte_controle

        def funcao_custo(delta_u):
            custo_previsto, _, _ = self.simular_trajetoria(
                potencia_bateria_anterior, delta_u,
                irradiancia_futura_w_m2, carga_futura_w,
                preco_compra_futuro_reais_kwh, preco_venda_futuro_reais_kwh)
            termo_esforco = self.peso_lambda * np.sum(delta_u ** 2)
            return custo_previsto + termo_esforco

        bounds_delta_u = [(-self.delta_potencia_bateria_max_w,
                            self.delta_potencia_bateria_max_w) for _ in range(Nu)]

        matriz_soma_cumulativa = np.tril(np.ones((Nu, Nu)))

        def restricao_soc(delta_u):
            _, soc_predito, _ = self.simular_trajetoria(
                potencia_bateria_anterior, delta_u,
                irradiancia_futura_w_m2, carga_futura_w,
                preco_compra_futuro_reais_kwh, preco_venda_futuro_reais_kwh)
            folga_superior = self.soc_maximo - soc_predito
            folga_inferior = soc_predito - self.soc_minimo
            return np.concatenate([folga_superior, folga_inferior])

        restricoes = [
            {
                'type': 'ineq',
                'fun': lambda delta_u: self.potencia_bateria_max_w -
                       (potencia_bateria_anterior + np.dot(matriz_soma_cumulativa, delta_u))
            },
            {
                'type': 'ineq',
                'fun': lambda delta_u: (potencia_bateria_anterior +
                       np.dot(matriz_soma_cumulativa, delta_u)) - self.potencia_bateria_min_w
            },
            {
                'type': 'ineq',
                'fun': restricao_soc
            },
        ]

        # ------------------------------------------------------------- #
        # Otimização multi-start: como a função de custo simula um
        # modelo NÃO-LINEAR (com saturações de potência e de SOC), a
        # superfície de custo não é suave, e o problema deixa de ser
        # convexo. O SLSQP resolve isso localmente a partir de um único
        # ponto de partida -- e, dependendo de detalhes numéricos da
        # instalação do scipy/BLAS (diferenças de arredondamento na
        # diferenciação por diferenças finitas usada internamente pelo
        # SLSQP), pode convergir para ótimos locais DIFERENTES em
        # máquinas diferentes, mesmo com o mesmo código.
        #
        # Para tornar o resultado robusto a isso, a otimização é
        # repetida a partir de alguns pontos de partida distintos --
        # zero, potência máxima de carga e potência máxima de descarga
        # -- e fica-se com o menor custo obtido entre eles.
        # ------------------------------------------------------------- #
        pontos_iniciais = [
            np.zeros(Nu),
            np.full(Nu, -self.delta_potencia_bateria_max_w),
            np.full(Nu, self.delta_potencia_bateria_max_w),
        ]

        melhor_delta_u = np.zeros(Nu)
        melhor_custo = np.inf

        for delta_u_inicial in pontos_iniciais:
            resultado = minimize(
                funcao_custo,
                delta_u_inicial,
                method='SLSQP',
                bounds=bounds_delta_u,
                constraints=restricoes,
                options={
                    'ftol': 1e-9,
                    'maxiter': 80
                }
            )
            if resultado.success and resultado.fun < melhor_custo:
                melhor_custo = resultado.fun
                melhor_delta_u = resultado.x

        delta_u_efetivo = melhor_delta_u[0]
        potencia_bateria_aplicada = potencia_bateria_anterior + delta_u_efetivo

        # Saturação de segurança (fallback; o próprio modelo físico
        # também satura ao ser executado de fato pela planta real)
        potencia_bateria_aplicada = np.clip(potencia_bateria_aplicada,
                                             self.potencia_bateria_min_w,
                                             self.potencia_bateria_max_w)

        return potencia_bateria_aplicada