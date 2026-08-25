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
    A função objetivo é diretamente o CUSTO ECONÔMICO acumulado da
    microrrede ao longo do horizonte de predição:

        min_{Delta_Pbat}  sum_{j=1}^{N2} J_grid(k+j) + lambda * ||Delta_Pbat||^2

    onde J_grid(t) = Gamma_pur(t)*P_grid_compra(t) - Gamma_sale(t)*P_grid_venda(t)
    é exatamente o mesmo termo de custo já implementado em
    MicrorredeVirtual.calcular_custo_intervalo (Eq. de custo do controle
    terciário). O termo lambda*||Delta_Pbat||^2 é apenas uma pequena
    regularização para evitar chaveamentos bruscos/"bang-bang" de
    potência, e não representa nenhum objetivo de rastreamento.

    A predição continua sendo feita SIMULANDO diretamente o próprio
    modelo não-linear da planta: uma cópia da microrrede é rodada
    internamente, passo a passo, para cada candidata
    de sequência de incrementos de controle avaliada pelo otimizador
    SLSQP.

    TRATAMENTO DA RESTRIÇÃO DE SOC -- DOIS MODOS SELECIONÁVEIS:

    'penalidade': a violação de soc_minimo/soc_maximo entra como termo
        max(0, violação)^2 diretamente somado ao custo. NÃO há restrição
        de SOC no NLP (só as de potência), e NÃO há variáveis extras no
        vetor de decisão -- otimização mais rápida (vetor de tamanho
        Nu). A violação nunca é impedida de crescer além de um teto; ela
        só fica cada vez mais "cara".

    'folga': variável de folga (slack) não-negativa epsilon, uma por
        passo do horizonte de predição, é adicionada ao vetor de
        decisão. A restrição de SOC continua sendo uma restrição de
        verdade no NLP (soc_predito >= soc_minimo - epsilon e
        soc_predito <= soc_maximo + epsilon), mas SEMPRE FACTÍVEL, pois
        epsilon absorve a violação. epsilon é penalizada no custo E tem
        um teto rígido (folga_maxima_soc) -- formulação clássica de MPC
        com restrição soft via slack variable (Camacho & Bordons;
        material de Normey-Rico). Vetor de decisão maior (Nu + N2),
        portanto mais lenta, mas com controle explícito do tamanho
        máximo de violação permitido.
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
                 soc_maximo=1.0,
                 peso_penalidade_soc=1e6,
                 folga_maxima_soc=0.05,
                 modo_restricao_soc='penalidade'):
        """
        Args:
            microrrede (MicrorredeVirtual): instância do modelo completo
                da microrrede (painel + bateria + contabilidade
                econômica), usada como "gerador de predições". O
                controlador nunca modifica esta instância diretamente: a
                cada chamada de calcular_u uma CÓPIA é criada
                internamente (copy.deepcopy) para simular o futuro sem
                afetar o estado real da planta.
            horizonte_predicao (int): N2.
            horizonte_controle (int): Nu.
            peso_lambda (float): pondera (levemente) o esforço de
                controle -- apenas para suavizar a solução.
            potencia_bateria_min_w / potencia_bateria_max_w (float):
                limites físicos de potência do inversor/bateria [W].
            delta_potencia_bateria_max_w (float): variação máxima de
                potência da bateria permitida por passo [W].
            soc_minimo / soc_maximo (float): limites operacionais de
                SOC, usados apenas como RESTRIÇÕES, nunca como
                referência a ser seguida.
            peso_penalidade_soc (float): pondera o custo de violar a
                faixa operacional (em 'penalidade': peso sobre a
                violação bruta; em 'folga': peso sobre a variável de
                folga). Quanto maior, mais "cara" a violação.
            folga_maxima_soc (float): teto absoluto para a variável de
                folga. Usado apenas no modo 'folga'; ignorado em
                'penalidade'.
            modo_restricao_soc (str): 'folga' ou 'penalidade'.
                Ver docstring da classe para a diferença entre os dois.
        """
        if modo_restricao_soc not in ('folga', 'penalidade'):
            raise ValueError(
                "modo_restricao_soc deve ser 'folga' ou 'penalidade', "
                f"recebido: {modo_restricao_soc!r}")

        self._microrrede_referencia = microrrede
        self.horizonte_predicao = horizonte_predicao
        self.horizonte_controle = horizonte_controle
        self.peso_lambda = peso_lambda

        self.potencia_bateria_min_w = potencia_bateria_min_w
        self.potencia_bateria_max_w = potencia_bateria_max_w
        self.delta_potencia_bateria_max_w = delta_potencia_bateria_max_w
        self.soc_minimo = soc_minimo
        self.soc_maximo = soc_maximo
        self.peso_penalidade_soc = peso_penalidade_soc
        self.folga_maxima_soc = folga_maxima_soc
        self.modo_restricao_soc = modo_restricao_soc
        
        self._ultimo_delta_u_otimo = None

    # ------------------------------------------------------------------ #
    # Núcleo da predição: roda o próprio modelo da microrrede para frente
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
    # Dispatcher público: chama a implementação do modo escolhido
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
        ECONÔMICO previsto da microrrede ao longo do horizonte. O
        tratamento da restrição de SOC segue self.modo_restricao_soc.
        """
        if self.modo_restricao_soc == 'folga':
            return self._calcular_u_folga(
                potencia_bateria_anterior, irradiancia_futura_w_m2,
                carga_futura_w, preco_compra_futuro_reais_kwh,
                preco_venda_futuro_reais_kwh)
        else:
            return self._calcular_u_penalidade(
                potencia_bateria_anterior, irradiancia_futura_w_m2,
                carga_futura_w, preco_compra_futuro_reais_kwh,
                preco_venda_futuro_reais_kwh)
    
    def _gerar_pontos_iniciais(self, Nu):
        """
        Gera um conjunto mais rico de pontos de partida para o multi-
        start do SLSQP:

        - Vértices da caixa: zero, carga máxima, descarga máxima
          (mantidos -- ainda são bons candidatos "extremos").
        - Rampas (linspace): cobrem transições graduais entre carga e
          descarga, úteis quando o ótimo real é uma trajetória suave.
        - Alternado (bang-bang): cobre padrões de chaveamento rápido
          entre carga/descarga -- relevante porque a arbitragem
          tarifária tende a ter soluções ótimas no estilo bang-bang, e
          os 3 vértices sozinhos não representam esse padrão.
        - Warm-start: o delta_u ótimo da última chamada, deslocado em
          um passo (descarta o primeiro elemento, já aplicado, e repete
          o último para preencher) -- em MPC de horizonte deslizante,
          costuma ser o melhor chute de todos, pois o problema muda
          pouco de uma hora para a outra.
        """
        pontos = [
            np.zeros(Nu),
            np.full(Nu, -self.delta_potencia_bateria_max_w),
            np.full(Nu, self.delta_potencia_bateria_max_w),
            np.linspace(-self.delta_potencia_bateria_max_w,
                        self.delta_potencia_bateria_max_w, Nu),
            np.linspace(self.delta_potencia_bateria_max_w,
                        -self.delta_potencia_bateria_max_w, Nu),
        ]

        # Alternado bang-bang: +max, -max, +max, -max, ...
        sinais_alternados = np.array([1 if i % 2 == 0 else -1 for i in range(Nu)])
        pontos.append(sinais_alternados * self.delta_potencia_bateria_max_w)

        # Warm-start deslocado (só se já houver solução de uma chamada anterior)
        if self._ultimo_delta_u_otimo is not None and len(self._ultimo_delta_u_otimo) == Nu:
            deslocado = np.roll(self._ultimo_delta_u_otimo, -1)
            deslocado[-1] = self._ultimo_delta_u_otimo[-1]  # repete o último valor
            pontos.append(deslocado)

        return pontos

    # ------------------------------------------------------------------ #
    # MODO 'penalidade': sem restrição de SOC no NLP, sem variável extra
    # ------------------------------------------------------------------ #
    def _calcular_u_penalidade(self,
                                potencia_bateria_anterior,
                                irradiancia_futura_w_m2,
                                carga_futura_w,
                                preco_compra_futuro_reais_kwh,
                                preco_venda_futuro_reais_kwh):
        Nu = self.horizonte_controle

        def funcao_custo(delta_u):
            custo_previsto, soc_predito, _ = self.simular_trajetoria(
                potencia_bateria_anterior, delta_u,
                irradiancia_futura_w_m2, carga_futura_w,
                preco_compra_futuro_reais_kwh, preco_venda_futuro_reais_kwh)
            termo_esforco = self.peso_lambda * np.sum(delta_u ** 2)

            violacao_inferior = np.maximum(0.0, self.soc_minimo - soc_predito)
            violacao_superior = np.maximum(0.0, soc_predito - self.soc_maximo)
            termo_penalidade_soc = self.peso_penalidade_soc * np.sum(
                violacao_inferior ** 2 + violacao_superior ** 2)

            return custo_previsto + termo_esforco + termo_penalidade_soc

        bounds_delta_u = [(-self.delta_potencia_bateria_max_w,
                            self.delta_potencia_bateria_max_w) for _ in range(Nu)]

        matriz_soma_cumulativa = np.tril(np.ones((Nu, Nu)))

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
        ]

        pontos_iniciais = self._gerar_pontos_iniciais(Nu)

        melhor_delta_u = None
        melhor_custo = np.inf

        for delta_u_inicial in pontos_iniciais:
            resultado = minimize(
                funcao_custo,
                delta_u_inicial,
                method='SLSQP',
                bounds=bounds_delta_u,
                constraints=restricoes,
                options={'ftol': 1e-9, 'maxiter': 200}
            )
            if resultado.fun < melhor_custo:
                melhor_custo = resultado.fun
                melhor_delta_u = resultado.x

        if melhor_delta_u is not None:
            self._ultimo_delta_u_otimo = melhor_delta_u.copy()

        if melhor_delta_u is None:
            delta_u_efetivo = -self.delta_potencia_bateria_max_w
        else:
            delta_u_efetivo = melhor_delta_u[0]

        potencia_bateria_aplicada = potencia_bateria_anterior + delta_u_efetivo
        potencia_bateria_aplicada = np.clip(potencia_bateria_aplicada,
                                             self.potencia_bateria_min_w,
                                             self.potencia_bateria_max_w)
        return potencia_bateria_aplicada

    # ------------------------------------------------------------------ #
    # MODO 'folga': variável de folga explícita, restrição sempre factível
    # ------------------------------------------------------------------ #
    def _calcular_u_folga(self,
                           potencia_bateria_anterior,
                           irradiancia_futura_w_m2,
                           carga_futura_w,
                           preco_compra_futuro_reais_kwh,
                           preco_venda_futuro_reais_kwh):
        Nu = self.horizonte_controle
        N2 = self.horizonte_predicao

        n_delta_u = Nu
        n_epsilon = N2

        def desempacotar(x):
            delta_u = x[:n_delta_u]
            epsilon = x[n_delta_u:]
            return delta_u, epsilon

        # Cache de um único ponto: evita rodar simular_trajetoria duas
        # vezes (custo + restrição) para o mesmo x, o que dobra o custo
        # computacional sem necessidade -- especialmente relevante aqui
        # pois o SLSQP avalia ambas repetidamente no mesmo ponto durante
        # o cálculo numérico do gradiente/Jacobiana.
        cache = {'x': None, 'custo': None, 'soc_predito': None}

        def simular_com_cache(delta_u, x_chave):
            if cache['x'] is None or not np.array_equal(cache['x'], x_chave):
                custo_previsto, soc_predito, _ = self.simular_trajetoria(
                    potencia_bateria_anterior, delta_u,
                    irradiancia_futura_w_m2, carga_futura_w,
                    preco_compra_futuro_reais_kwh, preco_venda_futuro_reais_kwh)
                cache['x'] = x_chave.copy()
                cache['custo'] = custo_previsto
                cache['soc_predito'] = soc_predito
            return cache['custo'], cache['soc_predito']

        def funcao_custo(x):
            delta_u, epsilon = desempacotar(x)
            custo_previsto, _ = simular_com_cache(delta_u, x)
            termo_esforco = self.peso_lambda * np.sum(delta_u ** 2)
            termo_folga = self.peso_penalidade_soc * np.sum(epsilon ** 2)
            return custo_previsto + termo_esforco + termo_folga

        bounds_delta_u = [(-self.delta_potencia_bateria_max_w,
                            self.delta_potencia_bateria_max_w) for _ in range(n_delta_u)]
        bounds_epsilon = [(0.0, self.folga_maxima_soc) for _ in range(n_epsilon)]
        bounds_totais = bounds_delta_u + bounds_epsilon

        matriz_soma_cumulativa = np.tril(np.ones((Nu, Nu)))

        def restricao_potencia_max(x):
            delta_u, _ = desempacotar(x)
            return self.potencia_bateria_max_w - \
                (potencia_bateria_anterior + np.dot(matriz_soma_cumulativa, delta_u))

        def restricao_potencia_min(x):
            delta_u, _ = desempacotar(x)
            return (potencia_bateria_anterior +
                    np.dot(matriz_soma_cumulativa, delta_u)) - self.potencia_bateria_min_w

        def restricao_soc_com_folga(x):
            delta_u, epsilon = desempacotar(x)
            _, soc_predito = simular_com_cache(delta_u, x)
            folga_inferior = soc_predito - (self.soc_minimo - epsilon)
            folga_superior = (self.soc_maximo + epsilon) - soc_predito
            return np.concatenate([folga_inferior, folga_superior])

        restricoes = [
            {'type': 'ineq', 'fun': restricao_potencia_max},
            {'type': 'ineq', 'fun': restricao_potencia_min},
            {'type': 'ineq', 'fun': restricao_soc_com_folga},
        ]

        pontos_delta_u = self._gerar_pontos_iniciais(n_delta_u)
        pontos_iniciais = [np.concatenate([p, np.zeros(n_epsilon)]) for p in pontos_delta_u]
        
        melhor_x = None
        melhor_custo = np.inf

        for x_inicial in pontos_iniciais:
            resultado = minimize(
                funcao_custo,
                x_inicial,
                method='SLSQP',
                bounds=bounds_totais,
                constraints=restricoes,
                options={'ftol': 1e-9, 'maxiter': 200}
            )
            if resultado.fun < melhor_custo:
                melhor_custo = resultado.fun
                melhor_x = resultado.x

        if melhor_x is not None:
            delta_u_para_warm_start, _ = desempacotar(melhor_x)
            self._ultimo_delta_u_otimo = delta_u_para_warm_start.copy()

        if melhor_x is None:
            delta_u_efetivo = -self.delta_potencia_bateria_max_w
            print("[MPC] Fallback do otimizador")
        else:
            delta_u_otimo, epsilon_opt = desempacotar(melhor_x)
            delta_u_efetivo = delta_u_otimo[0]

            violacao_maxima = epsilon_opt.max()
            if violacao_maxima > 1e-4:
                print(f"[MPC] folga de SOC utilizada: {violacao_maxima:.4f} "
                      f"(limite de folga: {self.folga_maxima_soc})")

        potencia_bateria_aplicada = potencia_bateria_anterior + delta_u_efetivo
        potencia_bateria_aplicada = np.clip(potencia_bateria_aplicada,
                                             self.potencia_bateria_min_w,
                                             self.potencia_bateria_max_w)
        return potencia_bateria_aplicada
