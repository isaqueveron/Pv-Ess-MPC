import numpy as np


class BateriaVirtual:
    """
    Modelo virtual do Sistema de Armazenamento de Energia (ESS) por baterias.

    Implementa a dinâmica do Estado de Carga (SOC) descrita pela Eq. (5.1) de
    [14], considerando eficiências distintas para carga e descarga. A potência
    de barramento da bateria (Pbat) é decomposta em sua parcela de carga
    (Pbat,ch) e de descarga (Pbat,dis) através das relações lógicas P6/P7
    (Tabela 5.2), aqui resolvidas de forma direta (determinística), já que em
    malha aberta a potência de referência da bateria é uma entrada conhecida,
    e não uma variável de decisão do otimizador.
    """

    def __init__(self,
                 capacidade_maxima_wh,
                 eficiencia_carga,
                 eficiencia_descarga,
                 potencia_maxima_carga_w,
                 potencia_maxima_descarga_w,
                 soc_minimo=0.0,
                 soc_maximo=1.0,
                 soc_inicial=0.5,
                 passo_tempo_segundos=60.0):
        """
        Inicializa o modelo da bateria.

        Args:
            capacidade_maxima_wh (float): Capacidade máxima de energia armazenável (Cmax_bat) [Wh].
            eficiencia_carga (float): Eficiência de carga (eta_bat,ch), entre 0 e 1.
            eficiencia_descarga (float): Eficiência de descarga (eta_bat,dis), entre 0 e 1.
            potencia_maxima_carga_w (float): Potência máxima de carga (Pmin_bat, valor negativo) [W].
            potencia_maxima_descarga_w (float): Potência máxima de descarga (Pmax_bat, valor positivo) [W].
            soc_minimo (float): Limite inferior de SOC permitido.
            soc_maximo (float): Limite superior de SOC permitido.
            soc_inicial (float): SOC inicial da bateria.
            passo_tempo_segundos (float): Período de amostragem Ts [s].
        """

        self._capacidade_maxima_wh = capacidade_maxima_wh
        self._eficiencia_carga = eficiencia_carga
        self._eficiencia_descarga = eficiencia_descarga

        # Convenção de sinais: carga (absorve potência da rede) é negativa,
        # descarga (fornece potência à rede) é positiva.
        self._potencia_minima_bat_w = -abs(potencia_maxima_carga_w)
        self._potencia_maxima_bat_w = abs(potencia_maxima_descarga_w)

        self._soc_minimo = soc_minimo
        self._soc_maximo = soc_maximo
        self._passo_tempo_segundos = passo_tempo_segundos

        self._soc = soc_inicial

        self._potencia_bateria_w = 0.0
        self._potencia_carga_w = 0.0
        self._potencia_descarga_w = 0.0

        self._delta_carga = 0
        self._delta_descarga = 0

        self._flag_saturacao_potencia = False
        self._flag_saturacao_soc = False

    # ------------------------------------------------------------------ #
    # Getters
    # ------------------------------------------------------------------ #
    def get_soc(self):                          return self._soc
    def get_potencia_bateria_w(self):            return self._potencia_bateria_w
    def get_potencia_carga_w(self):              return self._potencia_carga_w
    def get_potencia_descarga_w(self):           return self._potencia_descarga_w
    def get_delta_carga(self):                   return self._delta_carga
    def get_delta_descarga(self):                return self._delta_descarga
    def get_flag_saturacao_potencia(self):       return self._flag_saturacao_potencia
    def get_flag_saturacao_soc(self):            return self._flag_saturacao_soc

    # ------------------------------------------------------------------ #
    # Núcleo do modelo
    # ------------------------------------------------------------------ #
    def separar_potencia_carga_descarga(self, potencia_bateria_referencia_w):
        """
        Decompõe a potência de referência da bateria em suas parcelas de
        carga e descarga, equivalente às relações lógicas das Eqs. (5.2) a
        (5.11):

            Pbat(t) <= 0  <=>  delta_ch(t) = 1   (Eq. 5.2)
            Pbat,ch(t)  = -Pbat(t) * delta_ch(t)  (Eq. 5.5, produto misto P7)
            Pbat,dis(t) =  Pbat(t) * delta_dis(t) (equivalente por Eq. 5.6)
            delta_ch(t) + delta_dis(t) = 1        (Eq. 5.7)

        Em malha aberta essas relações não precisam ser tratadas como
        restrições MILP: como Pbat já é conhecido, delta_ch e delta_dis saem
        diretamente do sinal de Pbat.
        """

        if potencia_bateria_referencia_w <= 0.0:
            delta_carga = 1
            delta_descarga = 0
        else:
            delta_carga = 0
            delta_descarga = 1

        potencia_carga_w = -potencia_bateria_referencia_w * delta_carga
        potencia_descarga_w = potencia_bateria_referencia_w * delta_descarga

        return potencia_carga_w, potencia_descarga_w, delta_carga, delta_descarga

    def atualizar_soc(self, potencia_carga_w, potencia_descarga_w):
        """
        Atualiza o SOC de acordo com a Eq. (5.1):

            SOC(t+1) = SOC(t) + eta_ch * Pch(t) * Ts / Cmax
                               + Pdis(t) * Ts / (eta_dis * Cmax)

        Notar que Pdis(t) já é tratada aqui como grandeza positiva sendo
        retirada do sistema, portanto entra com sinal negativo na equação de
        estado.
        """

        passo_horas = self._passo_tempo_segundos / 3600.0

        termo_carga = (self._eficiencia_carga * potencia_carga_w * passo_horas) / self._capacidade_maxima_wh
        termo_descarga = (potencia_descarga_w * passo_horas) / (self._eficiencia_descarga * self._capacidade_maxima_wh)

        novo_soc = self._soc + termo_carga - termo_descarga

        self._flag_saturacao_soc = not (self._soc_minimo <= novo_soc <= self._soc_maximo)
        novo_soc_saturado = np.clip(novo_soc, self._soc_minimo, self._soc_maximo)

        return novo_soc_saturado

    def executar_passo_simulacao(self, potencia_bateria_referencia_w):
        """
        Executa um passo de simulação em malha aberta limitando a potência
        tanto pelos limites físicos do inversor quanto pela capacidade 
        (SOC) remanescente na bateria.
        """

        # 1. Saturação pelos limites físicos de hardware (inversor/bateria)
        potencia_saturada_w = np.clip(potencia_bateria_referencia_w,
                                       self._potencia_minima_bat_w,
                                       self._potencia_maxima_bat_w)
                                       
        passo_horas = self._passo_tempo_segundos / 3600.0

        # 2. Saturação de potência dinâmica (pelos limites de SOC)
        if potencia_saturada_w > 0:
            # Descarga: garante que não retira mais energia do que o SOC mínimo permite
            potencia_max_descarga_soc = ((self._soc - self._soc_minimo) * 
                                         self._capacidade_maxima_wh * 
                                         self._eficiencia_descarga) / passo_horas
            # Pega o menor valor entre a potência solicitada e a permitida pelo SOC
            potencia_real_w = min(potencia_saturada_w, potencia_max_descarga_soc)
            
        elif potencia_saturada_w < 0:
            # Carga: garante que não injeta mais energia do que o SOC máximo permite
            potencia_max_carga_soc_mag = ((self._soc_maximo - self._soc) * 
                                          self._capacidade_maxima_wh) / (self._eficiencia_carga * passo_horas)
            # Como a carga é negativa, pegamos o maior valor (mais próximo de zero)
            potencia_real_w = max(potencia_saturada_w, -potencia_max_carga_soc_mag)
            
        else:
            potencia_real_w = 0.0

        # Atualiza as flags (se cortou por hardware OU por SOC, conta como saturação)
        self._flag_saturacao_potencia = potencia_real_w != potencia_bateria_referencia_w

        # 3. Decompõe a potência limitadada e atualiza os estados do modelo
        potencia_carga_w, potencia_descarga_w, delta_carga, delta_descarga = \
            self.separar_potencia_carga_descarga(potencia_real_w)

        self._soc = self.atualizar_soc(potencia_carga_w, potencia_descarga_w)

        # 4. Registra os valores reais que efetivamente fluíram pelo barramento
        self._potencia_bateria_w = potencia_real_w
        self._potencia_carga_w = potencia_carga_w
        self._potencia_descarga_w = potencia_descarga_w
        self._delta_carga = delta_carga
        self._delta_descarga = delta_descarga


class PainelFotovoltaicoVirtual:
    """
    Modelo virtual simplificado do array fotovoltaico, utilizado no bloco de
    "Plant Model" do controle terciário para estimar a potência gerada
    (Ppv) a partir da irradiância prevista (Gamb).
    """

    def __init__(self,
                 potencia_nominal_w,
                 irradiancia_referencia_w_m2=1000.0):
        """
        Args:
            potencia_nominal_w (float): Potência nominal do array em STC [W].
            irradiancia_referencia_w_m2 (float): Irradiância de referência (STC) [W/m2].
        """

        self._potencia_nominal_w = potencia_nominal_w
        self._irradiancia_referencia_w_m2 = irradiancia_referencia_w_m2

        self._potencia_gerada_w = 0.0

    def get_potencia_gerada_w(self):             return self._potencia_gerada_w

    def executar_passo_simulacao(self, irradiancia_w_m2):
        """
        Calcula a potência fotovoltaica gerada (Ppv) a partir da irradiância
        solar (Gamb).
        """

        irradiancia_w_m2 = max(0.0, irradiancia_w_m2)

        potencia_bruta_w = self._potencia_nominal_w * (irradiancia_w_m2 / self._irradiancia_referencia_w_m2)

        self._potencia_gerada_w = max(0.0, potencia_bruta_w)

        return self._potencia_gerada_w


class MicrorredeVirtual:
    """
    Modelo de planta (Plant Model) da microrrede simplificada, contendo
    apenas geração fotovoltaica e armazenamento por bateria (sem hidrogênio
    e sem geração eólica).

    Por enquanto o sistema é simulado em MALHA ABERTA: a potência de
    despacho da bateria (Psch_bat) é fornecida externamente (perfil de teste
    ou cronograma pré-definido), e não calculada por um otimizador MPC. A
    camada de decisão (MPC Híbrido / MLD) poderá ser conectada
    posteriormente, substituindo a fonte do sinal potencia_bateria_referencia_w
    pela saída do controlador terciário.
    """

    def __init__(self,
                 painel_fotovoltaico: PainelFotovoltaicoVirtual,
                 bateria: BateriaVirtual,
                 passo_tempo_segundos=60.0):

        self._painel_fotovoltaico = painel_fotovoltaico
        self._bateria = bateria
        self._passo_tempo_segundos = passo_tempo_segundos

        self._potencia_carga_consumidora_w = 0.0
        self._potencia_rede_w = 0.0

        # --- Contabilidade econômica (Gamma_pur / Gamma_sale) ---
        self._custo_instantaneo_reais = 0.0
        self._custo_acumulado_reais = 0.0
        self._energia_comprada_kwh_acumulada = 0.0
        self._energia_vendida_kwh_acumulada = 0.0

    def get_potencia_rede_w(self):              return self._potencia_rede_w
    def get_custo_instantaneo_reais(self):      return self._custo_instantaneo_reais
    def get_custo_acumulado_reais(self):        return self._custo_acumulado_reais
    def get_energia_comprada_kwh_acumulada(self): return self._energia_comprada_kwh_acumulada
    def get_energia_vendida_kwh_acumulada(self):  return self._energia_vendida_kwh_acumulada

    def calcular_custo_intervalo(self, potencia_rede_w, preco_compra_reais_kwh, preco_venda_reais_kwh):
        """
        Calcula o custo econômico do intervalo de amostragem, equivalente ao
        termo de custo da rede na função objetivo do controle terciário:

            J_grid(t) = Gamma_pur(t) * P_grid_compra(t) - Gamma_sale(t) * P_grid_venda(t)

        Convenção de sinal (igual ao balanço de potência):
            P_grid(t) >= 0  -> a microrrede está IMPORTANDO energia da rede
                               (comprando), custo positivo (Gamma_pur).
            P_grid(t) <  0  -> a microrrede está EXPORTANDO energia à rede
                               (vendendo), custo negativo, ou seja, receita
                               (Gamma_sale).
        """

        passo_horas = self._passo_tempo_segundos / 3600.0
        energia_rede_kwh = (potencia_rede_w * passo_horas) / 1000.0

        if potencia_rede_w >= 0.0:
            custo_reais = energia_rede_kwh * preco_compra_reais_kwh
            self._energia_comprada_kwh_acumulada += energia_rede_kwh
        else:
            # energia_rede_kwh já é negativa aqui; o custo resultante também
            # é negativo (representa receita da venda de energia).
            custo_reais = energia_rede_kwh * preco_venda_reais_kwh
            self._energia_vendida_kwh_acumulada += -energia_rede_kwh

        return custo_reais

    def executar_passo_simulacao(self,
                                  irradiancia_w_m2,
                                  potencia_carga_consumidora_w,
                                  potencia_bateria_referencia_w,
                                  preco_compra_reais_kwh=0.0,
                                  preco_venda_reais_kwh=0.0):
        """
        Executa um passo de simulação da microrrede em malha aberta.

        Balanço de potência instantâneo da microrrede:

            P_grid(t) = P_load(t) - P_pv(t) - P_bat(t)

        Convenção de sinais (igual à usada em BateriaVirtual):
            P_bat(t) > 0 (descarga) -> a bateria FORNECE potência à
                microrrede, assim como o PV, e por isso entra SUBTRAÍDA
                no balanço (reduz a necessidade de importar da rede).
            P_bat(t) < 0 (carga) -> a bateria CONSOME potência da
                microrrede, assim como a carga, e por isso entra SOMADA
                no balanço (aumenta a necessidade de importar da rede).

            P_grid(t) > 0 -> a microrrede está IMPORTANDO energia da rede.
            P_grid(t) < 0 -> a microrrede está EXPORTANDO energia à rede.

        Args:
            irradiancia_w_m2 (float): Irradiância solar medida/prevista Gamb [W/m2].
            potencia_carga_consumidora_w (float): Potência de consumo (carga) da microrrede [W].
            potencia_bateria_referencia_w (float): Potência de referência da bateria vinda do
                cronograma Psch_bat(t) (em malha aberta, definida externamente) [W].
            preco_compra_reais_kwh (float): Tarifa de compra de energia da rede no instante
                atual, equivalente a Gamma_pur(t) [R$/kWh].
            preco_venda_reais_kwh (float): Tarifa de venda de energia à rede no instante
                atual, equivalente a Gamma_sale(t) [R$/kWh].
        """

        potencia_pv_w = self._painel_fotovoltaico.executar_passo_simulacao(irradiancia_w_m2)

        self._bateria.executar_passo_simulacao(potencia_bateria_referencia_w)
        potencia_bateria_real_w = self._bateria.get_potencia_bateria_w()

        self._potencia_carga_consumidora_w = potencia_carga_consumidora_w
        self._potencia_rede_w = potencia_carga_consumidora_w - potencia_pv_w - potencia_bateria_real_w

        self._custo_instantaneo_reais = self.calcular_custo_intervalo(
            self._potencia_rede_w, preco_compra_reais_kwh, preco_venda_reais_kwh)
        self._custo_acumulado_reais += self._custo_instantaneo_reais

        return {
            "potencia_pv_w": potencia_pv_w,
            "potencia_bateria_w": potencia_bateria_real_w,
            "potencia_rede_w": self._potencia_rede_w,
            "soc_bateria": self._bateria.get_soc(),
            "custo_instantaneo_reais": self._custo_instantaneo_reais,
            "custo_acumulado_reais": self._custo_acumulado_reais,
        }


# --------------------------------------------------------------------------- #
# EXEMPLO DE USO EM MALHA ABERTA
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    painel = PainelFotovoltaicoVirtual(potencia_nominal_w=5000.0)
    bateria = BateriaVirtual(
        capacidade_maxima_wh=10000.0,
        eficiencia_carga=0.95,
        eficiencia_descarga=0.95,
        potencia_maxima_carga_w=3000.0,
        potencia_maxima_descarga_w=3000.0,
        soc_inicial=0.5,
        passo_tempo_segundos=3600.0,   # Ts = 1h, típico de um horizonte de agendamento
    )

    microrrede = MicrorredeVirtual(painel, bateria, passo_tempo_segundos=3600.0)

    # Perfil sintético de 24h (apenas ilustrativo, sem otimizador ainda)
    horas = np.arange(24)
    irradiancia_perfil = np.clip(800 * np.sin(np.pi * (horas - 6) / 12), 0, None)
    carga_perfil = 1500 + 500 * np.sin(np.pi * horas / 12)

    # Cronograma de referência da bateria (Psch_bat) definido manualmente por
    # enquanto: carrega durante o pico solar, descarrega à noite.
    referencia_bateria_perfil = np.where(
        (horas >= 10) & (horas <= 15), -2000.0,
        np.where((horas >= 18) | (horas <= 5), 1500.0, 0.0)
    )

    # Perfil simplificado de tarifa (ex.: tarifa branca): mais cara no
    # horário de ponta (18h-21h), intermediária durante o dia, mais barata
    # de madrugada. A tarifa de venda é tipicamente uma fração da de compra.
    preco_compra_perfil = np.where(
        (horas >= 18) & (horas <= 21), 1.20,
        np.where((horas >= 0) & (horas <= 5), 0.45, 0.75)
    )
    preco_venda_perfil = preco_compra_perfil * 0.6

    for h in horas:
        resultado = microrrede.executar_passo_simulacao(
            irradiancia_w_m2=irradiancia_perfil[h],
            potencia_carga_consumidora_w=carga_perfil[h],
            potencia_bateria_referencia_w=referencia_bateria_perfil[h],
            preco_compra_reais_kwh=preco_compra_perfil[h],
            preco_venda_reais_kwh=preco_venda_perfil[h],
        )
        print(f"h={h:02d}h | Ppv={resultado['potencia_pv_w']:7.1f}W | "
              f"Pbat={resultado['potencia_bateria_w']:7.1f}W | "
              f"Pgrid={resultado['potencia_rede_w']:7.1f}W | "
              f"SOC={resultado['soc_bateria']:.3f} | "
              f"Custo(h)=R${resultado['custo_instantaneo_reais']:6.2f} | "
              f"Custo(acum)=R${resultado['custo_acumulado_reais']:7.2f}")

    print(f"\nCusto total do dia: R$ {microrrede.get_custo_acumulado_reais():.2f}")
    print(f"Energia comprada da rede: {microrrede.get_energia_comprada_kwh_acumulada():.2f} kWh")
    print(f"Energia vendida à rede:   {microrrede.get_energia_vendida_kwh_acumulada():.2f} kWh")