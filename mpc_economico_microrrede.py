import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# GPC ECONÔMICO -- PREDIÇÃO MATRICIAL (DYNAMIC MATRIX), CUSTO EM R$
# --------------------------------------------------------------------------- #
class MPC_Economico:
    def __init__(self,
                 capacidade_maxima_bateria_wh,
                 eficiencia_carga,
                 eficiencia_descarga,
                 potencia_nominal_pv_w,
                 horizonte_predicao,
                 horizonte_controle,
                 peso_lambda,
                 potencia_bateria_min_w,
                 potencia_bateria_max_w,
                 delta_potencia_bateria_max_w,
                 irradiancia_referencia_w_m2=1000.0,
                 passo_tempo_segundos=3600.0,
                 soc_minimo=0.0,
                 soc_maximo=1.0,
                 peso_penalidade_soc=1e6):

        N2, Nu = horizonte_predicao, horizonte_controle

        self.horizonte_predicao = N2
        self.horizonte_controle = Nu
        self.peso_lambda = peso_lambda

        self.potencia_bateria_min_w = potencia_bateria_min_w
        self.potencia_bateria_max_w = potencia_bateria_max_w
        self.delta_potencia_bateria_max_w = delta_potencia_bateria_max_w
        self.soc_minimo = soc_minimo
        self.soc_maximo = soc_maximo
        self.peso_penalidade_soc = peso_penalidade_soc

        # --- Parâmetros fixos do modelo linear interno de predição --- #
        self._potencia_nominal_pv_w = potencia_nominal_pv_w
        self._irradiancia_referencia_w_m2 = irradiancia_referencia_w_m2
        self._passo_tempo_segundos = passo_tempo_segundos

        self.eficiencia_carga = eficiencia_carga
        self.eficiencia_descarga = eficiencia_descarga
        
        passo_horas = self._passo_tempo_segundos / 3600.0
        self._ganho_soc_k = -(passo_horas) / capacidade_maxima_bateria_wh

        # --- Matrizes GPC --- #
        self._matriz_m                  = self._construir_matriz_dinamica_potencia(N2, Nu)
        self._matriz_l2                 = np.tril(np.ones((N2, N2)))
        self._matriz_soma_cumulativa_nu = np.tril(np.ones((Nu, Nu)))

        self._ultimo_delta_u_otimo = None

    # ------------------------------------------------------------------ #
    # Getters (para inspeção/depuração do preditor interno)
    # ------------------------------------------------------------------ #
    def get_ganho_soc_k(self):    return self._ganho_soc_k

    # ------------------------------------------------------------------ #
    # Construção da matriz dinâmica M (núcleo do GPC)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _construir_matriz_dinamica_potencia(N2, Nu):
        M = np.zeros((N2, Nu))
        for j in range(N2):
            limite = min(j + 1, Nu)
            M[j, :limite] = 1.0
        return M

    # ------------------------------------------------------------------ #
    # Núcleo da predição:
    # ------------------------------------------------------------------ #
    def _predizer_trajetorias_virtual_for(self, 
                                  soc_atual, 
                                  potencia_virtual_anterior, 
                                  irradiancia_futura_w_m2, 
                                  carga_futura_w, 
                                  delta_u_virtual):
    
        N2 = self.horizonte_predicao
        Nu = self.horizonte_controle
        
        potencia_virtual_predita    = np.zeros(N2)
        potencia_real_predita       = np.zeros(N2)
        soc_predito                 = np.zeros(N2)
        potencia_rede_predita_w     = np.zeros(N2)
        
        # PV predito
        potencia_pv_predita_w = self._potencia_nominal_pv_w * (
            np.clip(irradiancia_futura_w_m2, 0.0, None) / self._irradiancia_referencia_w_m2)
    
        soc_k = soc_atual
        p_bat_virtual_k = potencia_virtual_anterior
        
        for k in range(N2):
            du_k = delta_u_virtual[k] if k < Nu else 0.0
            p_bat_virtual_k = p_bat_virtual_k + du_k
            potencia_virtual_predita[k] = p_bat_virtual_k
            
            soc_k = soc_k + self._ganho_soc_k * p_bat_virtual_k
            soc_predito[k] = soc_k
        
            if p_bat_virtual_k > 0:
                p_bat_real_k = p_bat_virtual_k * self.eficiencia_descarga
            else:
                p_bat_real_k = p_bat_virtual_k / self.eficiencia_carga
                
            potencia_real_predita[k] = p_bat_real_k
            
            potencia_rede_predita_w[k] = carga_futura_w[k] - potencia_pv_predita_w[k] - p_bat_real_k
    
        return potencia_real_predita, soc_predito, potencia_rede_predita_w
    
    def _predizer_trajetorias_virtual_matriz(self,
                                  soc_atual,
                                  potencia_virtual_anterior,
                                  irradiancia_futura_w_m2,
                                  carga_futura_w,
                                  delta_u_virtual):

            N2 = self.horizonte_predicao
            
            potencia_pv_predita_w = self._potencia_nominal_pv_w * (
                np.clip(irradiancia_futura_w_m2, 0.0, None) / self._irradiancia_referencia_w_m2)
            
            potencia_virtual_predita = (potencia_virtual_anterior * np.ones(N2) +
                                         self._matriz_m @ delta_u_virtual)
    
            soc_predito = (soc_atual * np.ones(N2) +
                           self._ganho_soc_k * (self._matriz_l2 @ potencia_virtual_predita))
    
            potencia_real_predita = np.where(
                potencia_virtual_predita > 0,
                potencia_virtual_predita * self.eficiencia_descarga,
                potencia_virtual_predita / self.eficiencia_carga
            )
    
            potencia_rede_predita_w = carga_futura_w - potencia_pv_predita_w - potencia_real_predita
    
            return potencia_real_predita, soc_predito, potencia_rede_predita_w

    def _custo_economico(self, potencia_rede_predita_w, preco_compra, preco_venda):
 
        passo_horas = self._passo_tempo_segundos / 3600.0
        energia_rede_kwh = (potencia_rede_predita_w * passo_horas) / 1000.0

        custo = np.where(
            potencia_rede_predita_w >= 0.0,
            energia_rede_kwh * preco_compra,
            energia_rede_kwh * preco_venda,
        )
        
        return np.sum(custo)

    def calcular_u(self,
                    soc_atual,
                    potencia_bateria_anterior,
                    irradiancia_futura_w_m2,
                    carga_futura_w,
                    preco_compra_futuro_reais_kwh,
                    preco_venda_futuro_reais_kwh):

        if potencia_bateria_anterior > 0:
            potencia_bateria_anterior /= self.eficiencia_descarga
        else:
            potencia_bateria_anterior *= self.eficiencia_carga
        
        def funcao_custo(delta_u):
            _, soc_predito, potencia_rede_predita = self._predizer_trajetorias_virtual_matriz(
                soc_atual, potencia_bateria_anterior,
                irradiancia_futura_w_m2, carga_futura_w, delta_u)

            custo_economico = self._custo_economico(
                potencia_rede_predita, preco_compra_futuro_reais_kwh, preco_venda_futuro_reais_kwh)

            termo_esforco = self.peso_lambda * np.sum(delta_u ** 2)

            violacao_inferior = np.maximum(0.0, self.soc_minimo - soc_predito)
            violacao_superior = np.maximum(0.0, soc_predito - self.soc_maximo)
            termo_penalidade_soc = self.peso_penalidade_soc * np.sum(
                violacao_inferior ** 2 + violacao_superior ** 2)

            return custo_economico + termo_esforco + termo_penalidade_soc

        bounds_delta_u = [(-self.delta_potencia_bateria_max_w,
                            self.delta_potencia_bateria_max_w) for _ in range(self.horizonte_controle)]

        matriz_soma_cumulativa = self._matriz_soma_cumulativa_nu

        restricoes = [
            {
                'type': 'ineq',
                'fun': lambda delta_u: self.potencia_bateria_max_w -
                       (potencia_bateria_anterior + matriz_soma_cumulativa @ delta_u)
            },
            {
                'type': 'ineq',
                'fun': lambda delta_u: (potencia_bateria_anterior +
                       matriz_soma_cumulativa @ delta_u) - self.potencia_bateria_min_w
            },
        ]


        melhor_delta_u = None
        melhor_custo = np.inf

        resultado = minimize(
            funcao_custo,
            np.zeros(self.horizonte_controle),
            method='SLSQP',
            bounds=bounds_delta_u,
            constraints=restricoes,
            options={'ftol': 1e-9, 'maxiter': 200}
        )
        if resultado.fun < melhor_custo:
            melhor_custo = resultado.fun
            melhor_delta_u = resultado.x

        if melhor_delta_u is None:
            delta_u_efetivo = -self.delta_potencia_bateria_max_w
        else:
            delta_u_efetivo = melhor_delta_u[0]
        
        potencia_virtual_aplicada = potencia_bateria_anterior + delta_u_efetivo
        
        if potencia_virtual_aplicada > 0:
            potencia_real_comando = potencia_virtual_aplicada * self.eficiencia_descarga
        else:
            potencia_real_comando = potencia_virtual_aplicada / self.eficiencia_carga
        
        potencia_real_comando = np.clip(potencia_real_comando,
                                        self.potencia_bateria_min_w,
                                        self.potencia_bateria_max_w)
        
        return potencia_real_comando