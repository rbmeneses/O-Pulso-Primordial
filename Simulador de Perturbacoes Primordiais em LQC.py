import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. CAMADA DE MODELAGEM FÍSICA (LQCFluidModel)
# ==============================================================================
class LQCFluidModel:
    """
    Implementa a física da Cosmologia Quântica em Loops (LQC) para um fluido
    com equação de estado constante w. Inclui a correção rigorosa para a
    velocidade do som efetiva (c_s,eff).
    """
    def __init__(self, w, rho_planck):
        self.w = w
        self.rho_planck = rho_planck
        self.epsilon = 1e-30  # Evitar divisão por zero

    def background_equations(self, eta, y):
        """
        Equações diferenciais para o background (fundo).
        y = [a, rho]
        """
        a, rho = y
        
        # Garante densidade positiva para estabilidade numérica
        rho = max(rho, 0)
        
        # Densidade efetiva (correção holonômica da LQC)
        # rho_eff governa a expansão H^2 ~ rho_eff
        rho_eff = rho * (1 - rho / self.rho_planck)
        rho_eff = max(rho_eff, 0)
        
        # Equação de Friedmann modificada em tempo conforme
        H_conf = a * np.sqrt((8 * np.pi / 3) * rho_eff) # Assumindo G=1 para simplificar, ajustável nas consts
        
        # Conservação de energia
        p = self.w * rho
        d_rho_d_eta = -3 * H_conf * (rho + p)
        
        # Evolução do fator de escala
        d_a_d_eta = H_conf * a
        
        return [d_a_d_eta, d_rho_d_eta]

    def calculate_derived_quantities(self, eta_arr, a_arr, rho_arr):
        """
        Calcula as quantidades perturbativas derivadas (z, cs2_eff, etc.)
        com base na evolução do background.
        """
        # Preparar arrays de saída
        p_arr = self.w * rho_arr
        
        # 1. Quantidades Efetivas de Fundo
        rho_eff = rho_arr * (1 - rho_arr / self.rho_planck)
        
        # Pressão efetiva (inclui correções quânticas)
        p_eff = p_arr - (rho_arr / self.rho_planck) * (rho_arr + 2 * p_arr)
        
        # Parâmetro de slow-roll efetivo (epsilon = -H'/H^2 approx)
        # Usamos a definição baseada em p e rho para generalidade
        # epsilon_H = 3/2 * (rho + p) / rho (na RG). Aqui usamos as quantidades efetivas.
        epsilon_eff = (3/2) * (rho_eff + p_eff) / (rho_eff + self.epsilon)
        
        # 2. Velocidade do Som Efetiva (A CORREÇÃO SOLICITADA)
        # c_s^2 adiabático clássico
        c_s2 = self.w 
        
        # Fórmula exata da perturbação linear em LQC:
        numerator = c_s2 - (2 * (rho_arr + p_arr) / self.rho_planck) - (2 * rho_arr * c_s2 / self.rho_planck)
        denominator = 1 - 2 * rho_arr / self.rho_planck
        
        # Proteção contra singularidade no denominador (bounce)
        cs2_eff = numerator / (denominator + self.epsilon)
        
        # 3. Variável Mukhanov-Sasaki (z)
        # z = a * sqrt(2 * epsilon)
        sqrt_arg = 2 * epsilon_eff
        sqrt_arg[sqrt_arg < 0] = 0 # Segurança numérica
        z = a_arr * np.sqrt(sqrt_arg)
        
        # Derivadas de z para o potencial z''/z
        # Usamos gradiente numérico (diferenças finitas)
        z_prime = np.gradient(z, eta_arr)
        z_double_prime = np.gradient(z_prime, eta_arr)
        
        # Potencial efetivo
        z_pp_over_z = z_double_prime / (z + self.epsilon)
        
        return {
            'rho_eff': rho_eff,
            'epsilon_eff': epsilon_eff,
            'cs2_eff': cs2_eff,
            'z': z,
            'z_pp_over_z': z_pp_over_z
        }

    def mukhanov_sasaki_equation(self, eta, y, k, interp_cs2_eff, interp_z_pp_over_z):
        """
        Equação de Mukhanov-Sasaki para um modo v_k:
        v'' + (cs_eff^2 * k^2 - z''/z)v = 0
        """
        v, v_prime = y
        
        # Interpolação dos valores de fundo no tempo eta atual
        try:
            cs2_val = interp_cs2_eff(eta)
            z_pot_val = interp_z_pp_over_z(eta)
        except ValueError:
            # Fallback para bordas
            cs2_val = 1.0/3.0
            z_pot_val = 0.0

        omega2 = cs2_val * (k**2) - z_pot_val
        
        return [v_prime, -omega2 * v]


# ==============================================================================
# 2. CAMADA DE SIMULAÇÃO (PrimordialSimulator)
# ==============================================================================
class PrimordialSimulator:
    """
    Gerencia a integração numérica do background e das perturbações.
    """
    def __init__(self, model, sim_params):
        self.model = model
        self.params = sim_params
        self.results = {}

    def run(self):
        # Passo 1: Integrar o Background
        t_eval = np.linspace(self.params['eta_ini'], self.params['eta_fin'], self.params['num_points'])
        
        sol_bg = solve_ivp(
            self.model.background_equations,
            [self.params['eta_ini'], self.params['eta_fin']],
            [self.params['a_ini'], self.params['rho_ini']],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8, atol=1e-8
        )
        
        if not sol_bg.success:
            st.error("Falha na integração do background.")
            return None

        self.results['eta'] = sol_bg.t
        self.results['a'] = sol_bg.y[0]
        self.results['rho'] = sol_bg.y[1]

        # Passo 2: Calcular grandezas derivadas
        derived = self.model.calculate_derived_quantities(
            self.results['eta'], self.results['a'], self.results['rho']
        )
        self.results.update(derived)

        # Criar interpoladores para o solver das perturbações
        interp_cs2 = interp1d(self.results['eta'], self.results['cs2_eff'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        interp_zpp = interp1d(self.results['eta'], self.results['z_pp_over_z'], kind='cubic', bounds_error=False, fill_value="extrapolate")

        # Passo 3: Resolver Perturbações (Modo a Modo)
        power_spectrum = []
        
        for k in self.params['k_values']:
            # Condições Iniciais: Vácuo de Bunch-Davies
            # v_k ~ e^(-ik eta) / sqrt(2k)
            # Inicializamos bem antes do bounce, onde a física é aprox. Minkowski no UV ou padrão no IR
            v_ini = 1.0 / np.sqrt(2 * k)
            v_prime_ini = -1j * k * v_ini
            
            y0 = [v_ini, v_prime_ini] # Scipy solve_ivp lida com complexos automaticamente
            
            sol_k = solve_ivp(
                lambda t, y: self.model.mukhanov_sasaki_equation(t, y, k, interp_cs2, interp_zpp),
                [self.params['eta_ini'], self.params['eta_fin']],
                y0,
                t_eval=[self.params['eta_fin']], # Só precisamos do valor final
                method='RK45',
                rtol=1e-6, atol=1e-6
            )
            
            v_final = sol_k.y[0][-1]
            
            # Espectro de Potência Adimensional: P_R(k) = (k^3 / 2pi^2) * |R_k|^2
            # Onde Curvatura R_k = v_k / z
            z_final = self.results['z'][-1]
            if z_final == 0: z_final = 1e-30
            
            R_k_mod_sq = np.abs(v_final / z_final)**2
            P_R = (k**3 / (2 * np.pi**2)) * R_k_mod_sq
            
            # Ajuste de escala (k físico vs k conformal) - simplificado
            k_physical = k * self.params['alpha_scaling']
            
            power_spectrum.append({
                'k': k,
                'k_physical': k_physical,
                'P_R': P_R
            })

        self.results['power_spectrum'] = pd.DataFrame(power_spectrum)
        return self.results

# ==============================================================================
# 3. INTERFACE DO USUÁRIO (Streamlit)
# ==============================================================================

# Configuração da Página
st.set_page_config(layout="wide", page_title="Simulador LQC - Perturbações Primordiais")

st.title("🌌 Simulador de Perturbações Primordiais em LQC")
st.markdown("""
Este simulador resolve a evolução do universo através do "Big Bounce" (Grande Rebote) da 
Cosmologia Quântica em Loops (LQC). Ele calcula a evolução do fundo (densidade, fator de escala)
e a propagação de perturbações quânticas (espectro de potência), utilizando a 
**fórmula corrigida para a velocidade do som efetiva ($c_{s,eff}^2$)**.
""")

# --- Sidebar: Parâmetros ---
st.sidebar.header("Parâmetros do Modelo")

# Parâmetros Físicos
w_param = st.sidebar.slider("Eq. de Estado (w = p/ρ)", -0.1, 1.0, 1.0/3.0, 0.01, help="1/3 para Radiação, 0 para Matéria, 1 para Campo Escalar Rígido.")
rho_ini = 1.0 # Normalização
rho_p_mult = st.sidebar.slider("Densidade de Planck (x ρ_ini)", 1.5, 50.0, 10.0, 0.5)
rho_planck = rho_ini * rho_p_mult

# Parâmetros de Simulação
st.sidebar.header("Configuração da Simulação")
eta_ini = st.sidebar.number_input("Tempo Inicial (η)", value=-10.0)
eta_fin = st.sidebar.number_input("Tempo Final (η)", value=10.0)
num_points = st.sidebar.slider("Resolução (Pontos)", 500, 5000, 1000)

st.sidebar.subheader("Modos de Fourier (k)")
k_min = st.sidebar.number_input("k Mínimo", value=0.1)
k_max = st.sidebar.number_input("k Máximo", value=10.0)
n_k = st.sidebar.number_input("Qtd de Modos", value=20, min_value=5, max_value=100)

# Botão de Execução
if st.button("🚀 Iniciar Simulação"):
    
    # 1. Configurar
    model = LQCFluidModel(w=w_param, rho_planck=rho_planck)
    
    # Gerar logspace de k
    k_values = np.logspace(np.log10(k_min), np.log10(k_max), int(n_k))
    
    sim_params = {
        'eta_ini': eta_ini,
        'eta_fin': eta_fin,
        'num_points': num_points,
        'a_ini': 1.0, # Começamos com a=1 arbitrariamente longe do bounce
        'rho_ini': rho_ini,
        'k_values': k_values,
        'alpha_scaling': 1.0 # Simplificação para plot
    }
    
    simulator = PrimordialSimulator(model, sim_params)
    
    with st.spinner("Integrando equações de fundo e perturbativas..."):
        results = simulator.run()
    
    if results:
        st.success("Simulação concluída!")
        
        # --- Visualização dos Resultados ---
        
        # 1. Gráficos de Fundo (Background)
        st.subheader("1. Evolução do Background (Fundo)")
        fig_bg = make_subplots(rows=2, cols=2, subplot_titles=("Densidade de Energia (ρ)", "Fator de Escala (a)", "Velocidade do Som Efetiva (c_s,eff²)", "Potencial (z''/z)"))
        
        # Densidade
        fig_bg.add_trace(go.Scatter(x=results['eta'], y=results['rho'], name='ρ', line=dict(color='red')), row=1, col=1)
        fig_bg.add_hline(y=rho_planck, line_dash="dash", annotation_text="Densidade Planck", row=1, col=1)
        
        # Fator de Escala
        fig_bg.add_trace(go.Scatter(x=results['eta'], y=results['a'], name='a', line=dict(color='blue')), row=1, col=2)
        
        # Cs2 Efetiva
        fig_bg.add_trace(go.Scatter(x=results['eta'], y=results['cs2_eff'], name='cs2_eff', line=dict(color='green')), row=2, col=1)
        fig_bg.add_hline(y=w_param, line_dash="dot", annotation_text="w clássico", row=2, col=1)
        
        # Potencial
        fig_bg.add_trace(go.Scatter(x=results['eta'], y=results['z_pp_over_z'], name="z''/z", line=dict(color='purple')), row=2, col=2)
        
        fig_bg.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_bg, use_container_width=True)
        
        # Explicação sobre Cs2
        with st.expander("ℹ️ Detalhes sobre a Velocidade do Som Efetiva"):
            st.write(r"""
            O gráfico acima mostra $c_{s,eff}^2$. Note que perto do bounce (onde $\rho \to \rho_P$), 
            o valor desvia significativamente de $w$ (valor clássico). 
            Em alguns modelos, ele pode até se tornar negativo, indicando instabilidades taquiônicas momentâneas.
            A fórmula utilizada é:
            """)
            st.latex(r"c_{s,eff}^2 = \frac{c_s^2 - \frac{2(\rho+p)}{\rho_P} - \frac{2\rho c_s^2}{\rho_P}}{1 - \frac{2\rho}{\rho_P}}")

        # 2. Espectro de Potência
        st.subheader("2. Espectro de Potência Primordial")
        df_spec = results['power_spectrum']
        
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=df_spec['k'], 
            y=df_spec['P_R'], 
            mode='lines+markers',
            name='P_R(k)'
        ))
        
        fig_spec.update_layout(
            xaxis_title="Modo k (Escala)",
            yaxis_title="Potência P_R(k)",
            xaxis_type="log",
            yaxis_type="log",
            title="Espectro de Potência Escalar"
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        
        st.markdown("""
        **Interpretação:**
        * Modos com **k pequeno** (infravermelho, escalas maiores que o horizonte no bounce) tendem a ver uma curvatura diferente.
        * Modos com **k grande** (ultravioleta) "sentem" menos o bounce se estiverem profundamente dentro do horizonte, mas podem sofrer amplificação paramétrica dependendo da nitidez do rebote.
        """)

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Iniciar Simulação'.")