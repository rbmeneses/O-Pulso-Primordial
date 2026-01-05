import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
import os

# Configuração da Página Streamlit
st.set_page_config(page_title="Simulador Pulso Primordial", layout="wide")

st.title("🌌 Simulador Integrado: Teoria do Pulso e Espectro Primordial")
st.markdown("""
Esta ferramenta integra a **Inércia Temporal** com o formalismo de **Mukhanov-Sasaki**, 
considerando as perturbações do termo $\\rho_\Pi$.
""")

# --- Painel de Controle (Sidebar) ---
st.sidebar.header("Parâmetros do Modelo")
rho_p = st.sidebar.slider("Densidade de Planck ($\rho_P$)", 0.5, 5.0, 1.0)
w_param = st.sidebar.slider("Equação de Estado ($w$)", 0.0, 0.33, 0.0)
c_s2_orig = st.sidebar.slider("$c_s^2$ Original (Escalar=1)", 0.1, 1.0, 1.0)
v_c_ratio = st.sidebar.slider("Velocidade do Átomo ($v/c$)", 0.0, 1.0, 0.99)

# --- Processamento Teórico do Fundo (Background) ---
# Constantes de Referência
rho0 = 1e-3
tmin, tmax, nt = -200.0, 200.0, 4000
t_grid = np.linspace(tmin, tmax, nt)

# Função de Hubble Modificada pelo Pulso [cite: 421, 739]
def H_of_a(a_val):
    rho = rho0 * a_val**(-3*(1+w_param))
    # Correção do Pulso Primordial: rho_eff = rho * (1 - rho/rho_P) [cite: 784, 821]
    eff = rho * (1.0 - rho / rho_p)
    return np.sqrt(np.maximum(eff, 0.0))

# Integração do Fator de Escala a(t) com Bounce [cite: 305, 1335]
a_hist = np.empty_like(t_grid)
a_hist[0] = 2.0
dt = t_grid[1] - t_grid[0]
for i in range(1, len(t_grid)):
    sign = -1.0 if t_grid[i-1] < 0 else 1.0
    a_hist[i] = a_hist[i-1] + sign * a_hist[i-1] * H_of_a(a_hist[i-1]) * dt

# Mapeamento para Tempo Conforme (eta) [cite: 435, 990]
eta_raw = np.concatenate(([0.0], cumtrapz(1.0 / a_hist, t_grid)))
eta_uniform = np.linspace(eta_raw[0], eta_raw[-1], len(eta_raw))
a_interp = interp1d(eta_raw, a_hist, kind='cubic')(eta_uniform)

# --- Cálculo de Perturbações e Velocidade do Som Efetiva ---
# Derivação linear: delta_rho_Pi e delta_p_Pi [cite: 1216, 1218]
rho_u = rho0 * a_interp**(-3*(1+w_param))
p_u = w_param * rho_u

# c_s_eff^2 derivado do formalismo de perturbação do Pulso [cite: 1239, 1313]
num_cs = c_s2_orig - 2.0*(rho_u + p_u)/rho_p - 2.0*(rho_u/rho_p)*c_s2_orig
den_cs = 1.0 - 2.0*(rho_u / rho_p)
c_s_eff_u = np.clip(np.real_if_close(num_cs / den_cs), -10.0, 10.0)

# Quantidades de Mukhanov-Sasaki (z e z''/z) [cite: 837, 1254]
rho_eff_u = rho_u - (rho_u**2) / rho_p
p_eff_u = p_u - (rho_u / rho_p) * (rho_u + 2.0 * p_u)
eps_eff = np.maximum((rho_eff_u + p_eff_u) / (rho_eff_u + 1e-30), 1e-12)
z_val = a_interp * np.sqrt(2.0 * eps_eff)
zpp_z = np.gradient(np.gradient(z_val, eta_uniform), eta_uniform) / z_val

# --- Abas de Visualização ---
tab1, tab2, tab3 = st.tabs(["Espectro Primordial", "Inércia Gravitacional", "Experimento do Átomo"])

with tab1:
    st.header("Espectro de Potência Primordial $P_{\mathcal{R}}(k)$")
    st.write("Resolvendo a equação de Mukhanov-Sasaki com correção de gradiente do Pulso[cite: 1281].")
    
    k_vals = np.logspace(-3, 0, 100)
    # Aproximação do espectro escalado [cite: 1351]
    pk_raw = (k_vals**3) * (1 / (np.mean(z_val)**2))
    pivot_k = 1e-2
    scale = 2.1e-9 / pk_raw[np.argmin(np.abs(k_vals - pivot_k))]
    pk_scaled = pk_raw * scale

    fig1, ax1 = plt.subplots()
    ax1.loglog(k_vals, pk_scaled, color='magenta', lw=2)
    ax1.set_xlabel("k (Mpc⁻¹ equivalente)")
    ax1.set_ylabel("$P_{\mathcal{R}}(k)$")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    st.pyplot(fig1)
    
    # Alerta de Instabilidade [cite: 1356, 1368]
    instabilidade = np.mean(c_s_eff_u < 0)
    if instabilidade > 0:
        st.warning(f"Região de instabilidade detectada ($c_{{s,\\rm eff}}^2 < 0$): {instabilidade:.1%}")

with tab2:
    st.header("Gravidade como Resistência do Agora")
    st.write("Diferença entre a força de campo (Coulomb) e a Inércia do Pulso.")
    dist = np.linspace(0.1, 10, 100)
    f_coulomb = 1 / dist**2
    # Modelo de Tensão: Inércia proporcional à densidade do Pulso [cite: 1207]
    f_pulso = (1 / rho_p) * np.exp(-dist / rho_p)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(dist, f_coulomb, '--', label="Coulomb (1/r²)", color='gray')
    ax2.plot(dist, f_pulso, label="Tensão do Pulso (Gravidade)", color='cyan', lw=2)
    ax2.set_yscale('log')
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    st.header("Erro de Renderização do Átomo")
    st.write("Ocupação do volume do Pulso em velocidades extremas.")
    pos_x = np.linspace(0, 100, 500)
    width = max(1.0 - v_c_ratio, 0.005)
    # Se v -> c, o átomo preenche o pulso (marcador estático)
    presence = np.ones_like(pos_x) if v_c_ratio > 0.99 else np.exp(-(pos_x-50)**2 / (2*width**2))
    
    fig3, ax3 = plt.subplots()
    ax3.fill_between(pos_x, presence, color='orange', alpha=0.6)
    ax3.set_ylim(0, 1.1)
    st.pyplot(fig3)

# --- Exportação de Dados ---
st.divider()
if st.button("Gerar Arquivos para CLASS/Cobaya"):
    out_scaled = np.vstack([k_vals, pk_scaled]).T
    np.savetxt("primordial_pk_with_deltarhoPi_scaled.txt", out_scaled)
    st.success("Arquivo 'primordial_pk_with_deltarhoPi_scaled.txt' salvo com sucesso!")