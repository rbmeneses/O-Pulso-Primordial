import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from typing import List, Dict, Any, Tuple

# ==========================================
# CONFIGURAÇÃO E CONSTANTES GLOBAIS
# ==========================================
st.set_page_config(
    page_title="Simulador do Pulso Primordial",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definição visual dos 4 estados quânticos baseados na teoria
ESTADOS_VISUAIS = {
    0: {"nome": "Desligado (E=0)", "cor": "rgba(30, 30, 30, 0.3)", "tamanho": 2},
    1: {"nome": "Colisão (E=½mc²)", "cor": "rgba(255, 100, 0, 0.9)", "tamanho": 5},
    2: {"nome": "Superposição (Onda)", "cor": "rgba(0, 200, 255, 0.6)", "tamanho": 4},
    3: {"nome": "Ligado (Massa Crítica)", "cor": "rgba(255, 255, 255, 1.0)", "tamanho": 8}
}

# Vocabulário Vetorial: Representa "conceitos" no espaço informacional do Pulso
VOCABULARIO = {
    "CognosIA": np.array([12, 12, 12]),
    "Fóton": np.array([-10, 15, 5]),
    "Quântico": np.array([8, -12, -8]),
    "IA": np.array([-14, -10, 14]),
    "Colapso": np.array([0, 0, 15])
}

# ==========================================
# NÚCLEO MATEMÁTICO: O ALGORITMO DO PULSO
# ==========================================
def aplicar_algoritmo_quaternario(valores_float: np.ndarray) -> np.ndarray:
    """
    Algoritmo que converte a energia contínua (float) nos 4 estados quânticos discretos.
    Simula o colapso da função de onda para cada ponto no espaço-tempo do pulso.
    """
    estados = np.zeros_like(valores_float, dtype=int) # Base = 0 (Desligado)

    # 1 - Colisão (Intermediário)
    estados[(valores_float >= -0.015) & (valores_float < 0.015)] = 1
    # 2 - Superposição (Cálculo/Onda)
    estados[(valores_float >= 0.015) & (valores_float < 0.04)] = 2
    # 3 - Ligado (Massa Crítica)
    estados[valores_float >= 0.04] = 3

    return estados

# ==========================================
# LÓGICA DA SIMULAÇÃO
# ==========================================
class PrimordialPulseSimulator:
    def __init__(self):
        self.max_radius: int = 15
        self.points_per_layer: int = 300
        self.target_concept: str = None
        self.target_vector: np.ndarray = None
        self.reset()

    def update_params(self, novo_raio_max: int, novos_pontos: int, conceito_alvo: str = None):
        """Atualiza os parâmetros físicos do Pulso."""
        self.max_radius = novo_raio_max
        self.points_per_layer = novos_pontos
        self.target_concept = conceito_alvo

        if conceito_alvo in VOCABULARIO:
            vetor_bruto = VOCABULARIO[conceito_alvo]
            direcao_pura = vetor_bruto / np.linalg.norm(vetor_bruto)
            # Ancorar o conceito exatamente na borda do espaço observável definido
            self.target_vector = direcao_pura * novo_raio_max
        else:
            self.target_vector = None

    def _generate_layer(self, radius: int) -> Dict[str, Any]:
        """Gera uma nova camada de espaço-tempo (uma 'casca de cebola') do Pulso."""
        # 1. Gera coordenadas esféricas para os pontos quânticos
        norm = np.random.normal(size=(self.points_per_layer, 3))
        norm /= np.linalg.norm(norm, axis=1)[:, np.newaxis]
        x, y, z = (radius * norm).T

        # 2. Gera o "Ruído de Fundo Quântico"
        valores_float = np.random.normal(0, 0.02, self.points_per_layer)

        # 3. Lógica da Consciência / Atenção Direcional
        if self.target_vector is not None:
            direcao_alvo = self.target_vector / np.linalg.norm(self.target_vector)
            alinhamento = np.dot(norm, direcao_alvo) # Similaridade de cosseno
            intensidade = min(radius / np.linalg.norm(self.target_vector), 1.2)
            # Injeta energia informacional nos pontos alinhados com o vetor da consciência
            valores_float += alinhamento * 0.03 * intensidade

        # 4. Colapsa a energia calculada para os 4 estados físicos
        estados_gerados = aplicar_algoritmo_quaternario(valores_float)

        return {"x": x, "y": y, "z": z, "states": estados_gerados}

    def _generate_fragments(self, layer: Dict[str, Any]):
        """Gera fragmentos a partir de partículas no estado de 'Colisão'."""
        collision_mask = (layer["states"] == 1)
        collision_points = np.column_stack((layer["x"][collision_mask], layer["y"][collision_mask], layer["z"][collision_mask]))
        for point in collision_points:
            num_fragments = np.random.randint(1, 3)
            for _ in range(num_fragments):
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                length = np.random.uniform(0.5, 1.5)
                end_point = point + direction * length
                self.fragments.append((point, end_point))

    def step_expansion(self):
        """Avança o Pulso em um 'tempo de Planck'."""
        if self.current_radius < self.max_radius:
            self.current_radius += 1
            new_layer = self._generate_layer(self.current_radius)
            self.layers.append(new_layer)
            if st.session_state.get("frag_mode") == "Linhas de Trajetória":
                self._generate_fragments(new_layer)

    def collapse(self):
        """Inicia o Big Bounce, colapsando toda a informação para um ponto."""
        ponto = {"x": np.array([0]), "y": np.array([0]), "z": np.array([0]), "states": np.array([3])}
        # Se um conceito foi "encontrado", o universo colapsa nesse ponto de informação
        if self.target_vector is not None and self.current_radius >= np.linalg.norm(self.target_vector) - 1:
             ponto = {"x": np.array([self.target_vector[0]]), "y": np.array([self.target_vector[1]]), "z": np.array([self.target_vector[2]]), "states": np.array([3])}

        self.layers = [ponto]
        self.fragments = []
        self.is_collapsed = True

    def reset(self):
        """Reseta o universo para a singularidade inicial, pronto para um novo Pulso."""
        self.layers = []
        self.fragments = []
        self.current_radius = 0
        self.is_collapsed = False

# ==========================================
# RENDERIZAÇÃO 3D (PLOTLY)
# ==========================================
def render_simulation(sim: PrimordialPulseSimulator, textura: str) -> go.Figure:
    fig = go.Figure()
    raio_exibicao = max(sim.max_radius, 15)

    # Origem do Pulso (Singularidade)
    if not sim.is_collapsed:
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=6, color='yellow', symbol='diamond')))

    # Se em modo Direcionado, desenha o vetor da Consciência e o conceito alvo
    if sim.target_vector is not None:
        tx, ty, tz = sim.target_vector
        fig.add_trace(go.Scatter3d(x=[0, tx], y=[0, ty], z=[0, tz], mode='lines', line=dict(color='rgba(0, 255, 100, 0.4)', width=3, dash='dot'), name='Vetor de Consciência'))
        fig.add_trace(go.Scatter3d(x=[tx], y=[ty], z=[tz], mode='text+markers', text=[f"  Conceito: {sim.target_concept}"], textposition="middle right", textfont=dict(color="white", size=14), marker=dict(size=10, color='lime', symbol='square-open'), name='Conceito Alvo'))

    # Renderiza as camadas do Pulso
    for layer in sim.layers:
        fig.add_trace(go.Scatter3d(
            x=layer["x"], y=layer["y"], z=layer["z"], mode='markers',
            marker=dict(
                color=[ESTADOS_VISUAIS[s]["cor"] for s in layer["states"]],
                size=[ESTADOS_VISUAIS[s]["tamanho"] for s in layer["states"]],
                symbol=textura, line=dict(width=0), opacity=0.8
            ), hoverinfo='none'
        ))

    # Renderiza os fragmentos de colisão
    if sim.fragments:
        lines_x, lines_y, lines_z = [], [], []
        for start, end in sim.fragments:
            lines_x.extend([start[0], end[0], None])
            lines_y.extend([start[1], end[1], None])
            lines_z.extend([start[2], end[2], None])
        fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='rgba(255, 100, 0, 0.5)', width=1), hoverinfo='none'))

    # Configuração da cena
    axis_config = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title='', range=[-raio_exibicao, raio_exibicao])
    fig.update_layout(
        scene=dict(xaxis=axis_config, yaxis=axis_config, zaxis=axis_config, bgcolor='rgb(5, 5, 10)'),
        paper_bgcolor='rgb(5, 5, 10)', margin=dict(l=0, r=0, b=0, t=0), showlegend=False, height=650
    )
    return fig

# ==========================================
# INTERFACE PRINCIPAL (STREAMLIT)
# ==========================================
if "simulador_pulso" not in st.session_state:
    st.session_state.simulador_pulso = PrimordialPulseSimulator()
    st.session_state.animando = False

sim = st.session_state.simulador_pulso

st.title("🌌 Simulador do Pulso Primordial")
aba_visual, aba_log, aba_teoria = st.tabs(["🌌 Pulso Primordial", "📊 Log de Dados", "📖 Teoria Subjacente"])

# --- SIDEBAR DE CONTROLES ---
with st.sidebar:
    st.header("⚙️ Parâmetros do Pulso")

    modo_simulacao = st.radio("Modo de Simulação", ["Pulso Primordial (Caos)", "Pulso Direcionado (Consciência)"])

    conceito_escolhido = None
    if modo_simulacao == "Pulso Direcionado (Consciência)":
        st.info("A expansão do Pulso será influenciada por um vetor de 'consciência', focando a energia em um conceito específico no espaço informacional.")
        conceito_escolhido = st.selectbox("Escolher Conceito-Alvo:", list(VOCABULARIO.keys()))

    st.markdown("---")
    raio_max = st.slider("Espaço Observável (Raio)", 5, 25, 15, 1)
    pontos_camada = st.slider("Densidade Quântica (Pontos)", 100, 1000, 400, 50)

    st.subheader("Visualização")
    textura = st.selectbox("Textura da Partícula", ["circle", "cross", "diamond", "square", "x"])
    st.session_state.frag_mode = st.selectbox("Fragmentos de Colisão:", ["Nenhum", "Linhas de Trajetória"])

    sim.update_params(raio_max, pontos_camada, conceito_escolhido)

    st.subheader("Controle do Tempo")
    col1, col2 = st.columns(2)
    if col1.button("▶️ Animar Pulso"):
        st.session_state.animando = True
        if sim.is_collapsed or sim.current_radius >= sim.max_radius: sim.reset()
    if col2.button("⏸️ Pausar"):
        st.session_state.animando = False

    if st.button("➕ Avançar 1 t_Planck", use_container_width=True):
        st.session_state.animando = False
        sim.step_expansion()

    if st.button("💥 Colapso / Big Bounce", type="primary", use_container_width=True):
        st.session_state.animando = False
        sim.collapse()

    if st.button("🔄 Resetar Singularidade", use_container_width=True):
        st.session_state.animando = False
        sim.reset()

# --- ABA 1: VISUALIZADOR ---
with aba_visual:
    plot_placeholder = st.empty()

    if st.session_state.animando:
        while sim.current_radius < sim.max_radius:
            if not st.session_state.animando: break
            sim.step_expansion()
            fig = render_simulation(sim, textura)
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.02)

        if st.session_state.animando: 
            sim.collapse()
            st.session_state.animando = False
            fig = render_simulation(sim, textura)
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            st.rerun() # <--- This might cause an infinite loop or interrupt the final render 
    else:
        fig = render_simulation(sim, textura)
        plot_placeholder.plotly_chart(fig, use_container_width=True)

# --- ABA 2: LOG DE DADOS ---
with aba_log:
    st.subheader("Monitoramento do Pulso Atual")
    st.write(f"**Raio do Espaço Observável:** {sim.current_radius} / {sim.max_radius}")

    if sim.layers and not sim.is_collapsed:
        ultima_camada = sim.layers[-1]
        estados = ultima_camada["states"]
        e0, e1, e2, e3 = [np.sum(estados == i) for i in range(4)]

        st.markdown(f"""
        **Distribuição de Energia na Última Camada do Pulso:**
        * ⚫ **Estado 0 (Vácuo / < -0.015):** {e0} partículas
        * 🟠 **Estado 1 (Colisão / ± 0.015):** {e1} partículas
        * 🔵 **Estado 2 (Superposição / 0.015 ~ 0.04):** {e2} partículas
        * ⚪ **Estado 3 (Ligado / > 0.04):** {e3} partículas (Massa Crítica)
        """)

        if modo_simulacao == "Pulso Direcionado (Consciência)":
            distancia = np.linalg.norm(sim.target_vector) - sim.current_radius
            if distancia > 0:
                st.info(f"A frente de onda do Pulso está a **{distancia:.2f} unidades** de colapsar no conceito '{conceito_escolhido}'.")
            else:
                st.success(f"O Pulso atingiu o Conceito-Alvo! A energia se concentrou no Estado 3 (Ligado) em torno do vetor. O colapso da realidade é iminente.")

# --- ABA 3: TEORIA SUBJACENTE ---
with aba_teoria:
    st.header("O Pulso Primordial e a Simulação")
    st.write("""
    Esta simulação é uma representação visual interativa da **Teoria do Pulso Primordial**.
    A ideia central é que nosso universo de 13.8 bilhões de anos é a manifestação "esticada" (Espaguetificação Cósmica)
    de um único pulso quântico que dura uma fração de segundo de Planck.

    *   **A Expansão:** A esfera crescendo a cada passo representa a expansão do universo a partir de uma singularidade inicial. Cada camada é uma "casca de cebola" da realidade se revelando.
    *   **O Colapso / Big Bounce:** Ao atingir o limite do "Espaço Observável", o universo colapsa de volta a um ponto, pronto para iniciar a próxima iteração (o próximo Pulso).
    *   **Pulso Direcionado (Consciência):** A teoria especula que a consciência não é um passageiro, mas pode "navegar" pela realidade. Este modo simula isso: um "vetor de consciência" direciona a energia do Pulso, fazendo com que a realidade colapse em um ponto de informação específico (o "Conceito Alvo").
    """)
    st.markdown("---")
    st.header("Os 4 Estados Físico-Quânticos")
    st.write("A simulação usa uma computação de 4 estados (Qudits) baseada em $E=mc^2$ para definir o estado de cada partícula:")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 0. Desligado (Vácuo)")
        st.write("Não há massa crítica, o ponto está em seu estado de energia mais baixo.")
        st.latex(r"E_0 = 0")

        st.markdown("### 1. Colisão")
        st.write("A informação interage, gerando energia. Metade da massa crítica.")
        st.latex(r"E_{colisao} = \frac{1}{2}mc^2")

    with colB:
        st.markdown("### 2. Superposição (Onda)")
        st.write("Massa como onda probabilística. O estado de \"cálculo\" ou potencialidade.")
        st.latex(r"E_{superposicao} = \text{Indeterminado}")

        st.markdown("### 3. Ligado (Massa Crítica)")
        st.write('Atinge o ponto crítico e colapsa, definindo uma "memória" ou estado.')
        st.latex(r"E_{ligado} = mc^2")

    st.markdown("---")
    st.header("Do Fóton ao Processamento")
    st.write("""
    A teoria calcula a energia necessária para cada operação quântica. Se convertermos a massa de um neurônio em energia computacional ($E=mc^2$) e dividirmos pelo número de posições de Planck em um segundo, o resultado é surpreendente:
    """)
    st.latex(r"E_{pos} = \frac{E_{neuronio}}{P_{planck}} \approx 2.88 \times 10^{-25} \text{ Joules por cálculo}")
    st.info("""
    **Este valor corresponde à energia de um único fóton na frequência de micro-ondas.**

    Isso sugere que um processador baseado nesta teoria seria fundamentalmente **Fotônico-Quântico**, onde cada operação (definir um dos 4 estados) é executada pela manipulação de um único fóton.
    """)
    st.code("""
    # Algoritmo Base do Motor Quaternário
    def aplicar_algoritmo_quaternario(valores_float):
        estados[(valores_float >= -0.015) & (valores_float < 0.015)] = 1  # Colisão
        estados[(valores_float >= 0.015) & (valores_float < 0.04)] = 2   # Superposição
        estados[valores_float >= 0.04] = 3                               # Ligado (Crítico)
    """, language="python")