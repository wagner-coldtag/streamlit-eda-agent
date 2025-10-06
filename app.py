import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from openai import OpenAI
from langchain.memory import ConversationBufferMemory

# ======================
# CONFIGURAÇÃO
# ======================
st.set_page_config(page_title="💳 Agente de EDA", page_icon="💳", layout="wide")
st.title("💳 Agente de EDA — Analisador de CSV")

# ======================
# 1️⃣ Upload CSV
# ======================
uploaded_file = st.file_uploader("Faça upload do seu arquivo CSV", type=["csv"])
if uploaded_file:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(uploaded_file)
    st.write("Dataset carregado com shape:", df.shape)
    st.dataframe(df.head())
else:
    st.warning("Por favor, faça upload de um arquivo CSV para prosseguir.")
    st.stop()

# ======================
# Preparar colunas e estatísticas básicas
# ======================
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

stats_numeric = df[numeric_cols].describe().to_dict() if numeric_cols else {}
stats_categorical = df[cat_cols].describe().to_dict() if cat_cols else {}

# ======================
# 2️⃣ Setup LLM (Hugging Face OpenAI-compatible)
# ======================
HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_NAME = "CohereLabs/command-a-reasoning-08-2025:cohere"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# ======================
# 3️⃣ Memória do agente
# ======================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ======================
# 4️⃣ Pergunta do usuário
# ======================
st.header("🤖 Pergunte ao agente sobre o dataset")
user_question = st.text_area(
    "Pergunte qualquer coisa sobre o dataset, por exemplo: distribuições, correlações, outliers, tendências..."
)

if st.button("Perguntar"):
    if not user_question.strip():
        st.warning("Digite uma pergunta para continuar.")
    else:
        with st.spinner("O agente está analisando os dados..."):
            try:
                # Preparar resumo detalhado do dataset
                data_summary = {
                    "shape": df.shape,
                    "colunas": list(df.columns),
                    "colunas_numericas": numeric_cols,
                    "colunas_categoricas": cat_cols,
                    "stats_numericas": stats_numeric,
                    "stats_categoricas": stats_categorical,
                    "missing_values": df.isnull().sum().to_dict()
                }

                # Construir prompt para o LLM
                previous_messages = memory.load_memory_variables({}).get("chat_history", "")
                prompt = f"""
Você é um especialista em análise de dados. Aqui está o resumo do dataset:

{json.dumps(data_summary, ensure_ascii=False)}

Instruções:
1️⃣ Responda com conclusões concretas sobre os dados: tendências, correlações, outliers, padrões, variabilidade.
2️⃣ Se a pergunta pedir visualização explícita, gere um gráfico. Tipos possíveis: "histogram", "scatter", "boxplot".
3️⃣ Sempre retorne um JSON válido com esta estrutura:
   - Para gráfico:
   {{
       "action": "plot",
       "plot_type": "histogram" ou "scatter" ou "boxplot",
       "column": "nome_da_coluna",
       "y_column": "nome_da_coluna_secundaria" (opcional),
       "description": "descrição do gráfico"
   }}
   - Para texto:
   {{
       "action": "text",
       "text": "sua resposta detalhada"
   }}
4️⃣ Nunca invente gráficos; gere apenas quando explicitamente solicitado.
5️⃣ Responda em português.

Pergunta do usuário: "{user_question}"
"""

                if previous_messages:
                    prompt = f"{previous_messages}\n{prompt}"

                # Chamada ao LLM
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                )

                llm_response = completion.choices[0].message.content

                # Interpretar JSON retornado
                try:
                    response_json = json.loads(llm_response)
                except Exception:
                    response_json = {"action": "text", "text": llm_response}

                # Atualizar memória
                memory.save_context(
                    {"input": user_question},
                    {"output": llm_response}
                )

                # Executar ação
                if response_json.get("action") == "plot":
                    plot_type = response_json.get("plot_type")
                    col = response_json.get("column")
                    y_col = response_json.get("y_column")
                    description = response_json.get("description", col)

                    plt.figure(figsize=(8, 4))
                    if plot_type == "histogram" and col in df.columns:
                        sns.histplot(df[col], kde=True)
                        plt.title(description)
                        st.pyplot(plt)
                    elif plot_type == "scatter" and col in df.columns and y_col in df.columns:
                        sns.scatterplot(x=df[col], y=df[y_col])
                        plt.title(description)
                        st.pyplot(plt)
                    elif plot_type == "boxplot" and col in df.columns:
                        sns.boxplot(y=df[col])
                        plt.title(description)
                        st.pyplot(plt)
                    else:
                        st.write("Não foi possível gerar o gráfico solicitado.")
                else:
                    st.markdown("### 🤖 Resposta do Agente:")
                    st.write(response_json.get("text", llm_response))

            except Exception as e:
                st.error(f"Erro ao chamar o LLM: {e}")
