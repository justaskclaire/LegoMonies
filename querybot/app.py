import os
import sqlite3
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# === Setup ===
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
DB_PATH = "C:/PersonalProjects/LegoMonies/BricksetDB.db"

# === Helpers ===
def get_table_schema():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(SetList);")
        return [row[1] for row in cursor.fetchall()]

def generate_sql(natural_language, columns):
    system_prompt = f"""
You are a helpful assistant that converts natural language into SQL queries.
The table is named SetList and has the following columns: {', '.join(columns)}.
Do NOT use backticks or brackets. Only use standard SQL.
Only SELECT queries. Never write UPDATE, DELETE, or INSERT.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Translate this to SQL:\n{natural_language}"}
        ]
    )
    return response.choices[0].message.content.strip()

def run_query(sql):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(sql, conn)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def generate_recommendation(user_prompt):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT SetName, Theme, Subtheme, Pieces, USRetailPrice AS RRP, YearFrom, Availability
            FROM SetList
            WHERE Pieces > 50 AND USRetailPrice IS NOT NULL
        """, conn)

    def format_row(row):
        return f"{row['SetName']} ({row['Theme']} - {row['Subtheme']}), {row['Pieces']} pcs, ${row['RRP']}, released in {row['YearFrom']}, {row['Availability']}"

    # Limit to ~300 rows to avoid token limits
    summaries = [format_row(row) for _, row in df.head(300).iterrows()]
    data_blob = "\n".join(summaries)

    system_prompt = """You are a LEGO set expert who makes personalized recommendations based on user preferences and the available data.
Use the dataset provided to suggest specific sets and explain why they fit the user's request.
Only recommend sets from the dataset shown."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\nHere is a set dataset you can work with:\n{data_blob}"}
        ]
    )

    return response.choices[0].message.content.strip()

# === Streamlit UI ===

st.title("🧱 LEGO Set Querybot")

# --- SQL Query Box ---
user_input = st.text_input("Ask a question about your LEGO sets:", "")

if user_input:
    with st.spinner("Translating to SQL..."):
        columns = get_table_schema()
        sql = generate_sql(user_input, columns)
        st.code(sql, language="sql")

        results = run_query(sql)
        st.write(results)

        if not results.empty:
            st.success("Query executed successfully!")
        else:
            st.warning("No results found or an error occurred.")

# --- Recommendation Box ---
st.markdown("---")
st.subheader("🤖 Ask for LEGO advice")

recommendation_prompt = st.text_input("Ask something like: 'I liked the Insect Collection. What else might I enjoy?'")

if recommendation_prompt:
    with st.spinner("Thinking really hard about bricks..."):
        recommendation = generate_recommendation(recommendation_prompt)
        st.markdown("### GPT Recommends:")
        st.markdown(recommendation.replace("*", ""), unsafe_allow_html=True)
