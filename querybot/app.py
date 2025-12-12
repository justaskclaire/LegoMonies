
import logging
import os
import sqlite3

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lego_querybot")

# === Setup ===
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client configured from environment variables.")
else:
    client = None
    logger.error("OPENAI_API_KEY is not set. API calls will fail until it is configured.")

DB_PATH = os.getenv(
    "BRICKSET_DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "BricksetDB.db")),
)
logger.info("Using database at %s", DB_PATH)

# === Helpers ===
def get_table_schema():
    if not os.path.exists(DB_PATH):
        message = f"Database not found at {DB_PATH}. Update BRICKSET_DB_PATH to a valid file."
        logger.error(message)
        raise FileNotFoundError(message)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(SetList);")
        return [row[1] for row in cursor.fetchall()]

def generate_sql(natural_language, columns):
    if client is None:
        raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY and restart the app.")

    system_prompt = f"""
You are a helpful assistant that converts natural language into SQL queries.
The table is named SetList and has the following columns: {', '.join(columns)}.
Do NOT use backticks or brackets. Only use standard SQL.
Only SELECT queries. Never write UPDATE, DELETE, or INSERT.
"""
    try:
        logger.info("Translating user input to SQL.")
        logger.debug("Natural language input: %s", natural_language)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate this to SQL:\n{natural_language}"}
            ]
        )
        sql = response.choices[0].message.content.strip()
        logger.info("Generated SQL: %s", sql)
        return sql
    except Exception as exc:
        logger.exception("Failed to generate SQL from user input.")
        raise RuntimeError(
            "I couldn't translate that request into SQL. Please verify your OpenAI API key and try again."
        ) from exc

def run_query(sql):
    try:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"Database not found at {DB_PATH}. Please set BRICKSET_DB_PATH to the correct location."
            )

        logger.info("Executing SQL query against the database.")
        logger.debug("SQL query: %s", sql)
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(sql, conn)
    except Exception as e:
        logger.exception("Database query failed.")
        return pd.DataFrame({
            "error": [
                "Unable to run the SQL query. "
                "Check that your database path is correct, the SQL only references columns in SetList, "
                f"and that the query is valid. Details: {e}"
            ]
        })

def generate_recommendation(user_prompt):
    if client is None:
        raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY and restart the app.")

    # Build the dataset by translating the user's prompt into SQL
    try:
        columns = get_table_schema()
        sql = generate_sql(user_prompt, columns)
        logger.info("Generated SQL for recommendations: %s", sql)
        df = run_query(sql)
        if "error" in df.columns:
            raise RuntimeError(df.iloc[0]["error"])
    except Exception as exc:
        logger.exception("Failed to build dataset for recommendations.")
        raise RuntimeError("Failed to build dataset for recommendations.") from exc

    def format_row(row):
        return f"{row['SetName']} ({row['Theme']} - {row['Subtheme']}), {row['Pieces']} pcs, {row['USRetailPrice']}, released in {row['YearFrom']}, {row['Availability']}"

    # Limit to ~300 rows to avoid token limits
    summaries = [format_row(row) for _, row in df.head(300).iterrows()]
    data_blob = "\n".join(summaries)

    system_prompt = """You are a LEGO set expert who makes personalized recommendations based on user preferences and the available data.
Use the dataset provided to suggest specific sets and explain why they fit the user's request.
Only recommend sets from the dataset shown."""

    try:
        logger.info("Generating recommendation for user prompt.")
        logger.debug("Recommendation prompt: %s", user_prompt)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\nHere is a set dataset you can work with:\n{data_blob}"}
            ]
        )

        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.exception("Failed to generate recommendation from OpenAI.")
        raise RuntimeError(
            "I couldn't generate a recommendation. Please confirm your OpenAI API key and try again."
        ) from exc

# === Streamlit UI ===

st.title("🧱 LEGO Set Querybot")

# --- SQL Query Box ---
user_input = st.text_input("Ask a question about your LEGO sets:", "")

if user_input:
    with st.spinner("Translating to SQL..."):
        try:
            columns = get_table_schema()
            sql = generate_sql(user_input, columns)
            st.code(sql, language="sql")

            results = run_query(sql)
            if "error" in results.columns:
                st.error(results.iloc[0]["error"])
            else:
                st.write(results)
                if not results.empty:
                    st.success("Query executed successfully!")
                else:
                    st.warning("No results found for that query. Try different filters or wording.")
        except Exception as exc:
            st.error(str(exc))

# --- Recommendation Box ---
st.markdown("---")
st.subheader("🤖 Ask for LEGO advice")

recommendation_prompt = st.text_input("Ask something like: 'I liked the Insect Collection. What else might I enjoy?'")

if recommendation_prompt:
    with st.spinner("Thinking really hard about bricks..."):
        try:
            recommendation = generate_recommendation(recommendation_prompt)
            st.markdown("### GPT Recommends:")
            st.markdown(recommendation.replace("*", ""), unsafe_allow_html=True)
        except Exception as exc:
            st.error(str(exc))