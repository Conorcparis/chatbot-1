import streamlit as st
import pandas as pd
import openai
import json

# Path to your JSON file
DATA_PATH = "/workspaces/chatbot-1/Reduced_Client_Data_25_Clients.json"

# Path to your logo
LOGO_PATH = "/workspaces/chatbot-1/Direct-Distribution-Logo new.jpg"  # Update with the correct path

# Load JSON data
@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load entire JSON file
        return pd.DataFrame(data)
    except json.JSONDecodeError as e:
        st.error(f"Error loading JSON data: {e}")
        return pd.DataFrame()

# Query OpenAI using the latest API
def query_openai(prompt, api_key):
    if not api_key:
        raise ValueError("OpenAI API key is missing.")
    
    openai.api_key = api_key
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message['content']
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while querying OpenAI: {e}")
# Streamlit UI
st.set_page_config(page_title="Sales Geni AI", layout="wide")

# Display the logo at the top of the page
st.image(LOGO_PATH, width=200)  # Adjust width as needed
st.title("📊 Sales Geni AI")
st.markdown("Ask questions about your sales data and get actionable insights.")

# Sidebar for configuration
st.sidebar.image(LOGO_PATH, width=150)  # Add the logo to the sidebar as well
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Main interaction
try:
    # Load sales data
    data = load_data(DATA_PATH)
    if data.empty:
        st.warning("The sales data could not be loaded. Please check the file path and try again.")
    else:
        # Show sample data
        st.sidebar.markdown("### Data Preview")
        rows_to_display = st.sidebar.slider("Rows to display:", min_value=1, max_value=min(len(data), 100), value=5)
        st.sidebar.write(data.head(rows_to_display))

        # User query
        st.subheader("Ask the Assistant")
        user_query = st.text_area("Enter your question:", placeholder="E.g., 'What are the sales trends for Q1 2023?'")

        if st.button("Get Insights"):
            if user_query.strip():
                with st.spinner("Generating insights..."):
                    # Create a summary of the data for the AI
                    sales_data_summary = data.describe().to_string()
                    prompt = (
                        f"Based on the following sales data summary:\n\n{sales_data_summary}\n\n"
                        f"Answer the user's question: {user_query}"
                    )
                    try:
                        response = query_openai(prompt, openai_api_key)
                        st.success("Here are your insights:")
                        st.write(response)
                    except ValueError as ve:
                        st.error(ve)
                    except Exception as e:
                        st.error(f"An error occurred while querying OpenAI: {e}")
            else:
                st.warning("Please enter a question before clicking 'Get Insights'.")
except FileNotFoundError:
    st.error(f"Could not find the data file at: {DATA_PATH}. Please ensure the file exists and try again.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("💡 *Powered by OpenAI and Streamlit*")
