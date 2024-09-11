import streamlit as st
from app_helper import llm_db_chain
import requests


def get_flipkart_logo():
    """Fetches the Amazon logo from the web."""
    url = "https://1000logos.net/wp-content/uploads/2021/02/Flipkart-logo.png"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return response.content
    else:
        return None


logo_data = get_flipkart_logo()

if logo_data:
    st.logo(logo_data, )
else:
    st.error('Failed to fetch Flipkart logo.')

st.title("Flipkart: Database Q&A 📦")

question = st.text_input("Question: ")

if question:
    try:
        chain = llm_db_chain()
        input_dict = {"query": question}
        response = chain.invoke(input_dict)
        # response = chain.run(question)
        st.header("Answer")
        st.write(response['result'])
    except Exception as e:
        st.error(f"Error: {str(e)}")
