import streamlit as st
from Mappers.gen_ai_mapper import GENAI_MAPPER
from Utilities.html_element_creator import HTMLElementCreator

st.set_page_config(
    page_title="Generative AI",
    page_icon="💬",
    layout="wide",
)

html_creator = HTMLElementCreator()
css_file_path = "Style/main.css"
with open(css_file_path) as css:
    st.markdown(f"<style>{css.read()}</style", unsafe_allow_html=True)


try:
    st.markdown("## Generative AI")
    gen_ai_tabs = st.tabs(GENAI_MAPPER.keys())
    for index, (title, value) in enumerate(GENAI_MAPPER.items()):
        with gen_ai_tabs[index]:
            st.image(value[1])
            st.markdown(value[0])


except Exception:
    st.markdown("# Check Instruction File")
