import streamlit as st
from Mappers.dl_mapper import ANN_MAPPER, CNN_MAPPER, DL_CONCEPTS_MAPPER
from Utilities.html_element_creator import HTMLElementCreator


st.set_page_config(layout="wide")

html_creator = HTMLElementCreator()
css_file_path = "Style/main.css"
with open(css_file_path) as css:
    st.markdown(f"<style>{css.read()}</style", unsafe_allow_html=True)

st.header("Deep Learning")
st.divider()

try:
    st.header("Deep Learning Concepts")
    with st.columns([0.05, 0.95])[1]:
        st.markdown("Common DL Concepts")
        for index, (title, value) in enumerate(DL_CONCEPTS_MAPPER.items()):
            with st.expander(label=title, expanded=False):
                st.header(title)
                html_creator.create_youtube_link_button(
                    label=title, url=value[2])
                with st.columns([0.7, 0.3])[0]:
                    if len(value[1]) >= 1:
                        for image_path in value[1]:
                            st.image(image=image_path,
                                     caption=(image_path.split('/')[-1].split(".")[0]).capitalize())
                st.markdown(value[0])

    # 1
    st.header("Artificial Neural Networks (ANN)")
    with st.columns([0.05, 0.95])[1]:
        st.markdown("An Artificial Neural Network (ANN) is a computational model inspired by the structure of the human brain, composed of layers of interconnected nodes called neurons. Each neuron processes input, applies weights, passes the result through an activation function, and produces an output. ANNs are used in a wide range of applications, such as classification, regression, and pattern recognition tasks.")
        for index, (title, value) in enumerate(ANN_MAPPER.items()):
            with st.expander(label=title, expanded=False):
                st.header(title)
                html_creator.create_youtube_link_button(
                    label=title, url=value[2])
                with st.columns([0.6, 0.4])[0]:
                    st.image(image=value[1])
                st.markdown(value[0])

   # 2
    st.header("Convolutional Neural Network (CNN)")
    with st.columns([0.05, 0.95])[1]:
        st.markdown("A Convolutional Neural Network (CNN) is a type of ANN primarily used for image processing tasks. It employs convolutional layers that apply filters to input data to detect patterns like edges or textures, followed by pooling layers to reduce the dimensionality. CNNs are highly effective for tasks like image recognition, object detection, and visual data analysis due to their ability to capture spatial features.")
        for index, (title, value) in enumerate(CNN_MAPPER.items()):
            with st.expander(label=title, expanded=False):
                st.header(title)
                html_creator.create_youtube_link_button(
                    label=title, url=value[2])
                with st.columns([0.6, 0.4])[0]:
                    st.image(image=value[1])
                st.markdown(value[0])

    # 2
    st.header("Recurrent Neural Network (RNN)")
    with st.columns([0.05, 0.95])[1]:
        st.markdown("A Recurrent Neural Network (RNN) is a class of neural networks designed to process sequential data by retaining memory of previous inputs through recurrent connections. This allows RNNs to capture temporal dependencies, making them ideal for tasks like natural language processing, speech recognition, and time series analysis, where the order of inputs matters.")
        for index, (title, value) in enumerate(ANN_MAPPER.items()):
            with st.expander(label=title, expanded=False):
                st.header(title)
                html_creator.create_youtube_link_button(
                    label=title, url=value[2])
                with st.columns([0.6, 0.4])[0]:
                    st.image(image=value[1])
                st.markdown(value[0])

except Exception as e:
    st.markdown("# Check Instruction File")
    st.error(e)
