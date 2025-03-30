import streamlit as st
from Instructions.Machine_Learning.ml_interview_q import ML_INTERVIEW_Q
from Utilities.html_element_creator import HTMLElementCreator
from Mappers.ml_mapper import SML_CLS_MAPPER, SML_REG_MAPPER, USML_ANADET_MAPPER, USML_CLUS_MAPPER, USML_DIMRED_MAPPER

from Instructions.Machine_Learning.Supervised.Classification.cls_evaluation_metrices import classification_evaluation_metrics
from Instructions.Machine_Learning.Supervised.Regression.reg_evaluation_metrices import regression_evaluation_metrices
from Instructions.Machine_Learning.Unsupervised.Clustering.clus_evaluation_metrices import clustering_evaluation_metrices


st.set_page_config(
    page_title="Machine Learning",
    page_icon="🤖",
    layout="wide",
)

html_creator = HTMLElementCreator()
# css_file_path = "Style/main.css"
# with open(css_file_path) as css:
#     st.markdown(f"<style>{css.read()}</style", unsafe_allow_html=True)

st.header("Classical Machine Learning")

ml_tabs = st.tabs(['Supervised', 'Unsupervised',
                  'Reinforcement', 'Interview Questions'])
try:
    with ml_tabs[0]:
        st.header("Supervised Learning")
        st.markdown("In supervised learning, the model is trained on a labeled dataset, which means that the input data is paired with the correct output. The goal is to learn a mapping from inputs to outputs.")
        sl_tabs = st.tabs(['Classification', 'Regression'])
        with st.columns([0.05, 0.95])[1]:

            with sl_tabs[0]:
                st.header("Classification")
                st.markdown(
                    'Classification involves predicting categorical labels for new instances based on learned patterns from a labeled dataset.')
                sl_cl_tabs = st.tabs(['Evaluation Metrices', 'Algorithms'])
                with sl_cl_tabs[0]:
                    st.markdown(classification_evaluation_metrics)
                with sl_cl_tabs[1]:
                    for index, (title, value) in enumerate(SML_CLS_MAPPER.items()):

                        with st.expander(label=title, expanded=False):
                            st.header(title)
                            html_creator.create_youtube_link_button(
                                label=title, url=value[2])
                            with st.columns([0.6, 0.4])[0]:
                                st.image(image=value[1])
                            st.markdown(value[0])

            with sl_tabs[1]:
                st.header("Regression")
                st.markdown("Regression predicts continuous numerical values based on input features. The model learns the relationship between the input variables and a continuous output variable")
                sl_reg_tabs = st.tabs(['Evaluation Metrices', 'Algorithms'])
                with sl_reg_tabs[0]:
                    st.markdown(regression_evaluation_metrices)
                
                with sl_reg_tabs[1]:
                    for index, (title, value) in enumerate(SML_REG_MAPPER.items()):
                        with st.expander(label=title, expanded=False):
                            st.header(title)
                            html_creator.create_youtube_link_button(
                                label=title, url=value[2])
                            with st.columns([0.6, 0.4])[0]:
                                st.image(image=value[1])
                            st.markdown(value[0])

    with ml_tabs[1]:
        st.header("Unsupervised Learning")
        with st.columns([0.05, 0.95])[1]:
            st.markdown("Unsupervised Learning is a type of machine learning where the algorithm learns from unlabeled data. Unlike supervised learning,which requires labeled data to train the model, unsupervised learning finds patterns, structures, or relationships within the data itself.")
            usl_tabs = st.tabs(['Clustering','Dimensionality Reduction','Anomaly Detection'])
            with usl_tabs[0]:
                st.header("Clustering")
                st.markdown(
                    'Clustering is an unsupervised learning technique that groups similar data points together based on their features. The goal is to identify distinct groups (clusters) in the data without prior labels.')
                if st.toggle("Evaluation Metrices for Clustering"):
                    st.markdown(clustering_evaluation_metrices)
                for index, (title, value) in enumerate(USML_CLUS_MAPPER.items()):
                    with st.expander(label=title, expanded=False):
                        st.header(title)
                        html_creator.create_youtube_link_button(
                            label=title, url=value[2])
                        with st.columns([0.6, 0.4])[0]:
                            st.image(image=value[1])
                        st.markdown(value[0])

            with usl_tabs[1]:
                st.header("Dimensionality Reduction")
                st.markdown("Dimensionality reduction reduces the number of features (dimensions) in a dataset while retaining essential information. This is useful for simplifying data, visualizing high-dimensional data, and improving computational efficiency. ")
                for index, (title, value) in enumerate(USML_DIMRED_MAPPER.items()):
                    with st.expander(label=title, expanded=False):
                        st.header(title)
                        html_creator.create_youtube_link_button(
                            label=title, url=value[2])
                        with st.columns([0.6, 0.4])[0]:
                            st.image(image=value[1])
                        st.markdown(value[0])

            with usl_tabs[2]:
                st.header("Anomaly Detection")
                st.markdown("Anomaly detection identifies rare or unusual observations in a dataset that deviate significantly from the norm. This technique is valuable for spotting outliers that may indicate fraud, errors, or other significant events. ")
                for index, (title, value) in enumerate(USML_ANADET_MAPPER.items()):
                    with st.expander(label=title, expanded=False):
                        st.header(title)
                        html_creator.create_youtube_link_button(
                            label=title, url=value[2])
                        with st.columns([0.6, 0.4])[0]:
                            st.image(image=value[1])
                        st.markdown(value[0])

    with ml_tabs[2]:
        st.header("Reinforcement Learning")
        with st.columns([0.05, 0.95])[1]:
            st.markdown("Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions, and its goal is to maximize its cumulative reward over time.")
    with ml_tabs[3]:
        st.image("./Instructions/Machine_Learning/ml_int.png")
        st.markdown(ML_INTERVIEW_Q)

except Exception as e:
    st.markdown("# Check Instruction File")
    st.error(e)
