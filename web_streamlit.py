import streamlit as st
import pandas as pd
import numpy as np
import time
from fpdf import FPDF

# Инициализация состояний
if 'model_applied' not in st.session_state:
    st.session_state['model_applied'] = False
if 'rows' not in st.session_state:
    st.session_state['rows'] = 0
if 'editing' not in st.session_state:
    st.session_state['editing'] = {}

def generate_pdf():
    """Generate an example pdf file and save it to example.pdf"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Welcome to Streamlit!", ln=1, align="C")
    pdf.output("example.pdf")

def apply_model():
    st.session_state['model_applied'] = True
    st.session_state['rows'] += 1

# Функция для изменения состояния кнопки сохранения изменений
def toggle_edit_save(index, text):
    st.session_state[f'editing_{index}'] = False
    st.write(text)


def display_buttons(index, text):
    if index == 0:
        col1, col2 = st.columns(2)
        with col1:
            st.button(f'Сохранить изменения', key=f'edit_{index}', on_click=toggle_edit_save(index, text), args=(index,))
        with col2:
            col1, col2 = st.columns(2)
            with col1:
                format = st.selectbox('Формат отчёта', ('pdf', 'txt'))

            with col2:
                st.download_button(label='Сгенерировать отчёт', data=text, file_name=f'Отчёт.{format}')



st.set_page_config(page_title="Классификация дефектов ноутбуков",
                           page_icon="💻")
st.title("Классификация дефектов ноутбуков на базе ИИ")
uploaded_files = st.file_uploader("Загрузите изображения ноутбука в форматах png | jpg | jpeg:",
                                 type=["png", "jpg", "jpeg"],
                                 accept_multiple_files=True)


for file in uploaded_files:
    st.image(file)

# Кнопка для применения модели
if st.button(f"Применить модель", key='apply_model'):
    apply_model()


# Отображение кнопок после применения модели
for i in range(st.session_state['rows']):
    txt = st.text_area('Описание', key='text', value='Test_text, aaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaa')
    display_buttons(i, txt)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://vk.com/korol.shamanov" target="_blank">Mirea Team</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)