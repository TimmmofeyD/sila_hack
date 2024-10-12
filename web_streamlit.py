import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from fpdf import FPDF
from ultralytics import YOLO

model_load = YOLO('best.pt')

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


def apply_model(test_image_path):
    global class_label
    st.session_state['model_applied'] = True
    st.session_state['rows'] += 1

    image = cv2.imread(test_image_path)
    results = model_load(test_image_path, conf=0.05)

    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        confidence = result.conf[0].item()
        class_id = int(result.cls[0].item())

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # добавляем метку класса
        class_label = class_id
        cv2.putText(image, f"{class_label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    2)

    output_path = './pathtophotos'
    cv2.imwrite(output_path, image)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"имя класса {class_label}")
    plt.show()


# Функция для изменения состояния кнопки сохранения изменений
def toggle_edit_save(index, text):
    st.session_state[f'editing_{index}'] = False
    st.write(text)


def display_buttons(index, text):
    if index == 0:
        col1, col2 = st.columns(2)
        with col1:
            st.button(f'Сохранить изменения', key=f'edit_{index}', on_click=toggle_edit_save(index, text),
                      args=(index,))
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
    txt = st.text_area('Описание', key='text',
                       value='Test_text, aaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaa')
    display_buttons(i, txt)

footer = """<style>
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
st.markdown(footer, unsafe_allow_html=True)
