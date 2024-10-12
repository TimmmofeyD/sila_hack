import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# Загрузка модели
model_load = YOLO('best.pt')

# Словарь для отображения класса в текст
class_mapping = {
    0: 'scratches',
    1: 'wrong pixels',
    2: 'keyboard defects',
    3: 'lock',
    4: 'crews trubles',
    5: 'chipped',
}

# Инициализация состояний
if 'model_applied' not in st.session_state:
    st.session_state['model_applied'] = False
if 'rows' not in st.session_state:
    st.session_state['rows'] = 0

# Функция для применения модели и отрисовки только самых уверенных bbox
def apply_model(image):
    st.session_state['model_applied'] = True
    st.session_state['rows'] += 1

    # Применение модели YOLO
    results = model_load(image, conf = 0.05)

    # Получаем результаты предсказаний для изображения
    boxes = results[0].boxes.data.cpu().numpy()  # Получаем bbox: [x1, y1, x2, y2, score, class]

    # Находим самый уверенный bbox для каждого класса
    best_boxes = []
    class_max_scores = {}

    for box in boxes:
        score = box[4]
        cls = int(box[5])  # индекс класса

        if cls not in class_max_scores or score > class_max_scores[cls]:
            class_max_scores[cls] = score
            if cls in class_max_scores:
                best_boxes = [b for b in best_boxes if int(b[5]) != cls]  # удаляем предыдущий bbox этого класса
            best_boxes.append(box)

    # Копируем изображение для отрисовки bbox
    annotated_image = image.copy()

    # Отрисовываем bbox на изображении
    for box in best_boxes:
        x1, y1, x2, y2, score, cls = box
        cls_label = class_mapping.get(int(cls), 'неизвестный класс')  # Получаем текстовое значение класса
        annotated_image = cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        annotated_image = cv2.putText(annotated_image, f'{cls_label}: {score:.2f}', (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated_image

# Конфигурация страницы
st.set_page_config(page_title="Классификация дефектов ноутбуков", page_icon="💻")
st.title("Классификация дефектов ноутбуков на базе ИИ")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение ноутбука (png, jpg, jpeg):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Преобразование загруженного файла в изображение
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Кнопка для применения модели
    if st.button("Применить модель"):
        # Применение модели к изображению
        annotated_image = apply_model(image_np)

        # Вывод изображения с наложенной разметкой
        st.image(annotated_image, caption="Изображение с разметкой", use_column_width=True)

# Footer
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
