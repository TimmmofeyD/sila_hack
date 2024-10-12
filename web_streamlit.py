import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import shutil

# Если в сессии нет датасета, создаем пустой DataFrame
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = pd.DataFrame(
        columns=['main_class', 'additional_info', 'filename', 'x_left_bottom', 'y_left_bottom', 'length', 'width'])

# Словарь классов для аннотаций
class_mapping = {
    'царапины': 0,
    'битые пиксели': 1,
    'проблемы с клавишами': 2,
    'замок': 3,
    'отсутствует шуруп': 4,
    'сколы': 5,
    'без дефектов': 6
}

# Загрузка изображений для аннотаций
uploaded_files = st.file_uploader("Загрузите изображения ноутбуков для аннотации (png, jpg, jpeg):",
                                  accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Форма для аннотаций
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Изображение: {uploaded_file.name}", use_column_width=True)

        with st.form(key=f"annotation_form_{uploaded_file.name}"):
            # Поля для ввода данных аннотаций (класс и координаты бокса)
            selected_class = st.selectbox('Выберите класс', list(class_mapping.keys()),
                                          key=f"class_{uploaded_file.name}")
            additional_info = st.text_input('Описание дефекта', key=f"info_{uploaded_file.name}")
            x_left_bottom = st.number_input('Координата X левого нижнего угла', min_value=0,
                                            key=f"x_{uploaded_file.name}")
            y_left_bottom = st.number_input('Координата Y левого нижнего угла', min_value=0,
                                            key=f"y_{uploaded_file.name}")
            width = st.number_input('Ширина', min_value=1, key=f"width_{uploaded_file.name}")
            length = st.number_input('Длина', min_value=1, key=f"length_{uploaded_file.name}")

            if st.form_submit_button('Сохранить разметку'):
                # Добавляем строку с новой аннотацией в DataFrame
                new_row = {
                    'main_class': selected_class,
                    'additional_info': additional_info,
                    'filename': uploaded_file.name,
                    'x_left_bottom': x_left_bottom,
                    'y_left_bottom': y_left_bottom,
                    'length': length,
                    'width': width
                }
                st.session_state['dataset'] = pd.concat([st.session_state['dataset'], pd.DataFrame([new_row])], ignore_index=True)

# Отображаем таблицу с аннотациями и кнопку для скачивания CSV
if not st.session_state['dataset'].empty:
    st.write(st.session_state['dataset'])
    st.download_button(
        label="Скачать разметку",
        data=st.session_state['dataset'].to_csv(index=False).encode('utf-8'),
        file_name="annotations.csv",
        mime="text/csv"
    )


# Предобработка и аугментация изображений
def crop_and_augment(image, bbox, target_size=640):
    x_left_bottom, y_left_bottom, width, length = bbox

    img_width, img_height = image.size

    if any(np.isnan([x_left_bottom, y_left_bottom, width, length])):
        crop_x = (img_width - target_size) // 2
        crop_y = (img_height - target_size) // 2

        crop_x = min(max(crop_x, 0), img_width - target_size)
        crop_y = min(max(crop_y, 0), img_height - target_size)

        cropped_image = image.crop((crop_x, crop_y, crop_x + target_size, crop_y + target_size))
        return cropped_image, None

    img_width, img_height = image.size

    crop_x = min(max(int(x_left_bottom - (target_size - width) // 2), 0), img_width - target_size)
    crop_y = min(max(int(y_left_bottom - (target_size - length) // 2), 0), img_height - target_size)

    cropped_image = image.crop((crop_x, crop_y, crop_x + target_size, crop_y + target_size))

    new_x_left_bottom = x_left_bottom - crop_x
    new_y_left_bottom = y_left_bottom - crop_y
    new_bbox = [new_x_left_bottom, new_y_left_bottom, width, length]

    return cropped_image, new_bbox


# Путь для хранения обрезанных и аугментированных изображений
output_zip_path = '/kaggle/working/dataset.zip'
image_folder = '/kaggle/input/sila-dataset/data'

# Если аннотации есть и пользователь хочет дообучить модель
if st.button("Дообучить модель"):
    df = st.session_state['dataset']

    # Структура папок для изображений и аннотаций
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(f'/kaggle/working/{split}', folder), exist_ok=True)
        for class_name in class_mapping.keys():
            os.makedirs(os.path.join(f'/kaggle/working/{split}/images', class_name), exist_ok=True)
            os.makedirs(os.path.join(f'/kaggle/working/{split}/labels', class_name), exist_ok=True)

    for index, row in df.iterrows():
        filename = row['filename']
        x_left_bottom = row['x_left_bottom']
        y_left_bottom = row['y_left_bottom']
        width = row['width']
        length = row['length']
        main_class = row['main_class']

        bbox = [x_left_bottom, y_left_bottom, width, length]
        image_path = os.path.join(image_folder, filename)

        class_folder = main_class.lower()
        split = 'train' if np.random.rand() > 0.2 else 'val'

        class_image_folder = os.path.join(f'/kaggle/working/{split}/images', class_folder)
        class_annotation_folder = os.path.join(f'/kaggle/working/{split}/labels', class_folder)

        saved_image_path = os.path.join(class_image_folder, filename)
        annotation_path = os.path.join(class_annotation_folder,
                                       filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.JPG',
                                                                                                         '.txt'))

        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                aug_img, new_bbox = crop_and_augment(img, bbox, target_size=640)
                aug_img.save(saved_image_path)

                with open(annotation_path, 'w') as f:
                    if new_bbox is None:  # сохраняем только метку класса
                        f.write(f"{class_mapping[main_class]}\n")
                    else:
                        image_width, image_height = aug_img.size
                        x_center = (new_bbox[0] + new_bbox[2] / 2) / image_width
                        y_center = (new_bbox[1] + new_bbox[3] / 2) / image_height
                        norm_width = new_bbox[2] / image_width
                        norm_height = new_bbox[3] / image_height

                        f.write(f"{class_mapping[main_class]} {x_center} {y_center} {norm_width} {norm_height}\n")
        else:
            print(f"Файл не найден: {image_path}")

    #shutil.make_archive('/kaggle/working/dataset', 'zip', '/kaggle/working')

    # Дообучение модели YOLO
    data_yaml_path = './data.yaml'
    epochs = 50
    img_size = 640
    batch = 8

    model = YOLO('best.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Запуск дообучения
    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=img_size, batch=batch, augment=True,
                          hsv_h=0.7, hsv_s=0.7, hsv_v=0.7, flipud=0.7, fliplr=0.7, mosaic=1.0, mixup=0.7)

    st.write("Модель успешно дообучена!")
    st.write(f"Архив с изображениями и аннотациями создан по пути: {output_zip_path}")

# Загрузка изображений для детекции
detection_files = st.file_uploader("Загрузите изображения для детекции (png, jpg, jpeg):", accept_multiple_files=True,
                                   type=["png", "jpg", "jpeg"], key="detection_uploader")

# Отображение результатов детекции
if detection_files:
    model = YOLO('best.pt')  # Загружаем предобученную модель

    for detection_file in detection_files:
        image = Image.open(detection_file)
        results = model(image)

        # Отображение результатов
        st.image(results[0].plot(), caption=f"Детекция для {detection_file.name}", use_column_width=True)
