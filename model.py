from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import warnings
warnings.filterwarnings("ignore")

model = load_model("CNN.h5")

def sample():

    df_test = pd.read_csv("data/test.csv")

    sample = df_test.sample(1)

    sample = sample.values.reshape(-1, 28, 28, 1) / 255.0
    return sample


def first_pred(sample):
    col1, col2 = st.columns(2)


    with col1:

        if st.button('Generation'):
            st.image(sample, use_column_width="always")
            with col2:
                    st.header(f"Chiffre predit : {model.predict(sample).argmax()}")

    if st.button("True"):
        st.balloons()

    if st.button("False"):
        st.error("FAUX", icon="🚨")



def draw():
    if 'stat' not in st.session_state:
        st.session_state.stat = []
    if 'count' not in st.session_state:
        st.session_state.count = 0
    SIZE = 400
    col1, col2, col3 = st.columns(3)

    with col1:
        canvas_result = st_canvas(
            fill_color='#ffffff',
            stroke_width=30,
            stroke_color='#ffffff',
            background_color='#000000',
            height=400,width=400,
            drawing_mode='freedraw',
            key='canvas'
        )

    with col2:
        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28,28))
            img_rescaling = cv2.resize(img, (SIZE, SIZE))
            st.write('input image')
            st.image(img_rescaling)

    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model.predict(test_x.reshape(-1, 28, 28, 1) / 255.0)

    with col3:
        st.write(f'result: {np.argmax(pred[0])}')
        if st.button("True"):
            st.session_state.count += 1
            st.session_state.stat.append(1)
        if st.button("False"):
            st.session_state.count += 1
            st.session_state.stat.append(0)

        if st.session_state.count > 0:
            percent = sum(st.session_state.stat) / st.session_state.count * 100
            st.text(f"Le taux de bonne prediction est de {percent:.2f} %")
            st.text(f"Nombre d'iteration : {st.session_state.count}")

        if st.button("Réinitialisation"):
            st.session_state.stat = []
            st.session_state.count = 0

    st.bar_chart(pred[0])

    successive_outputs = [layer.output for layer in model.layers[0:]]
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
    test = ((test_x).reshape((-1,28,28,1)))/255.0
    successive_feature_maps = visualization_model.predict(test)
    layer_names = [layer.name for layer in model.layers]
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
                n_features = feature_map.shape[-1]
                size = feature_map.shape[ 1]
                display_grid = np.zeros((size, size * n_features))
                for i in range(n_features):
                        x  = feature_map[-1, :, :, i]
                        x -= x.mean()
                        x /= x.std ()
                        x *=  64
                        x += 128
                        x  = np.clip(x, 0, 255).astype('uint8')
                        display_grid[:, i * size : (i + 1) * size] = x
                        scale = 20. / n_features
                fig = plt.figure( figsize=(scale * n_features, scale) )
                plt.title ( layer_name )
                plt.grid  ( False )
                plt.imshow( display_grid, aspect='auto', cmap='viridis' )
                st.pyplot(fig)
