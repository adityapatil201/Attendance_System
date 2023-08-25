import os
from tensorflow.keras.applications.vgg16 import preprocess_input
import streamlit as st
from PIL import Image
from datetime import date
import uuid
from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

detector=MTCNN()
model = load_model('resnet50_NotopLayer_244x244x3_AvgPooling.h5')


def get_name(face_embeddings, embedding_list):
    max_sim = 0
    name = ' '

    for i in embedding_list:

        k = cosine_similarity(face_embeddings.reshape(1, -1), list(i.values())[0].reshape(1,-1))  # .reshape(1,-1) converts the one dimensional face_embeddings and features[0] (of shape (2048,)) to shape(1,2048) as the cosine_similarity function takes only 2d inputs

        if k[0][0] > max_sim:
            max_sim = k[0][0]
            name = list(i.keys())[0]

    return name


st.title('Attendance System')

pkl_file_path = st.text_input("Enter the path of .pkl file:")
excel_sheet_path = st.text_input("Enter the path of excel file:")

embedding_list=pickle.load(open(pkl_file_path ,'rb'))

# Read the Excel file into a DataFrame
data_df = pd.read_excel(excel_sheet_path, sheet_name='Sheet1',engine='openpyxl')

current_date = date.today()
today=str(current_date)

uploaded_files = st.file_uploader("Choose several images",
                                 accept_multiple_files=True,
                                 type=['png', 'jpg', 'jpeg'])

st.dataframe(data_df)


if uploaded_files is not None:

    directory_path=os.path.join('uploaded_imgs','Class1',today)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    
    
    if today not in data_df.columns:
        data_df[today]=0
            

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        

        unique_id = str(uuid.uuid4())+'.png'

        img.save(os.path.join(directory_path, unique_id))

        sample_image_array = cv2.imread(os.path.join(directory_path, unique_id))
        photo_info = detector.detect_faces(sample_image_array)
        for i in photo_info:
            x, y, width, height = i['box']
            face = sample_image_array[y:y + height, x:x + width]

            face_image = Image.fromarray(face)
            face_image = face_image.resize((244, 244))

            face_array = np.asarray(face_image)

            face_array = face_array.astype('float32')

            expanded_face_array = np.expand_dims(face_array, axis=0)

            preprocessed_face_array = preprocess_input(expanded_face_array)

            face_embeddings = model.predict(preprocessed_face_array).flatten()

            face_name=get_name(face_embeddings, embedding_list)
            
            data_df[today][data_df.Students==face_name] = 1
        
    st.dataframe(data_df)   

st.markdown("<h2 style='text-align: left; color: white;'>List of Absent students with their photos</h2>", unsafe_allow_html=True)  
for i in data_df[data_df[today]==0].Students:
    st.text(i)

    if st.button(f"Show Photo of {i}"):
        
        img_name=str(i)+'.jpg'
        
        st.image(os.path.join('Students_images_class1',img_name))
        
        

st.text("")
st.text("")
st.text("")
st.markdown("<h2 style='text-align: left; color: white;'>Mention students which are present but marked absent:</h2>", unsafe_allow_html=True)
names = st.text_input("")
#names = st.text_input("Mention students which are present but marked absent:")
names_list = names.split(",")
for i in range(0,len(names_list)):
    names_list[i] = names_list[i].strip()

if st.button('Add Attendance'):    
    
    for name in names_list:
      
        if name in list(data_df.Students):
            data_df[today][data_df.Students==name] = 1
            st.text(f"Added {name}'s Attendance")
        else:
            st.text(f'Name {name} not in the Sheet')
            
            
    data_df.to_excel(excel_sheet_path, index=False, engine='openpyxl', sheet_name='Sheet1')

    st.dataframe(data_df) 
    

  
            
        
        
        
        
        
        
        
        
        
        
        
        
