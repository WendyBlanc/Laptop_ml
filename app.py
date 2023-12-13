import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle


with st.sidebar :
    st.title("")
st.write("Projet de prediction")
pages = ["Context du projet", "Exploration de données", "Visualisation de donées","Model prediction"]

data = pd.read_csv("./laptops.csv", encoding='ISO-8859-1')
data_cleaned = pd.read_csv("./laptops_cleaned.csv").drop(columns=['Unnamed: 0'],axis=1)
st.sidebar.title("Sommaire")

page = st.sidebar.radio("", pages)

if page == pages[0]: 
    st.image("./laptop.png")
    st.write("## Context du projet")
    st.write("Ce projet s'inscrit dans un contexte de ventes d'ordinateur. L'objectif est de prédire le prix des ordinateurs et de découvrir le prix de l'ordinateur a partir de ses features. ")
    


elif page == pages[1]:
    st.write("## Exploration de données")
    st.write('')
    st.dataframe(data.head())
    st.write('Shape du dataframe :')
    st.write(data.shape)
    
    if st.checkbox("Afficher les statistiques descriptives"):
        st.write(data.describe())
        
    if st.checkbox("Afficher les doublons"):
        
        st.write(data.duplicated().sum())
    if st.checkbox("Afficher les valeurs null"):
        st.write(data.isna().sum())
    
elif page == pages[2]:
    st.write("## Visualisation de donées")
    
    
    percentage_counts = data_cleaned['Manufacturer'].value_counts(normalize=True) * 100
    st.title('Pourcentage par marque')

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=percentage_counts[:10].index, y=percentage_counts[:10].values, palette="viridis", ax=ax)

    plt.title('Réponses en pourcentage')
    plt.xticks(rotation=45)
    plt.xlabel('Marque')
    plt.ylabel('Pourcentage')
    st.pyplot(fig)
    
    limit = 5
    top_categories = percentage_counts[percentage_counts >= limit]
    autre_category = percentage_counts[percentage_counts < limit].sum()
    top_categories['Autre'] = autre_category
    fig, ax = plt.subplots()
    ax.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%', shadow=True, startangle=90)
    st.pyplot(fig)
    
    st.title('Prix des laptops par marque')
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=data_cleaned['Manufacturer'], y=data_cleaned['Price'], palette="viridis",ax=ax)
    plt.xticks(rotation="vertical")
    st.pyplot(fig)
    
    st.title("Corrélation entre les features")
    data_num = data_cleaned.select_dtypes(include=['float64', 'int64'])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(data_num.corr(), annot=True, cmap='coolwarm', square=True, ax=ax)
    st.pyplot(fig)
    
    
    
    
    
elif page == pages[3]:
    model = pickle.load(open("./knr_model.pkl", "rb"))
    
    option = st.selectbox('Choisir la marque',data_cleaned['Manufacturer'].unique())
    
    screen_Size = st.number_input('Screen Size')
    ram = st.selectbox('RAM',np.sort(data_cleaned['RAM'].unique() ))
    storage = st.number_input('Storage')
    weight = st.number_input('Weight')
    ssd = st.checkbox('SSD')
    hdd = st.checkbox('HDD')
    
    row = np.array([screen_Size,ram,storage,weight,ssd,hdd] )
    def predict():
        
        
    
        
        st.write(model.predict(row.reshape(1, -1))) 
        
        
    st.button("Prédiction", on_click=predict)

