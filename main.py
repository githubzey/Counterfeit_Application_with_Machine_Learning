import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
# Sklearn regression algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score , precision_score , recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
# Sklearn regression model evaluation functions
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


background = Image.open("data/logo.png")
st.image(background,use_column_width=True)

siteHeader = st.container()
choisis = st.container()
dataExploration = st.container()
modelTraining = st.container()
dataresults = st.container()

with siteHeader:
    st.markdown("<h1 style='text-align: center; color: black;'>Welcome to Counterfeit Test Application !</h5>", unsafe_allow_html=True) 
    #st.title('Welcome to Counterfeit Test Application !')   
    st.markdown("With this application you can identify counterfeit banknotes from their geometric dimensions.")
with choisis :
    
    st.sidebar.title('Choisis') #Sidebar navigation
    choisis = st.sidebar.radio('Select your choise:', ['Model Data Overview','Model Evaluation','Test Your Data'])

    
if choisis == "Model Data Overview" : 
    with dataExploration:
        st.header('Model Data Overview')
        #st.text('* I made this application for a project during my formation Data Analyst\n OpenClassrooms  and I used the dataset that given by the formation center.')
        #st.text('* You can download -the dataset that cleaned and pprepared for application-\n from my git hub account https://github.com/githubzey')
        st.markdown('* I made this application for a project during my formation Data Analyst OpenClassrooms  and I used the dataset that given by the formation center.')
        st.markdown('* You can download -the dataset that cleaned and prepared for application- and also a dataset for testing the application from my git hub account https://github.com/githubzey')
        st.write("")
        st.markdown("<h5 style='text-align: center; color: orange;'>First five rows of the data</h5>", unsafe_allow_html=True)    
        billets_clean = pd.read_csv("data/billets_clean.csv")
        billets_clean["is_genuine"]=billets_clean["is_genuine"].replace([True,False],[0,1])
        st.table(billets_clean.head())

        st.markdown("<h6 style='text-align: left; color: red;'>Pay attention please !</h6>", unsafe_allow_html=True) 
        #col1, col2 = st.columns(2)
        #col1.write("Vrai billet = 0")
        #col2.write ("Faux billet = 1")
        st.write("The banknote is genuine = 0")
        st.write ("The banknote is Not genuine = 1")
        bar_data = billets_clean["is_genuine"].value_counts()
        st.markdown("<h5 style='text-align: center; color: orange;'>The dispersion of the banknotes</h5>", unsafe_allow_html=True)
        st.bar_chart(bar_data)    

elif choisis == "Model Evaluation" :
    with modelTraining:
        st.subheader('Model Scores')
        st.write("")
        st.markdown("<h5 style='text-align: center; color: black;'>For this application we use the LogisticRegression Model</h5>", unsafe_allow_html=True)
        st.write("")
        #st.markdown("**For this application we use the LinearRegression Model**")
    # on définit les variables explicatives et à expliquer
        features=['height_right','margin_low', 'margin_up', 'length']
        billets_clean = pd.read_csv("data/billets_clean.csv")
        billets_clean["is_genuine"]=billets_clean["is_genuine"].replace([True,False],[0,1])
        X = billets_clean[features] # Features
        y = billets_clean.is_genuine # Target variable

    # split X and y en training et testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
    # Créatin de model
        logreg = LogisticRegression(random_state=3)

    # fit le model avec data
        logreg.fit(X_train, y_train)

        y_pred_train = logreg.predict(X_train)
        y_pred_test = logreg.predict(X_test)    
        ac_tr = round(accuracy_score(y_train, y_pred_train),2)
        ac_ts = accuracy_score(y_test, y_pred_test)
        st.write("LogisticRegression Accuracy Score pour train set = " ,ac_tr)
        st.write("LogisticRegression Accuracy Score pour test set = " ,ac_ts)

        st.set_option('deprecation.showPyplotGlobalUse', False)
      
        f1_score = round(f1_score(y_test, y_pred_test),3)
        precision_score = round(precision_score(y_test, y_pred_test),3)
        recall_score = round(recall_score(y_test, y_pred_test),3)
        st.write("precision_score  = " ,precision_score )
        st.write("recall_score = " ,recall_score)
        st.write("f1_score = " ,f1_score)

        report = classification_report(y_test, y_pred_test)

        st.subheader("Model Report")
        st.text(report)

        st.subheader("Confusion Matrix") 
        model = logreg
        plot_confusion_matrix(model, X_test, y_test)
        st.pyplot()

        st.subheader("ROC Curve") 
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()
        
else :
    # prédiction
        upload_file = st.sidebar.file_uploader('Upload your csv file containing dimensions')
        if upload_file is not None:
            billets_clean = pd.read_csv("data/billets_clean.csv")
            billets_clean["is_genuine"]=billets_clean["is_genuine"].replace([True,False],[0,1])
            features=['height_right','margin_low', 'margin_up', 'length']
            X = billets_clean[features] # Features
            y = billets_clean.is_genuine # Target variable
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
            logreg = LogisticRegression(random_state=3)
            logreg.fit(X_train, y_train)
            data = pd.read_csv(upload_file)
          
            data_test=data[features]
            y_pred = logreg.predict(data_test)
    
            # affichage les résultats
            #print(y_pred)
            resultat=data.copy()
            resultat["prediction_lr"]=y_pred
            probas = np.round(logreg.predict_proba(data_test),3)
            resultat['probas_faux'] = probas[:,1]
            result= []
            for i in y_pred:
                if i == 1 :
                    result.append("Faux billet")
                else :
                    result.append("Vrai billet")
            resultat["result"]=result
            #print(resultat)
            #print(resultat[["id","prediction_lr","result"]])
            #st.write(resultat)

            with dataresults:
                st.header('Results of your dataset')
                st.dataframe(resultat)
               
            st.subheader("You can download your results !")       
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(resultat)

            st.download_button(
            label="Download result as CSV",
            data=csv,
            file_name='result.csv',
            mime='text/csv',
            )
            st.write("")   
            st.markdown("<h5 style='text-align: center; color: orange;'>The dispersion of the banknotes</h5>", unsafe_allow_html=True)
            bar_data_result = resultat["result"].value_counts()
            st.bar_chart(bar_data_result)
        else :
          st.subheader("Please input your csv file with the button on the left side")
