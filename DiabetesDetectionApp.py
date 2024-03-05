#!/usr/bin/env python
# coding: utf-8

# # Diabetes Detection App

# In[89]:


# import pip
# pip.main(["install","streamlit"])
# pip.main(["install","PIL"])


# In[90]:


# !pip install Pillow


# In[91]:


import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[92]:


st.header("Diabetes Detection App")


# In[93]:


image=Image.open("E:\\From Desktop\\Data Science\\DataScience Project\\diab.jpg")


# In[94]:


st.image(image)


# In[95]:


data=pd.read_csv("E:\\From Desktop\\Data Science\\DataScience Project\\diabetes.csv")


# In[96]:


data.head()


# In[97]:


#st.subheader("Data")


# In[98]:


#st.dataframe(data)


# In[99]:


#st.subheader("Data Description")


# In[100]:


#st.write(data.iloc[:,:8].describe())


# In[ ]:





# Machine Learning part

# In[101]:


x=data.iloc[:,:8].values
y=data.iloc[:,8].values


# In[102]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[103]:


model=RandomForestClassifier(n_estimators=500)


# In[104]:


model.fit(x_train,y_train)


# In[105]:


y_pred=model.predict(x_test)


# In[106]:


#st.subheader("Accuracy of train model")


# In[107]:


#st.write(accuracy_score(y_test,y_pred))


# In[ ]:





# text box 

# In[108]:


# st.text_input(label="Enter your age: ")


# In[109]:


# st.text_area(label="Describe you")


# In[110]:


# st.slider("Set your age",0,100,0)


# In[ ]:





# In[111]:


st.subheader("Enter Your Input Data")


# In[112]:


def user_inputs():
    preg=st.slider("Pregnancy",0,20,0)
    glu=st.slider("Glucose",0,200,0)
    blpre=st.slider("Blood Pressure",0,130,0)
    skinth=st.slider("Skin Thickness",0,100,0)
    ins=st.slider("Insulin",0.0,1000.0,0.0)
    bmi=st.slider("BMI",0.0,70.0,0.0)
    diabtest=st.slider("DPF",0.000,3.000,0.0)
    age=st.slider("Age",0,100,0)

    input_dict={"Pregnancies":preg,
                "Glucose":glu,
                "Blood Pressure":blpre,
                "Skin Thickness":skinth,
                "Insulin":ins,
                "BMI":bmi,
                "DPF":diabtest,
                "Age":age}
    
    return pd.DataFrame(input_dict,index=["User Input Values"])


# In[113]:


user_input=user_inputs()


# In[114]:


st.subheader("Entered Input Data")


# In[115]:


st.write(user_input)


# In[116]:


st.subheader("Predictions (0-Non Diabetes,1-Diabetes)")


# In[117]:


# st.write(model.predict(user_input))


# In[118]:


st.write(model.predict(user_input))

