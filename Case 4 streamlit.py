#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st


# In[ ]:


def intro():
    import streamlit as st
    import pandas as pd
    import geopandas


    st.write("# Case 4 – Maken van een dashboard")
#     st.sidebar.success("Selecteer een pagina.")

    st.markdown("""
    Streamlit is een open-source app framework wat specifiek is gemaakt voor
    Machine Learning en Data Science projecten.
    In dit project is een data-analyse gedaan over de verandering van de levensverwachting over de hele wereld
    en over verschillende invloeden hierop.
    
    Voor dit project is gebruik gemaakt van meerdere datasets:
    
    Een dataset van kaggle gebruikt. Deze dataset is ingeladen m.b.v. een API.
    Deze dataset is te vinden via de volgende link: https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated
    """)
    
    # API en data inladen
    code_API = """
    # data inladen via API
    !kaggle datasets download -d lashagoch/life-expectancy-who-updated
    !unzip life-expectancy-who-updated.zip
  
    # Data inladen m.b.v. csv
    pd.set_option('display.max_columns', None)
    life_exp = pd.read_csv('Life-Expectancy-Data-Updated.csv')"""
    
    pd.set_option('display.max_columns', None)
    life_exp = pd.read_csv('Life-Expectancy-Data-Updated.csv')
    life_exp_head = life_exp.head()
    life_exp_rijen = life_exp.shape[0]
    life_exp_kolom = life_exp.shape[1]

    st.code(code_API, language = 'python')
    st.write("De dataset ziet er nu als volgt uit:", life_exp_head, "De dataset bestaat nu uit ",
             life_exp_rijen, " rijen en ", life_exp_kolommen, " aantal_kolommen.")
    
    st.markdown("""
    Ook is gewerkt met een geopandas dataset die gaat over landen over de hele wereld.
    Deze is als volgt ingeladen:
    """)
    
    # Geodata inladen
    code_geo = """
    # geodata over de landen inladen
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))"""
    
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world_head = world.head()
    world_rijen = world.shape[0]
    world_kolom = world.shape[1]

    st.code(code_geo, language = 'python')
    st.write("De dataset ziet er nu als volgt uit:", world_head, "De dataset bestaat nu uit ",
             world_rijen, " rijen en ", world_kolommen, " aantal_kolommen.")
    
    st.markdown("""
    Deze 2 datasets zijn samengevoegd, maar eerst zijn hiervoor de waarden in de kolommen 'name' uit het geopandas
    dataframe en de waarden in de kolom 'Country' gelijk gemaakt. Hoe de datasets zijn samengevoegd is hieronder te zien:
    """)
    
    # Geodata inladen
    code_df = """
    df = life_exp.merge(world, left_on = 'Country', right_on = 'name', how = 'left')"""
    
    df = life_exp.merge(world, left_on = 'Country', right_on = 'name', how = 'left')
    df_head = df.head()
    df_rijen = df.shape[0]
    df_kolom = df.shape[1]

    st.code(code_df, language = 'python')
    st.write("De dataset ziet er nu als volgt uit:", df_head, "De dataset bestaat nu uit ",
             df_rijen, " rijen en ", df_kolommen, " aantal_kolommen.")
    
    st.markdown("""
    Om vervolgens meer informatie over het project te lezen
    
    **👈 Selecteer dan een keuze uit de balk hiernaast**.""")


# In[ ]:


def grafieken():
    import streamlit as st
    
    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    st.write("""
        Op deze pagina zijn grafieken te vinden van situaties die team 10 graag wilde onderzoeken.
        Heeft de hoeveelheid dagelijkse alcoholgebruik invloed op het uiteindelijke cijfer? Halen leerlingen minder hoge
        cijfers ze een langere reistijd hebben naar school? Dit zijn enkele vragen die beantwoord worden op deze pagina.
        Eerst zullen twee grafieken worden getoond met informatie over beide vakken.
        Vervolgens kan voor verschillende grafieken informatie gekregen worden over een specifiek vak.
        
        **👈 Hiervoor kan een keuze worden gemaakt in de balk hiernaast**""")


# In[ ]:


page_names_to_funcs = {
    "Opdrachtomschrijving": intro,
    "Grafieken": grafieken,
}

demo_name = st.sidebar.selectbox("Kies een pagina", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

