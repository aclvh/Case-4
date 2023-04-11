#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
st.set_page_config(layout = 'wide')


# In[ ]:


def intro():
    import streamlit as st
    import pandas as pd
    import geopandas
    
    st.write("# Case 4 – Maken van een dashboard")
#     st.sidebar.success("Selecteer een pagina.")

    st.markdown("""
    In dit project is een data-analyse gedaan over de verandering van de levensverwachting over de hele wereld
    en over verschillende invloeden hierop.
    
    Voor dit project is gebruik gemaakt van meerdere datasets:""")
    
    st.markdown("""
    **Levensverwachting dataset**
    
    Een dataset van kaggle gebruikt die gaat over de levensverwachting in landen over de hele wereld door de jaren heen 
    (2000 t/m 2015).
    Deze dataset is ingeladen m.b.v. een API en is te vinden via de volgende link:
    
    https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated""")
    
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
             life_exp_rijen, " rijen en ", life_exp_kolom, " aantal_kolommen.")
    
    st.markdown("""
    **Wereld dataset**
    
    Ook is gewerkt met een geopandas dataset die gaat over landen over de hele wereld.
    Deze is als volgt ingeladen:""")
    
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
             world_rijen, " rijen en ", world_kolom, " aantal_kolommen.")
    
    st.markdown("""
    ** Datasets samenvoegen**
    
    Deze 2 datasets zijn samengevoegd, maar eerst zijn hiervoor de waarden in de kolommen 'name' uit het geopandas
    dataframe en de waarden in de kolom 'Country' gelijk gemaakt. Hoe de datasets zijn samengevoegd is hieronder te zien:
    """)
    
    # Geodata inladen
    code_df = """
    df = life_exp.merge(world, left_on = 'Country', right_on = 'name', how = 'left')"""
    
    df = pd.read_csv('df.csv')
    df_head = df.head()
    df_rijen = df.shape[0]
    df_kolom = df.shape[1]

    st.code(code_df, language = 'python')
    st.write("De dataset ziet er nu als volgt uit:", df_head, "De dataset bestaat nu uit ",
             df_rijen, " rijen en ", df_kolom, " aantal_kolommen.")


# In[ ]:


def grafieken():
    import streamlit as st
    import geopandas
    import pandas as pd
    import folium
    from streamlit_folium import st_folium
    import plotly.express as px
    
#     import matplotlib.pyplot as plt
#     import seaborn as sns

    ###################################################################################################################

    # Datasets inladen
    df = pd.read_csv('df.csv')
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    ###################################################################################################################
    
    st.markdown("""
    # Inzichtin data m.b.v. grafieken.
    Aan de hand van de data zijn verschillende ondervindingen gedaan. Deze zijn hieronder te lezen en te zien in
    verschillende plotjes.""")
    
    ###################################################################################################################
    
    st.markdown("""
    ## Levensverwachting over de hele wereld
    Als eerst is gekeken naar de levensverwachting die mensen hebben in verschillende landen over de jaren heen.""")
    
    jaren = ('2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
            '2011', '2012', '2013', '2014', '2015')
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Add the choicemenu to the first column
        jaar = col1.radio('Jaar', jaren)
    
    with col2:
        # Filter the data based on the selected year
        df_kaart = df[df['Year'] == int(jaar)].dropna()

        # Create the map and add it to the second column
        m = folium.Map(location = [0,0],
                       zoom_start = 10,
                       zoom_control = False,
                       min_zoom = 2,
                       max_zoom = 2,
                       tiles = 'openstreetmap')

        m.choropleth(geo_data = world,
                     name = 'geometry',
                     data = df_kaart,
                     columns = ['Country', 'Life_expectancy'],
                     key_on = 'feature.properties.name',
                     fill_color = 'YlGn',
                     fill_opacity = 0.75,
                     line_opacity = 0.5,
                     legend_name = 'Life expectancy')

        st_data = st_folium(m, width = 725, height = 500)
        
    ###################################################################################################################
    
    st.markdown("""
    ## Levensverwachting over de tijd per regio
    Hieronder is gekeken naar de levensverwachting door de jaren heen per regio.""")
    
    # Dataframe voor life expectancy per regio maken
    df_time = df[['Country', 'Region', 'Year', 'Life_expectancy']]

    # Dataframe sorteren per regio en jaar
    df_time.sort_values(by = ['Region', 'Year'], inplace = True)

    # Nieuwe kolom met gemiddelde levensverwachting per regio per jaar aanmaken
    df_time = df_time.groupby(['Region', 'Year'])['Life_expectancy'].mean().reset_index(name = 'Mean_life_expectancy')

    # Kolom datum toevoegen voor mooie plot
    df_time['Date'] = pd.to_datetime(df_time['Year'].astype(str) + '-01-01')

    fig = px.line(df_time,
                  y = 'Mean_life_expectancy',
                  x = 'Date',
                  color = 'Region')

    fig.update_layout(title = 'Gemiddelde levensverwachting per regio',
                      xaxis_title = 'Datum',
                      yaxis_title = 'Levensverwachting (in jaren)',
                      legend_title = 'Regio',
                      xaxis = dict(rangeslider = dict(visible = True)))

    st.plotly_chart(fig)
    
    st.markdown("""
    In deze grafiek is te zien dat over de jaren heen de levensverwachting over het algemeen is toegenomen.
    Wat opvalt is dat de levensverwachting het meest is toegenomen in Afrika.""")
    
    ###################################################################################################################
    
    st.markdown("""
    ## Levensverwachting per regio
    Hieronder is gekeken naar de verdeling van de levensverwachting per regio.""")
    
    col1, col2 = st.columns(2)

    with col1:

        # Boxplot levensverwachting per regio (dropdown)
        jaren = ('Algemeen', 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                    2011, 2012, 2013, 2014, 2015)
        InvoerJaar_1 = st.selectbox('Selecteer het vak', jaren, key='jaar1')

        if InvoerJaar_1 == 'Algemeen':
            df_jaar = df
        else:
            df_jaar = df[df['Year'] == InvoerJaar_1]

        fig_jaar = px.box(df_jaar,
                          x = 'Region',
                          y = 'Life_expectancy')

        fig_jaar.update_layout(title = 'Relatie tussen levensverwachting en de regio',
                               xaxis_title = 'Regio',
                               yaxis_title = 'Levensverwachting (in jaren)',
                               yaxis_range = [25, 100],
                               width = 650)

        fig_jaar

    with col2:
    
        # Boxplot levensverwachting per regio (dropdown)
        InvoerJaar_2 = st.selectbox('Selecteer het vak', jaren, key='jaar2')

        if InvoerJaar_2 == 'Algemeen':
            df_jaar = df
        else:
            df_jaar = df[df['Year'] == InvoerJaar_2]

        fig_jaar = px.box(df_jaar,
                          x = 'Region',
                          y = 'Life_expectancy')

        fig_jaar.update_layout(title = 'Relatie tussen levensverwachting en de regio',
                               xaxis_title = 'Regio',
                               yaxis_title = 'Levensverwachting (in jaren)',
                               yaxis_range = [25, 100],
                               width = 650)
        
        fig_jaar


# In[ ]:


page_names_to_funcs = {
    "Opdrachtomschrijving": intro,
    "Grafieken": grafieken,
}

demo_name = st.sidebar.selectbox("Kies een pagina", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

