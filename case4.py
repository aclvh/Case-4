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
    
    st.markdown("")
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
    
    st.markdown("")
    st.markdown("""
    #### Datasets samenvoegen
    
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
    import statsmodels.api as sm
    
#     import matplotlib.pyplot as plt
#     import seaborn as sns

    ###################################################################################################################
    # Datasets inladen
    df = pd.read_csv('df.csv')
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    ###################################################################################################################
    # Eerste stuk tekst pagina
    
    st.markdown("""
    # Inzichtin data m.b.v. grafieken.
    Aan de hand van de data zijn verschillende ondervindingen gedaan. Deze zijn hieronder te lezen en te zien in
    verschillende plotjes.""")
    
    ###################################################################################################################
    # Kaart met levensverwachting over de hele wereld
    
    st.markdown("""
    ## Levensverwachting over de hele wereld
    Als eerst is gekeken naar de levensverwachting die mensen hebben in verschillende landen over de jaren heen.""")
    
    jaren = ('2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
            '2011', '2012', '2013', '2014', '2015')
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
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
    # Lijndiagram levensverwachting over de tijd per regio
    
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
    # Boxplots voor verdeling levensverwachting in het algemeen of voor specifieke jaren
    
    st.markdown("""
    ## Levensverwachting per regio
    Hieronder is gekeken naar de verdeling van de levensverwachting per regio.""")
    
    col1, col2 = st.columns(2)

    with col1:

        # Eerste boxplot voor vergelijken
        # Keuzemenu
        jaren = ('Algemeen', 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                    2011, 2012, 2013, 2014, 2015)
        InvoerJaar_1 = st.selectbox('Selecteer het jaar', jaren, key = 'Algemeen')

        if InvoerJaar_1 == 'Algemeen':
            df_jaar = df
        else:
            df_jaar = df.loc[df['Year'] == InvoerJaar_1].copy()
        
        # Stuk code hieronder is zodat de x-as volgorde voor iedere keuze gelijk blijft
        regio_volgorde = ['Asia', 'Africa', 'Middle East', 'European Union', 'Rest of Europe', 'North America',
                  'South America', 'Central America and Caribbean', 'Oceania']

        df_jaar['Region'] = pd.Categorical(df_jaar['Region'],
                                           categories = regio_volgorde)

        fig_jaar = px.box(df_jaar,
                          x = 'Region',
                          y = 'Life_expectancy',
                          category_orders = {'Region': regio_volgorde})

        fig_jaar.update_layout(title = 'Relatie tussen levensverwachting en de regio',
                               xaxis_title = 'Regio',
                               yaxis_title = 'Levensverwachting (in jaren)',
                               yaxis_range = [25, 100],
                               width = 650)

        fig_jaar

    with col2:
    
        # Tweede boxplot (voor vergelijken)
        # Keuzemenu
        InvoerJaar_2 = st.selectbox('Selecteer het jaar', jaren, key = 2001)

        if InvoerJaar_2 == 'Algemeen':
            df_jaar = df
        else:
            df_jaar = df.loc[df['Year'] == InvoerJaar_2].copy()

        # Stuk code hieronder is zodat de x-as volgorde voor iedere keuze gelijk blijft
        regio_volgorde = ['Asia', 'Africa', 'Middle East', 'European Union', 'Rest of Europe', 'North America',
                          'South America', 'Central America and Caribbean', 'Oceania']

        df_jaar['Region'] = pd.Categorical(df_jaar['Region'],
                                           categories = regio_volgorde)
        
        fig_jaar = px.box(df_jaar,
                          x = 'Region',
                          y = 'Life_expectancy',
                          category_orders={'Region': regio_volgorde})

        fig_jaar.update_layout(title = 'Relatie tussen levensverwachting en de regio',
                               xaxis_title = 'Regio',
                               yaxis_title = 'Levensverwachting (in jaren)',
                               yaxis_range = [25, 100],
                               width = 650)
        
        fig_jaar
        
    ###################################################################################################################
    # (Kijken of de stijging van de levensverwachting misschien komt door de inentingen)
    # Kijken of de minder hoge levensverwachting in Africa bijvoorbeeld komt doordat er minder wordt ingeënt
    
    st.markdown("""
    Om te onderzoeken of de minder hoge levensverwachting in Africa komt omdat er evenuteel minder ingeënt wordt,
    is per regio het gemiddelde percentage inentingen (vanaf 1 jaar oud) per ziekte berekend.""")
    
    df_im = df[['Region', 'Polio', 'Diphtheria', 'Measles', 'Hepatitis_B']]

    df_polio = df_im.assign(gem_polio = df_im.groupby('Region')['Polio'].transform('mean'))
    df_polio = df_polio[['Region', 'gem_polio']]
    df_dipth = df_im.assign(gem_dipth = df_im.groupby('Region')['Diphtheria'].transform('mean'))
    df_dipth = df_dipth[['Region', 'gem_dipth']]
    df_measl = df_im.assign(gem_measl = df_im.groupby('Region')['Measles'].transform('mean'))
    df_measl = df_measl[['Region', 'gem_measl']]
    df_hepat = df_im.assign(gem_hepat = df_im.groupby('Region')['Hepatitis_B'].transform('mean'))
    df_hepat = df_hepat[['Region', 'gem_hepat']]

    df_im_1 = df_polio.merge(df_dipth, on = 'Region')
    df_im_1 = df_im_1.drop_duplicates()
    df_im_1.head()

    df_im_2 = df_measl.merge(df_hepat, on = 'Region')
    df_im_2 = df_im_2.drop_duplicates()
    df_im_2.head()

    df_im_tot = df_im_1.merge(df_im_2, on = 'Region')
    
    col1, col2 = st.columns(2)

    with col1:
        # plot polio
        fig_polio = px.histogram(df_im_tot,
                         y = 'gem_polio',
                         x = 'Region',
                         category_orders = {'Region': regio_volgorde})

        fig_polio.update_layout(title = 'Gemiddeld percentage ingeënt tegen Polio per regio',
                                xaxis_title = 'Regio',
                                yaxis_title = 'Percentage',
                                yaxis_range = [0,100])

        fig_polio
        
        # Plot dipth
        fig_dipth = px.histogram(df_im_tot,
                         y = 'gem_dipth',
                         x = 'Region', 
                         category_orders = {'Region': regio_volgorde})

        fig_dipth.update_layout(title = 'Gemiddeld percentage ingeënt tegen dipth per regio',
                                xaxis_title = 'Regio',
                                yaxis_title = 'Percentage', 
                                yaxis_range = [0,100])

        fig_dipth
        
    with col2:
        # Plot measl
        fig_measl = px.histogram(df_im_tot,
                         y = 'gem_measl',
                         x = 'Region',
                         category_orders = {'Region': regio_volgorde})

        fig_measl.update_layout(title = 'Gemiddeld percentage ingeënt tegen measl per regio',
                                xaxis_title = 'Regio',
                                yaxis_title = 'Percentage', 
                                yaxis_range = [0,100])

        fig_measl
        
        # Plot hepat
        fig_hepat = px.histogram(df_im_tot,
                         y = 'gem_hepat',
                         x = 'Region',
                         category_orders = {'Region': regio_volgorde})

        fig_hepat.update_layout(title = 'Gemiddeld percentage ingeënt tegen hepat per regio',
                                xaxis_title = 'Regio',
                                yaxis_title = 'Percentage',
                                yaxis_range = [0,100])

        fig_hepat
        
    
    ###################################################################################################################
    # Plot inentingen in Afrika over de jaren heen
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("")
        st.markdown("""
            In afrika is inderdaad het minste percentage ingeënt per ziekte gemiddeld gezien over de jaren heen.
            Nu gaan we controleren of dit percentage inderdaad over de jaren heen ook is gestegen in plaats van alleen naar
            het gemiddelde kijken over de jaren heen.

            In de plot hier rechts weergegeven is inderdaad te zien dat er een sterke stijging is geweest in het percentage
            wat is ingeënt tegen hepatitis B, Polio en Diptheria. Bij de inentignen tegen de mazelen is deze stijging echter
            niet te zien.

            Het is waarschijnlijk om aan te nemen dat de levensverwachting in Afrika is toegenomen door de stijging in het 
            percentage mensen dat wordt ingeënt tegen deze ziektes.

            Het is dus ook waarschijnlijk om aan te nemen dat de levensverwachting in alle landen toeneemt, naarmate een 
            groter percentage van de bevolking is ingeënt tegen deze ziektes.""")
        
    with col2:
        # Dataframe met alleen het gebied Afrika
        df_afrika = df[df['Region'] == 'Africa']

        # Dataframe voor life expectancy per regio maken
        df_afrika = df_afrika[['Country', 'Region', 'Year', 'Hepatitis_B', 'Measles', 'Polio', 'Diphtheria']]

        # Dataframe sorteren per regio en jaar
        df_afrika.sort_values(by = ['Region', 'Year'], inplace = True)

        # Nieuwe kolom met gemiddelde percentage ingeënt per ziekte maken
        df_afrika_hepa = df_afrika.groupby(['Year'])['Hepatitis_B'].mean().reset_index(name = 'Mean')
        df_afrika_hepa['Inenting'] = 'Hepatitits_B'

        df_afrika_meas = df_afrika.groupby(['Year'])['Measles'].mean().reset_index(name = 'Mean')
        df_afrika_meas['Inenting'] = 'Measles'

        df_afrika_polio = df_afrika.groupby(['Year'])['Polio'].mean().reset_index(name = 'Mean')
        df_afrika_polio['Inenting'] = 'Polio'

        df_afrika_dipth = df_afrika.groupby(['Year'])['Diphtheria'].mean().reset_index(name = 'Mean')
        df_afrika_dipth['Inenting'] = 'Diphtheria'

        # Nieuwe dataframes per ziekte samenvoegen
        df_inenting_1 = pd.merge(df_afrika_hepa, df_afrika_meas, how = 'outer')
        df_inenting_2 = pd.merge(df_afrika_polio, df_afrika_dipth, how = 'outer')
        df_inenting = pd.merge(df_inenting_1, df_inenting_2, how = 'outer')

        # Kolom datum toevoegen voor mooie plot
        df_inenting['Date'] = pd.to_datetime(df_time['Year'].astype(str) + '-01-01')
        
        fig_inenting_afrika = px.line(df_inenting,
                                      y = 'Mean',
                                      x = 'Date',
                                      color = 'Inenting')

        fig_inenting_afrika.update_layout(title = 'Gemiddelde percentage ingeënt per ziekte',
                                          xaxis_title = 'Datum',
                                          yaxis_title = 'Percentage',
                                          legend_title = 'Inenting',
                                          xaxis = dict(rangeslider = dict(visible = True)))

        fig_inenting_afrika
        
    ###################################################################################################################
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("")
        st.markdown("""
        Om te onderzoeken of de levensverwachting inderdaad toeneemt, wanneer een hoger percentage van de bevolking is 
        ingeënt tegen verschillende ziekten is een scatterplot gemaakt.
        
        Het percentage wat is gebruikt is het gemiddelde van de percentages i....
        
        In de scatterplot is inderdaad te zien, dat wanneer het percentage wat ingeënt is stijgt de levensverwachting
        ook toeneemt.""")
        
    with col2:
        df_inenting = df[['Hepatitis_B', 'Measles', 'Polio', 'Diphtheria', 'Life_expectancy']]
        df_inenting['gem_percentage_ingeënt'] = df_inenting[['Hepatitis_B', 'Measles', 'Polio', 'Diphtheria']].mean(axis=1)

        # keuze ziekte
        fig = px.scatter(df_inenting,
                         x = 'gem_percentage_ingeënt',
                         y = 'Life_expectancy',
                         trendline = 'ols',
                         trendline_color_override = 'red')

        fig.update_layout(title = "Regressie tussen het gemiddelde percentage ingeënt en de levensverwachting",
                                 xaxis_title = "Gemiddeld percentage ingeënt voor verschillende ziekten",
                                 yaxis_title = "Levensverwachting in jaren")

        fig
        
    ###################################################################################################################
    col1, col2 = st.columns(2)

    with col1:
    
        regio_keuze = ('Asia', 'Africa', 'Middle East', 'European Union', 'Rest of Europe', 'North America',
                      'South America', 'Central America and Caribbean', 'Oceania')

        InvoerRegio = st.selectbox('Selecteer het jaar', regio_keuze, key = 'Asia')

        df_hist = df[df['Region'] == InvoerRegio]

        fig_BMI = px.histogram(df_hist,
                               x = 'BMI')

        if InvoerRegio == 'Asia':
            fig_BMI.add_vline(x = 23,
                              line_dash = 'dash',
                              line_color = 'firebrick')
        else:
            fig_BMI.add_vline(x = 25,
                              line_dash = 'dash',
                              line_color = 'firebrick')

        fig_BMI.update_layout(title = 'Verdeling BMI',
                              xaxis_title = 'BMI',
                              yaxis_title = 'Aantal',
                              xaxis_range = [19, 32.5])

        fig_BMI
    
    with col2:
        st.markdown("""
        In de plot is te zien dat er inderdaad meer mensen in ... zijn met een gezond BMI...""")


# In[ ]:


page_names_to_funcs = {
    "Opdrachtomschrijving": intro,
    "Grafieken": grafieken,
}

demo_name = st.sidebar.selectbox("Kies een pagina", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

