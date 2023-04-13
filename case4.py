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
    
    st.write("# Case 4 – Dashboard levensverwachting")
#     st.sidebar.success("Selecteer een pagina.")

    st.markdown("""
    In dit dashboard is een analyse te zien over de levensverwachting van landen over de hele wereld. Voor de tot stand
    koming van dit dashboard is gebruik gemaakt van meerdere datasets:""")
    
    st.markdown("""
    **1. Levensverwachting dataset**
    
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
    st.write(life_exp_head, "De dataset bestaat nu uit ",
             life_exp_rijen, " rijen en ", life_exp_kolom, " aantal_kolommen.")
    
    st.markdown("")
    st.markdown("""
    **2. Wereld dataset**
    
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
    st.write(world_head, "De dataset bestaat nu uit ",
             world_rijen, " rijen en ", world_kolom, " aantal_kolommen.")
    
    st.markdown("""---""")
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
    import numpy as np
    import folium
    from streamlit_folium import st_folium
    import plotly.express as px
    import statsmodels.api as sm
    
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    import plotly.graph_objs as go
    import plotly.figure_factory as ff
    
#     import matplotlib.pyplot as plt
#     import seaborn as sns

    ###################################################################################################################
    # Datasets inladen
    df = pd.read_csv('df.csv')
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    
    ###################################################################################################################
    # Eerste stuk tekst pagina
    
    st.markdown("""
    # Inzicht in data met behulp van grafieken
    Aan de hand van de data zijn verschillende ondervindingen gedaan. Deze zijn hieronder te lezen en te zien in
    verschillende plotjes.""")
    
    
    ###################################################################################################################
    # Kaart met levensverwachting over de hele wereld
    
    st.markdown("""
    ## Levensverwachting op de kaart
    Als eerst is gekeken naar de levensverwachting die mensen hebben in verschillende landen over de jaren heen.
    In het keuzemenu hieronder kan worden bepaald voor welk jaar men deze gegevens wil bekijken.""")
    
    jaren = ('2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
            '2011', '2012', '2013', '2014', '2015')

    st.markdown("### Levensverwachting op de kaart")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        jaar = col1.radio('Jaar', jaren)
            
    with col2:
        # Filter the data based on the selected year
        df_kaart = df[df['Year'] == int(jaar)].dropna()

        # Kaart levensverwachting
        m1 = folium.Map(location = [0,0],
                        zoom_start = 10,
                        zoom_control = False,
                        min_zoom = 2,
                        max_zoom = 2,
                        tiles = 'openstreetmap')

        m1.choropleth(geo_data = world,
                      name = 'geometry',
                      data = df_kaart,
                      columns = ['Country', 'Life_expectancy'],
                      key_on = 'feature.properties.name',
                      fill_color = 'YlGn',
                      fill_opacity = 0.75,
                      line_opacity = 0.5,
                      legend_name = 'Life expectancy')

        st_data = st_folium(m1, width = 750, height = 500)
        
    st.markdown("### BBP per hoofd van de bevolking op de kaart")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:        
        st.markdown("""
        Naast het keuzemenu is te zien wat de gemiddelde levensverwachting is per land en of dit hoog/laag is 
        ten opzichte van andere landen.
        
        Wanneer een land zwart gekleurd is is geen informatie bekend over dit land.
        
        Wat opvalt is dat de levensverwachting in Amerika, Europa en Oceanië gemiddeld gezien hoog is en in Afrika de
        levensverwachting laag.
        
        Wanneer men dit vergelijkt met de kaart die gaat over het BBP per hoofd van de bevolking per land, dan valt
        het op dat op ongeveer dezelfde plekken het BBP per hoofd van de bevolking ook hoger is.""")

    with col2:
        # Kaart ontwikkelingsstatus maken
        m2 = folium.Map(location = [0,0],
                        zoom_start = 10,
                        zoom_control=False,
                        min_zoom = 2,
                        max_zoom = 2,
                        tiles = 'openstreetmap')

        # Choropleth gekozen jaar plotten
        m2.choropleth(geo_data = world,
                      name = 'geometry',
                      data = df_kaart,
                      columns = ['Country', 'GDP_per_capita'],
                      key_on = 'feature.properties.name',
                      fill_color = 'YlGn',
                      fill_opacity = 0.75,
                      line_opacity = 0.5,
                      legend_name = 'BBP per hoofd van de bevolking')
        
        mst_data = st_folium(m2, width = 750, height = 500)
    
        
    ###################################################################################################################
    # Verdeling levensverwachting
    
    st.markdown("""---""")
    st.markdown("## Verdeling levensverwachting")       
    
    col1, col2 = st.columns([1.5, 1])

    with col1:
        verdeling = px.histogram(df,
                             x = 'Life_expectancy')
        
        verdeling.update_traces(marker_color = '#7FD7A4')
        verdeling.update_layout(title = "Verdeling van de levensverwachting",
                                xaxis_title = "Levensverwachting",
                                yaxis_title = "Aantal")
        verdeling
    
    with col2:
        st.markdown("")
        st.markdown("""
        In de grafiek hier links weergegeven is te zien dat de meerderheid van de mensen ouder dan 65 lijkt te worden en
        dat de levensverwachting linksscheef verdeeld is.""")
    
    
    ###################################################################################################################
    # Lijndiagram levensverwachting over de tijd per regio
    
    st.markdown("## Levensverwachting over de tijd per regio")
    
    col1, col2 = st.columns([1.5, 1])

    with col1:
        # Dataframe voor life expectancy per regio maken
        df_time = df.loc[:, ['Country', 'Region', 'Year', 'Life_expectancy']].copy()
        
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
                          xaxis = dict(rangeslider = dict(visible = True)),
                          width = 800)
        fig
    
    with col2:
        st.markdown("")
        st.markdown("""
        In deze grafiek hier links weergegeven is te zien dat de levensverwachting over het algemeen is toegenomen.
        Wat opvalt is dat de levensverwachting het meest is toegenomen in Afrika.""")
    
    
    ###################################################################################################################
    # Boxplots voor verdeling levensverwachting in het algemeen of voor specifieke jaren
    
    st.markdown("## Levensverwachting per regio")
    st.markdown("""
    Hieronder is de verdeling van de levensverwachting per regio te zien. Deze kan in het algemeen worden bekeken,
    maar kan ook over verschillende jaren met elkaar worden vergeleken.""")
    
    # Volgorde waarin de regio genoemd zal worden in meerdere plots
    regio_volgorde = ['Azië', 'Afrika', 'Midden-Oosten', 'Europese Unie', 'Rest van Europa', 'Noord-Amerika',
                      'Zuid-Amerikca', 'Midden-Amerika en het Caribisch gebied', 'Oceanië']

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
        
        df_jaar['Region'] = pd.Categorical(df_jaar['Region'],
                                           categories = regio_volgorde)

        fig_jaar = px.box(df_jaar,
                          x = 'Region',
                          y = 'Life_expectancy',
                          category_orders = {'Region': regio_volgorde})

        fig_jaar.update_traces(marker_color = '#7FD7A4')
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

        df_jaar['Region'] = pd.Categorical(df_jaar['Region'],
                                           categories = regio_volgorde)
        
        fig_jaar = px.box(df_jaar,
                          x = 'Region',
                          y = 'Life_expectancy',
                          category_orders={'Region': regio_volgorde})

        fig_jaar.update_traces(marker_color = '#7FD7A4')
        fig_jaar.update_layout(title = 'Relatie tussen levensverwachting en de regio',
                               xaxis_title = 'Regio',
                               yaxis_title = 'Levensverwachting (in jaren)',
                               yaxis_range = [25, 100],
                               width = 650)
        fig_jaar
    
    st.markdown("""
    Uit deze grafieken kan worden geconcludeerd dat de levensverwachting over het algemeen is gestegen, maar dat vooral 
    in Azië en Afrika een duidelijk zichtbare stijging is geweest in de levensverwachting. Ook is de spreiding hier 
    minder geworden over de jaren heen.""")
        
    ###################################################################################################################
    # ### Percentage ingeënt per regio (dat voor 4 verschillende ziektes, dus 4 keer)
    
    # (Kijken of de stijging van de levensverwachting misschien komt door de inentingen)
    # Kijken of de minder hoge levensverwachting in Africa bijvoorbeeld komt doordat er minder wordt ingeënt
    
    st.markdown("""---""")
    st.markdown("""
    ## Inentingen
    ### Percentage inentingen per regio en ziekte""")
    
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

    df_im_2 = df_measl.merge(df_hepat, on = 'Region')
    df_im_2 = df_im_2.drop_duplicates()

    df_im_tot = df_im_1.merge(df_im_2, on = 'Region')
        
    # Dataframe omzetten naar een long format zodat er een goede histogram gemaakt kan worden
    df_long = pd.melt(df_im_tot,
                      id_vars = 'Region',
                      var_name = 'Inenting tegen',
                      value_name = 'percentage')

    df_long.replace(['gem_polio', 'gem_dipth', 'gem_measl', 'gem_hepat'],
                    ['Polio', 'Difterie', 'Mazelen', 'Hepatitis B'],
                    inplace = True)
    
    col1, col2 = st.columns(2)

    with col1:
        inentingen = px.histogram(df_long,
                                 x = 'Region',
                                 y = 'percentage',
                                 color = 'Inenting tegen',
                                 barmode = 'group',
                                 category_orders = {'Region': regio_volgorde},
                                 color_discrete_sequence = ['red', 'pink', '#7FD7A4', '#3776ab'])
#                                  color_discrete_sequence = ['green', 'lightgreen', '#3CB371', '#7FD7A4']
#                                  color_discrete_sequence = ['red', 'pink', '#3CB371', 'lightblue']

        inentingen.update_layout(title = 'Gemiddeld percentage inentingen tegen ziektes per regio',
                                 xaxis_title = 'Regio',
                                 yaxis_title = 'Percentage',
                                 width = 670)
        inentingen

    with col2:
        st.markdown("")
        st.markdown("""
        Om te onderzoeken of de minder hoge levensverwachting in Afrika komt omdat er evenuteel minder ingeënt wordt,
        is per regio het gemiddelde percentage inentingen per ziekte berekend.
        
        Ook is onderzocht of in andere regio's het gemiddelde percentage inentingen hoger is, wanneer in deze regio's ook
        een hogere levensverwachting is.
        
        In Afrika is inderdaad het gemiddeld gezien het minste percentage mensen ingeënt per ziekte over de jaren heen.""")
        
    
    ###################################################################################################################
    # Plot inentingen in Afrika over de jaren heen
    
    st.markdown("""
        ### Inentingen Afrika""")
    
    col1, col2 = st.columns(2)

    with col1:
        # Dataframe met alleen het gebied Afrika
        df_afrika = df[df['Region'] == 'Afrika']

        # Dataframe voor life expectancy per regio maken
        df_afrika = df_afrika[['Country', 'Region', 'Year', 'Hepatitis_B', 'Measles', 'Polio', 'Diphtheria']]

        # Dataframe sorteren per regio en jaar
        df_afrika.sort_values(by = ['Region', 'Year'], inplace = True)

        # Nieuwe kolom met gemiddelde percentage ingeënt per ziekte maken
        df_afrika_hepa = df_afrika.groupby(['Year'])['Hepatitis_B'].mean().reset_index(name = 'Mean')
        df_afrika_hepa['Inenting'] = 'Hepatitits B'

        df_afrika_meas = df_afrika.groupby(['Year'])['Measles'].mean().reset_index(name = 'Mean')
        df_afrika_meas['Inenting'] = 'Mazelen'

        df_afrika_polio = df_afrika.groupby(['Year'])['Polio'].mean().reset_index(name = 'Mean')
        df_afrika_polio['Inenting'] = 'Polio'

        df_afrika_dipth = df_afrika.groupby(['Year'])['Diphtheria'].mean().reset_index(name = 'Mean')
        df_afrika_dipth['Inenting'] = 'Difterie'

        # Nieuwe dataframes per ziekte samenvoegen
        df_inenting_1 = pd.merge(df_afrika_hepa, df_afrika_meas, how = 'outer')
        df_inenting_2 = pd.merge(df_afrika_polio, df_afrika_dipth, how = 'outer')
        df_inenting = pd.merge(df_inenting_1, df_inenting_2, how = 'outer')

        # Kolom datum toevoegen voor mooie plot
        df_inenting['Date'] = pd.to_datetime(df_time['Year'].astype(str) + '-01-01')

        fig = px.line(df_inenting,
                      y = 'Mean',
                      x = 'Date',
                      color = 'Inenting',
                      color_discrete_sequence = ['red', 'pink', '#7FD7A4', '#3776ab'])

        fig.update_layout(title = 'Gemiddelde percentage ingeënt per ziekte',
                          xaxis_title = 'Datum',
                          yaxis_title = 'Percentage',
                          legend_title = 'Inenting',
                          xaxis = dict(rangeslider = dict(visible = True)),
                          width = 670)
        fig

    with col2:
        st.markdown("")
        st.markdown("""
        
        Met behulp van deze grafiek wordt gecontroleerd of het aantal inentingen in Afrika over de jaren heen is
        gestegen.
        
        Uit de plot blijkt inderdaad dat er een sterke stijging is geweest in het percentage mensen wat is ingeënt 
        tegen hepatitis B, polio en difterie. Bij de inentignen tegen de mazelen is deze stijging echter niet te zien.

        Het is aannemelijk om aan te nemen dat de levensverwachting in Afrika is toegenomen door de stijging in het 
        percentage mensen dat wordt ingeënt tegen deze ziektes.""")
        
        
    ###################################################################################################################
    # Plot regressie tussen gemiddeld percentage ingeënd en levensverwachting
    st.markdown("""### Relatie tussen gemiddeld percentage ingeënt en levensverwachting""")
    
    col1, col2 = st.columns(2)

    with col1:
        gem_inenting = df.loc[:, ['Hepatitis_B', 'Measles', 'Polio', 'Diphtheria', 'Life_expectancy']].copy()
        gem_inenting['gem_percentage_ingeënt'] = gem_inenting[['Hepatitis_B', 'Measles', 'Polio', 'Diphtheria']].mean(axis=1)

        # keuze ziekte
        fig = px.scatter(gem_inenting,
                         x = 'gem_percentage_ingeënt',
                         y = 'Life_expectancy',
                         color_discrete_sequence=['#7FD7A4'],
                         trendline='ols',
                         trendline_color_override = 'red',
                         width = 670)

        fig.update_layout(title = "Relatie tussen het gemiddelde percentage ingeënt en de levensverwachting",
                                 xaxis_title = "Gemiddeld percentage ingeënt voor verschillende ziekten",
                                 yaxis_title = "Levensverwachting in jaren")
        fig

    with col2:
        st.markdown("")
        st.markdown("""
        Door de aanname die volgt uit de vorige plot zou men kunnen aannemen dat de levensverwacthing toeneemt 
        naarmate een groter percentage van de bevolking is ingeënt tegen deze ziektes.
        
        Om te onderzoeken of de levensverwachting inderdaad toeneemt, wanneer een hoger percentage van de bevolking is 
        ingeënt tegen verschillende ziekten is een scatterplot gemaakt.
        
        Het percentage wat is gebruikt is het gemiddelde inentingspercentage van alle vier de inentingen
        (per land en jaar).
        
        In de scatterplot is inderdaad te zien, dat wanneer het percentage wat ingeënt is stijgt, de levensverwachting
        ook toeneemt.""")
        
        
    ###################################################################################################################
    # Variabele sterfgevallen
    st.markdown("""---""")
    st.markdown("""
    ## Relatie tussen het aantal sterfgevallen en de levensverwachting
    Na het maken van een heatmap bleek dat nog drie variabelen een sterke relatie hadden met de variabele
    levensverwachting. Dit waren de variabelen die het volgende weergaven:
    * Sterfgevallen van zuigelingen per 1000 inwoners
    * Sterfgevallen van volwassenen per 1000 inwoners
    De lineaire relatie tussen deze drie variabelen en de levensverwachting wordt hieronder weergegeven:""")
    
    col1, col2 = st.columns(2)

    with col1:
        # Zuigelingen
        fig = px.scatter(df,
                         x = 'Infant_deaths',
                         y = 'Life_expectancy',
                         trendline = 'ols',
                         trendline_color_override = 'red',
                         color_discrete_sequence = ['#7FD7A4'])

        fig.update_layout(title = 'Relatie tussen het aantal zuigeling sterfgevallen en de levensverwachting',
                          xaxis_title = 'Aantal sterfgevallen (per 1000 inwoners)',
                          yaxis_title = 'Levensverwachting in jaren',
                          width = 670)
        fig
        
    with col2:
        # Volwassenen
        fig = px.scatter(df,
                         x = 'Adult_mortality',
                         y = 'Life_expectancy',
                         trendline = 'ols',
                         trendline_color_override = 'red',
                         color_discrete_sequence = ['#7FD7A4'],
                         width = 670)

        fig.update_layout(title = 'Relatie tussen het aantal sterfgevallen van volwassenen en de levensverwachting',
                          xaxis_title = 'Aantal sterfgevallen (per 1000 inwoners)',
                          yaxis_title = 'Levensverwachting in jaren')
        fig
        
    ###################################################################################################################
    # Variabele sterfgevallen
    
    st.markdown("""---""")
    st.markdown("""
    ## Lineair regressiemodel
    Met behulp van alle voorgaande informatie is een lineair regressiemodel gemaakt die met behulp van de volgende
    variabelen de gemiddelde levensverwachting in een land kan voorspellen:
    * Sterfgevallen van zuigelingen per 1000 inwoners
    * Sterfgevallen van volwassenen per 1000 inwoners
    * Percentage ingeënt tegen hepatitis B
    * Percentage ingeënt tegen de mazelen
    * Percentage ingeënt tegen polio
    * Percentage ingeënt tegen difterie""")
    
    col1, col2 = st.columns(2)

    with col1:
        df_lr = df.copy()
        df_lr.dropna(inplace = True)

        # Selecteer de onafhankelijke variabelen (features) en afhankelijke variabele (target)
        X = df_lr[['Infant_deaths', 'Adult_mortality', 'Hepatitis_B', 'Measles', 'Polio', 'Diphtheria']]
        y = df_lr[['Life_expectancy']]

        # Split de dataset in train en test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # Bouw het lineaire regressiemodel
        lr_model = LinearRegression()

        # Train het model op de train set
        lr_model.fit(X_train, y_train)

        # Voorspel de levensverwachting op de test set
        y_pred = lr_model.predict(X_test)
        y_true = y_test

        ### Omzetten naar een dataframe voor plot
        y_pred = pd.DataFrame(y_pred)
        y_pred.reset_index(inplace = True)

        y_true.reset_index(inplace = True)
        y_true.drop(['index'], axis = 1, inplace = True)

        df_pred = pd.merge(y_pred, y_true, left_index = True, right_index = True)
        df_pred.rename(columns = {0: 'Prediction'}, inplace = True)
        df_pred.head()

        ### Plot
        fig = px.scatter(df_pred,
                         x = 'Life_expectancy',
                         y = 'Prediction')

        fig.update_traces(marker = dict(color = '#7FD7A4'))

        # Een lijn toevoegen voor wanneer de voorspelling perfect zou zijn
        fig.add_trace(go.Scatter(x = [35, 85],
                                 y = [35, 85],
                                 mode = 'lines',
                                 line = dict(color='red')))

        fig.update_layout(title = 'Voorspelde waarden vs echte waarden van de levensverwachting',
                          xaxis_title = 'Voorspelde waarden',
                          yaxis_title = 'Echte waarden',
                          showlegend = False,
                          xaxis_range = [35, 85],
                          yaxis_range = [35, 85],
                          width = 650)

        fig
    
    with col2:
        residuals = y_test - y_pred
        res = px.histogram(residuals)

        res.update_traces(marker_color = '#7FD7A4')
        res.update_layout(title = "Verdeling van de residuen",
                          xaxis_title="Levensverwachting",
                          yaxis_title = "Aantal",
                          showlegend = False)
        res

    r2 = r2_score(df_pred['Life_expectancy'], df_pred['Prediction'])
        
    st.markdown("""
    Bij dit model is gecontroleerd of de residuen normaal verdeeld zijn m.b.v een normaliteitsdiagram de 
    bijbehorende p-waarde en een histogram om te controleren of een lineair regressiemodel wel een goed model was.
    Hieruit kan men concluderen dat de residuen normaal verdeeld zijn.""")
    st.write("Het model heeft een regressiescore van ", r2, ".")


# In[ ]:


page_names_to_funcs = {
    "Opdrachtomschrijving": intro,
    "Grafieken": grafieken,
}

demo_name = st.sidebar.selectbox("Kies een pagina", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

