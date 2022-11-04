# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from soccerplots.radar_chart import Radar
from mplsoccer import PyPizza, add_image, FontManager
from highlight_text import fig_text
import streamlit as st
import requests
import io
pd.set_option('display.colheader_justify', 'center')

# %%
# Download specifc fonts from github
font_normal = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/roboto/'
                          'Roboto%5Bwdth,wght%5D.ttf')
font_italic = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/roboto/'
                          'Roboto-Italic%5Bwdth,wght%5D.ttf')
font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                        'RobotoSlab%5Bwght%5D.ttf')

# Create streamlit titel and markdown
st.title('Player Similarity Model')
st.markdown('Data: Wyscout | Leagues: Top 5 European Mens Leagues | Seasons: 2021-2022')

# %%
# Read both excel files
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def getData():
    similarity_url = 'https://raw.githubusercontent.com/MMurph11/player_similarity/main/similarity_df.csv'
    metrics_url = 'https://raw.githubusercontent.com/MMurph11/player_similarity/main/metrics_df.csv'
    similarity_download = requests.get(similarity_url).content
    metrics_download = requests.get(metrics_url).content
    similarity_df = pd.read_csv(io.StringIO(similarity_download.decode('utf-8')))
    metrics_df = pd.read_csv(io.StringIO(metrics_download.decode('utf-8')))
    return similarity_df, metrics_df

similarity_df, metrics_df = getData()

# Create list of players
players = similarity_df.columns.unique()
players = players.sort_values(ascending=True)
players = players.tolist()
players = [x for x in players if str(x) != 'nan']

# Create dropdown that filters the data
player_select = st.sidebar.selectbox('Select Player', players)
player_df = similarity_df[["Player","Team within selected timeframe","Position1","Age",player_select]].sort_values(player_select, ascending=False)

# Create team filter
teams = player_df['Team within selected timeframe'].unique()
teams = np.sort(teams)
teams = np.insert(teams, 0, 'All')
team_dropdown = st.selectbox('Teams', teams)
if team_dropdown == 'All':
    player_df = player_df
else:
    player_df = player_df.loc[player_df['Team within selected timeframe']==team_dropdown]

# Create position filter
player_df = player_df.rename(columns={'Position1':'Main Position'})
positions = player_df['Main Position'].unique()
positions = np.sort(positions)
positions = np.insert(positions, 0, 'All')
position_dropdown = st.selectbox('Positions', positions)
if position_dropdown == ('All'):
    player_df = player_df
else:
    player_df = player_df.loc[player_df['Main Position']==position_dropdown]

# Create age slider
age = player_df['Age'].unique()
age = np.sort(age)
start_age, end_age = st.select_slider('Age Range',options=age,value=(min(age),max(age)))
player_df = player_df.loc[player_df['Age']>start_age].loc[player_df['Age']<end_age]

# %%
#######.iloc[1:,:] #######
if player_df.Player.iloc[0] == player_select:
    player_df = player_df.reset_index(drop=True).iloc[1:,:]
else:
    player_df = player_df.reset_index(drop=True)

st.dataframe(player_df.head(10).style \
     .background_gradient(cmap='Blues',subset=[player_select]).format({player_select: "{:.2f}"}))

# %%
# selecting only numerical metrics
metrics = metrics_df.drop(['Birth country','Passport country','Foot','Height','Weight','On loan'], axis=1)
metrics = metrics.iloc[:, 11:-1]

metric_cols = list(metrics.columns)
metric_cols = list(metrics.columns)

# Z-Scores
#for col in metric_cols:
    #metrics[col] = (metrics[col] - metrics[col].mean())/metrics[col].std(ddof=0)

# Normalising 0-100
for col in metric_cols:
    metrics[col] = (metrics[col] - min(metrics[col])) / (max(metrics[col]) - min(metrics[col])) * 100

# Merge player name and minutes onto their metrics
metrics = metrics.reset_index()
metrics = metrics.merge(metrics_df[['index','Player','Minutes played','Position1']], how='left', on='index')

# %%
# Select two players to compare
player1 = metrics.loc[metrics['Player']==player_select].reset_index()
player2 = metrics.loc[metrics['Player']==player_df.Player.iloc[0]].reset_index()

# parameter list
params = [
    "Suc def actions P90", "PAdj Interceptions", "Interceptions P90",
    "Prog. runs P90", "Accelerations P90", "Dribbles P90", "Suc dribbles %",
    "Accurate passes %", "Deep completions P90", "Prog. passes P90", "Through passes P90",
    "xA P90", "Assists P90", "Crosses P90", "Smart passes P90",
    "xG P90", "Goals P90", "Goal conversion, %"
]

# value list
values = [
    round(player1['Successful defensive actions per 90'].loc[0],1),round(player1['PAdj Interceptions'].loc[0],1),round(player1['Interceptions per 90'].loc[0],1),
    round(player1['Progressive runs per 90'].loc[0],1), round(player1['Accelerations per 90'].loc[0],1), round(player1['Dribbles per 90'].loc[0],1), round(player1['Successful dribbles, %'].loc[0],1),
    round(player1['Accurate passes, %'].loc[0],1), round(player1['Progressive passes per 90'].loc[0],1), round(player1['Deep completions per 90'].loc[0],1), round(player1['Through passes per 90'].loc[0],1),
    round(player1['xA per 90'].loc[0],1), round(player1['Assists per 90'].loc[0],1), round(player1['Crosses per 90'].loc[0],1), round(player1['Smart passes per 90'].loc[0],1),
    round(player1['xG per 90'].loc[0],1), round(player1['Goals per 90'].loc[0],1),round(player1['Goal conversion, %'].loc[0],1)
]
values_2 = [
    round(player2['Successful defensive actions per 90'].loc[0],1),round(player2['PAdj Interceptions'].loc[0],1),round(player2['Interceptions per 90'].loc[0],1),
    round(player2['Progressive runs per 90'].loc[0],1), round(player2['Accelerations per 90'].loc[0],1), round(player2['Dribbles per 90'].loc[0],1), round(player2['Successful dribbles, %'].loc[0],1),
    round(player2['Accurate passes, %'].loc[0],1), round(player2['Progressive passes per 90'].loc[0],1), round(player2['Deep completions per 90'].loc[0],1), round(player2['Through passes per 90'].loc[0],1),
    round(player2['xA per 90'].loc[0],1), round(player2['Assists per 90'].loc[0],1), round(player2['Crosses per 90'].loc[0],1), round(player2['Smart passes per 90'].loc[0],1),
    round(player2['xG per 90'].loc[0],1), round(player2['Goals per 90'].loc[0],1), round(player1['Goal conversion, %'].loc[0],1)
]

# instantiate PyPizza class
baker = PyPizza(
    params=params,                  # list of parameters
    background_color="#EBEBE9",     # background color
    straight_line_color="#222222",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    last_circle_color="#222222",    # color of last circle
    other_circle_ls="-.",           # linestyle for other circles
    other_circle_lw=1               # linewidth for other circles
)

# plot pizza
fig, ax = baker.make_pizza(
    values,                     # list of values
    compare_values=values_2,    # comparison values
    figsize=(8, 8),             # adjust figsize according to your need
    kwargs_slices=dict(
        facecolor="#1A78CF", edgecolor="#222222",
        zorder=2, linewidth=1
    ),                          # values to be used when plotting slices
    kwargs_compare=dict(
        facecolor="#FF9300", edgecolor="#222222",
        zorder=2, linewidth=1,
    ),
    kwargs_params=dict(
        color="#000000", fontsize=9,
        fontproperties=font_normal.prop, va="center"
    ),                          # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=9,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    ),                          # values to be used when adding parameter-values labels
    kwargs_compare_values=dict(
        color="#000000", fontsize=9, fontproperties=font_normal.prop, zorder=3,
        bbox=dict(edgecolor="#000000", facecolor="#FF9300", boxstyle="round,pad=0.2", lw=1)
    ),                          # values to be used when adding parameter-values labels
)

# add title
fig_text(
    0.515, 0.99, f"<{player1[0]}> vs <{player2[0]}>", size=17, fig=fig,
    highlight_textprops=[{"color": '#1A78CF'}, {"color": '#EE8900'}],
    ha="center", fontproperties=font_bold.prop, color="#000000"
)

# add subtitle
fig.text(
    0.515, 0.942,
    "Percentile Rank | Season 2021-22",
    size=15,
    ha="center", fontproperties=font_bold.prop, color="#000000"
)

# add credits
CREDIT_1 = "data: Wyscout"
CREDIT_2 = "inspired by: @Worville, @FootballSlices, @somazerofc & @Soumyaj15209314"

fig.text(
    0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
    fontproperties=font_italic.prop, color="#000000",
    ha="right"
)

st.pyplot(fig=fig)

# %%



