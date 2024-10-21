import pandas as pd
import numpy as np
from scipy.stats import norm


def get_over_under():
    data = pd.read_csv('cfb_hist.csv')
    data = data.sort_values(by=['season', 'week'], ascending=[True, True])
    data = data.reset_index(drop=True)
    data = data.drop_duplicates(subset='game_id')
    fbs_conferences = ["SEC", "Mountain West", "Mid-American", "Big Ten", 
                       "American Athletic", "Conference USA", "Pac-12", 
                       "FBS Independents", "Sun Belt", "ACC", "Big 12"]
    data = data[(data['home_conference'].isin(fbs_conferences)) |(data['away_conference'].isin(fbs_conferences))]
    data = data.drop(columns={'provider', 'formatted_spread', 'spread_open', 'over_under_open', 
                          'home_moneyline', 'away_moneyline', 'home_conference', 'away_conference'})
    data = data.rename(columns={'start_date':'gameday', 'spread':'spread_line', 'over_under':'total_line'})
    data['gameday'] = pd.to_datetime(data['gameday'])
    data['gameday'] = data['gameday'].dt.date
    data['team_favored'] = data.apply(lambda row: row['home_team'] if row['spread_line'] < 0 else row['away_team'], axis=1)
    data['spread_favorite'] = data.apply(lambda row: -row['spread_line'] if row['team_favored'] == row['away_team'] else row['spread_line'], axis=1)
    data['over_under_result'] = np.where((data['home_score'] + data['away_score']) > data['total_line'], 'over', 'under')
    data['home_favorite'] = np.where(data['spread_line'] < 0, 1, 0)

    def check_favorite_covered(row):
        if row['home_favorite'] == 1:
            return 1 if (row['home_score'] - row['away_score']) > abs(row['spread_favorite']) else 0
        else:
            return 1 if (row['away_score'] - row['home_score']) > abs(row['spread_favorite']) else 0
    data['favorite_covered'] = data.apply(check_favorite_covered, axis=1)
    data['winning_team'] = np.where(data['home_score'] > data['away_score'], data['home_team'],
                                    np.where(data['away_score'] > data['home_score'], data['away_team'], 'Tie'))
    data['losing_team'] = np.where(data['home_score'] < data['away_score'], data['home_team'],
                                   np.where(data['away_score'] < data['home_score'], data['away_team'], 'Tie'))
    team_wins = {}
    team_losses = {}
    team_points_for = {}
    team_points_against = {}
    data['home_wins'] = 0
    data['home_losses'] = 0
    data['away_wins'] = 0
    data['away_losses'] = 0
    data['home_points_for'] = 0
    data['home_points_against'] = 0
    data['away_points_for'] = 0
    data['away_points_against'] = 0
    data['home_win_pct_last_4'] = 0
    data['away_win_pct_last_4'] = 0
    
    for season, group in data.groupby('season'):
        team_wins = {}
        team_losses = {}
        team_points_for = {}
        team_points_against = {}
        team_wins_last_4 = {}
    
        for i, row in group.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            winning_team = row['winning_team']
            losing_team = row['losing_team']
            home_points = row['home_score']
            away_points = row['away_score']
    
            if home_team not in team_wins:
                team_wins[home_team] = 0
                team_losses[home_team] = 0
                team_points_for[home_team] = 0
                team_points_against[home_team] = 0
                team_wins_last_4[home_team] = []
            if away_team not in team_wins:
                team_wins[away_team] = 0
                team_losses[away_team] = 0
                team_points_for[away_team] = 0
                team_points_against[away_team] = 0
                team_wins_last_4[away_team] = []
    
            data.at[i, 'home_wins'] = team_wins[home_team]
            data.at[i, 'home_losses'] = team_losses[home_team]
            data.at[i, 'away_wins'] = team_wins[away_team]
            data.at[i, 'away_losses'] = team_losses[away_team]
            data.at[i, 'home_points_for'] = team_points_for[home_team]
            data.at[i, 'home_points_against'] = team_points_against[home_team]
            data.at[i, 'away_points_for'] = team_points_for[away_team]
            data.at[i, 'away_points_against'] = team_points_against[away_team]
    
            data.at[i, 'home_win_pct_last_4'] = (sum(team_wins_last_4[home_team]) / 4 if len(team_wins_last_4[home_team]) == 4 else sum(team_wins_last_4[home_team])
                                                 / len(team_wins_last_4[home_team]) if len(team_wins_last_4[home_team]) > 0 else 0)
            data.at[i, 'away_win_pct_last_4'] = (sum(team_wins_last_4[away_team]) / 4 if len(team_wins_last_4[away_team]) == 4 else sum(team_wins_last_4[away_team]) 
                                                 / len(team_wins_last_4[away_team]) if len(team_wins_last_4[away_team]) > 0 else 0)
    
            if winning_team != 'Tie':
                team_wins[winning_team] += 1
                team_losses[losing_team] += 1
    
                team_wins_last_4[winning_team].append(1)
                team_wins_last_4[losing_team].append(0)
                if len(team_wins_last_4[winning_team]) > 4:
                    team_wins_last_4[winning_team].pop(0)
                if len(team_wins_last_4[losing_team]) > 4:
                    team_wins_last_4[losing_team].pop(0)
    
            
            team_points_for[home_team] += home_points
            team_points_against[home_team] += away_points
            team_points_for[away_team] += away_points
            team_points_against[away_team] += home_points
    
    data['home_win_pct'] = data['home_wins'] / (data['home_wins'] + data['home_losses'])
    data['away_win_pct'] = data['away_wins'] / (data['away_wins'] + data['away_losses'])
    data['win_pct_diff'] = data['home_win_pct'] - data['away_win_pct']
    data['home_win_pct'] = data['home_win_pct'].fillna(0)
    data['away_win_pct'] = data['away_win_pct'].fillna(0)
    data['h_ppg'] = data['home_points_for'] / (data['home_wins'] + data['home_losses'])
    data['h_papg'] = data['home_points_against'] / (data['home_wins'] + data['home_losses'])
    data['a_ppg'] = data['away_points_for'] / (data['away_wins'] + data['away_losses'])
    data['a_papg'] = data['away_points_against'] / (data['away_wins'] + data['away_losses'])
    data['home_pt_diff_pg'] = (data['home_points_for'] - data['home_points_against']) / (data['home_wins'] + data['home_losses'])
    data['away_pt_diff_pg'] = (data['away_points_for'] - data['away_points_against']) / (data['away_wins'] + data['away_losses'])
    data['pt_diff_pg'] = data['home_pt_diff_pg'] + data['away_pt_diff_pg']
    data['total_score'] = data['home_score'] + data['away_score']
    data['result'] = data['home_score'] - data['away_score']
    data = data[~((data['week'] > 3) &
              ((data['home_wins'] + data['home_losses'] ==0) |
              (data['away_wins'] + data['away_losses'] ==0)
             ))]
    data = data.dropna(subset=['total_line'])
    def calculate_win_probability_from_spread(spread, std_dev=13.45):
        return norm.cdf(spread / std_dev)
    data['home_win_prob'] = calculate_win_probability_from_spread(data['spread_favorite'])
    data['away_win_prob'] = 1 - data['home_win_prob']
    raw = data[['gameday', 'season', 'week', 'home_team', 'away_team', 'team_favored', 
                'spread_favorite', 'home_score', 'away_score', 'total_score', 'total_line',
                'over_under_result', 'result', 'home_favorite', 'favorite_covered', 
                'winning_team', 'losing_team', 'home_wins', 'home_losses', 'away_wins',
                'away_losses', 'home_points_for', 'home_points_against', 'away_points_for',
                'away_points_against', 'home_win_pct', 'away_win_pct', 'win_pct_diff', 'h_ppg',
                'h_papg', 'a_ppg', 'a_papg', 'home_pt_diff_pg', 'pt_diff_pg', 'away_pt_diff_pg',
                'home_win_pct_last_4', 'away_win_pct_last_4', 'home_win_prob', 'away_win_prob']].copy()
    raw.fillna(0, inplace=True)


    return raw