from pylab import mpl, plt
plt.style.use('seaborn-v0_8')
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_squared_error, mean_squared_log_error
from sklearn.cluster import KMeans
import math
from scipy import stats
from scipy.stats import boxcox
from scipy.special import boxcox1p
import warnings
warnings.simplefilter('ignore')

#Model is sklearn's `lasso`, it finds weighing factors for the features & applies them to the current week

def get_testing():
    data = pd.read_csv('ou_full.csv', index_col=0)
    data['exp_win_difference'] = data.home_win_prob - data.away_win_prob
    data['gameday'] = data['gameday'].apply(pd.to_datetime)
    data['total_ppg'] = (data.h_ppg + data.a_ppg)
    data['total_points_against'] = (data.h_papg + data.a_papg)
    data['last_four_difference'] = data.home_win_pct_last_4 - data.away_win_pct_last_4
    data['over_under_result'] = np.where(data['total_score'] == data['total_line'], 2, data['over_under_result'])
    data['over_under_result'] = np.where(data['total_score'] > data['total_line'], 1, data['over_under_result'])
    data['over_under_result'] = np.where(data['total_score'] < data['total_line'], 0, data['over_under_result'])
    #data['spread_favorite_shifted'] = data['spread_favorite'] + abs(data['spread_favorite'].min()) + 1
    #data['exp_win_difference_shifted'] = data['exp_win_difference'] + abs(data['exp_win_difference'].min()) + 1
    #data['win_pct_shifted'] = data['win_pct_diff'] + abs(data['win_pct_diff'].min()) + 1
    #data['last_four_shifted'] = data['last_four_difference'] + abs(data['last_four_difference'].min()) + 1

    week5_df =  data[(data.week > 4) & (data.week < 17) & (data.season>1998) & (data.season<2024)]

    return week5_df


def untransform(arr, lambda_):
    result = np.exp(np.log(lambda_ * arr + 1) / lambda_)
    return result

def past_predictions(style=None):
    df = get_testing()
    df['total_line_transformed'], lambda_total_line = boxcox(df['total_line'] + 1)
    X = df[['total_points_against', 'total_ppg', 'temp', 'wind', 'season']]
    X_transformed = np.column_stack([
        boxcox(X[col] + 1)[0] for col in X.columns
    ])
    y = df['total_line_transformed'].values.flatten()
    model = Lasso(alpha=0.1)
    model.fit(X_transformed, y)
    coefficients = model.coef_
    intercept = model.intercept_
    combined = np.concatenate(([intercept], coefficients))
    if style is not None:
        return combined
    else:
        y_pred = (
            intercept +
            coefficients[0] * boxcox(df['total_points_against'] + 1)[0] +
            coefficients[1] * boxcox(df['total_ppg'] + 1)[0] +
            coefficients[2] * boxcox(df['temp'] + 1)[0] +
            coefficients[3] * boxcox(df['wind'] + 1)[0] +
            coefficients[4] * boxcox(df['season'] + 1)[0]
        )
        y_preds = untransform(y_pred, lambda_total_line)
        y_preds = np.round(y_preds, 1)
        
        return y_preds
def get_current_games(week=None, year=None):
    if year is not None:
        year = year
    else:
        year = 2024
    data = pd.read_csv('ou_full.csv', index_col=0)
    coef = past_predictions(style='')
    week5_df = get_testing
    data = data[data['season'].isin([year])]
    data = data[data['week'].isin([week])]
    data['TPAPG'] = data['h_papg'] + data['a_papg']
    data['TPPG'] = data['h_ppg'] + data['a_ppg']
    raw = data[['home_team', 'away_team', 'total_line', 'TPAPG', 'TPPG', 'temp', 'wind', 'gameday' ]]
    raw['good_o_good_d']= np.where((raw.TPPG>=48) & (raw.TPAPG<=48),1,0)
    raw['good_o_bad_d']= np.where((raw.TPPG>=48) & (raw.TPAPG>=48),1,0)
    raw['bad_o_good_d']= np.where((raw.TPPG<=48) & (raw.TPAPG<=48),1,0)
    raw['bad_o_bad_d']= np.where((raw.TPPG<=48) & (raw.TPAPG>=48),1,0)

    return raw
    
def week_predictions(df, coef):
    week5_df = get_testing()
    y_preds = []
    for index, row in df.iterrows():

        yt,max_lambda =boxcox(week5_df.total_line)
        xt1,x1_lam=boxcox(week5_df.total_points_against)
        xt2,x2_lam=boxcox(week5_df.total_ppg)
        xt3,x3_lam=boxcox(week5_df.temp+1)
        xt4,x4_lam=boxcox(week5_df.wind+1)
        xt7,xlam7=boxcox(week5_df.season)

        papg = boxcox1p(row.TPAPG,x1_lam)
        ppg = boxcox1p(row.TPPG,x2_lam)
        temp = boxcox1p(row.temp,x3_lam)
        wind = boxcox1p(row.wind,x4_lam)
        season = boxcox1p(2018,xlam7)
        
        y = coef[0] + coef[1]*papg +coef[2]*ppg + coef[3]*temp + coef[4]*wind +coef[5]*season 
        y_pred = np.round(untransform(y,max_lambda)*2)/2
        y_preds.append(y_pred)
            
    return y_preds

def week_lines(week, insert=None):
    if insert == None:
        data = get_current_games(week)
    else:
        data = pd.read_csv('updated_weather.csv', index_col=0)
    coef = past_predictions(style='')
    week_preds = week_predictions(data, coef)
    week = data
    week['prediction'] = week_preds
    week['pred-actual'] = week.prediction - week.total_line

    return week

def predictions_df():
    week5_df = get_testing()
    past_preds = past_predictions()
    preds_df = week5_df.drop(columns=['temp', 'wind', 'result', 'home_team', 'away_team',
       'home_favorite', 'favorite_covered', 'winning_team', 'losing_team',
       'home_wins', 'home_losses', 'away_wins', 'away_losses',
       'home_points_for', 'home_points_against',
       'away_points_for', 'away_points_against',
       'home_win_pct', 'away_win_pct', 'win_pct_diff', 'h_ppg', 'h_papg',
       'a_ppg', 'a_papg', 'home_pt_diff_pg', 'away_pt_diff_pg', 'pt_diff_pg',
       'home_win_prob', 'away_win_prob', 'home_win_pct_last_4',
       'away_win_pct_last_4', 'exp_win_difference', 'total_ppg',
       'total_points_against', 'last_four_difference', 'team_favored','spread_favorite'])
    preds_df['point_total'] = preds_df['home_score'] + preds_df['away_score']
    preds_df['over_under_pred'] = past_preds
    preds_df['pred-actual'] = preds_df.over_under_pred- preds_df.total_line
    
    preds_df['good_pred'] = np.where((preds_df.over_under_pred > preds_df.total_line) & 
                                     (preds_df.point_total > preds_df.total_line), 1, 0)
    preds_df['good_pred'] = np.where((preds_df.over_under_pred < preds_df.total_line) & 
                                     (preds_df.point_total < preds_df.total_line), 1, preds_df.good_pred)
    preds_df['good_pred'] = np.where((preds_df.over_under_pred == preds_df.total_line), 1, preds_df.good_pred)
    preds_df['good_pred'] = np.where((preds_df.point_total == preds_df.total_line), 1, preds_df.good_pred)
    
    preds_df['new_ou_result'] = np.where(preds_df.over_under_pred < preds_df.point_total, 1, 0)
    preds_df['new_ou_result'] = np.where(preds_df.over_under_pred == preds_df.point_total, 2, preds_df.new_ou_result)

    return preds_df


def probs(prob):
    week5_df = get_testing()
    o_u_pivot = week5_df.pivot_table(index='total_line', columns='over_under_result',
                    aggfunc={'over_under_result':len}, fill_value = 0)
    o_u_pivot['row_total'] = o_u_pivot.over_under_result[1] + o_u_pivot.over_under_result[0] + o_u_pivot.over_under_result[2]
    ou_covered = o_u_pivot.over_under_result[1]
    ou_no_cover = o_u_pivot.over_under_result[0]
    ou_push = o_u_pivot.over_under_result[2]
    lines = sorted(set(week5_df.total_line))
    epsilon = 1e-6
    x_lines = np.array([spread for spread in lines])
    y_over = [(ou_covered[value] + 1) / (o_u_pivot['row_total'][value] + 3) for value in x_lines]
    y_under = [(ou_no_cover[value] + 1) / (o_u_pivot['row_total'][value] + 3) for value in x_lines]
    y_neither = [(ou_push[value] + 1) / (o_u_pivot['row_total'][value] + 3) for value in x_lines]
    prob_over = list(zip(x_lines, y_over))
    prob_under = list(zip(x_lines, y_under))
    prob_push = list(zip(x_lines, y_neither))
    if prob == 'over':
        return prob_over
    if prob == 'under':
        return prob_under
    if prob == 'push':
        return prob_push

def bayes(line, our_prediction):
    prob_over = probs('over')
    prob_under = probs('under')
    prob_push = probs('push')
    preds_df = predictions_df()
    over = [prob[1] for prob in prob_over if prob[0] == line][0]
    under = [prob[1] for prob in prob_under if prob[0] == line][0]
    push = [prob[1] for prob in prob_push if prob[0] == line][0]
    pred_o_given_o = len(preds_df[(preds_df.over_under_result==1) & (preds_df.total_line==line) & (preds_df.over_under_pred>preds_df.total_line)])/len(preds_df[(preds_df.over_under_result==1) & (preds_df.total_line==line)])
    pred_p_given_o = len(preds_df[(preds_df.over_under_result==1) & (preds_df.total_line==line) & (preds_df.over_under_pred==preds_df.total_line)])/len(preds_df[(preds_df.over_under_result==1) & (preds_df.total_line==line)])
    pred_u_given_o = len(preds_df[(preds_df.over_under_result==1) & (preds_df.total_line==line) & (preds_df.over_under_pred<preds_df.total_line)])/len(preds_df[(preds_df.over_under_result==1) & (preds_df.total_line==line)])

    pred_o_given_p = len(preds_df[(preds_df.over_under_result==2) & (preds_df.total_line==line) & (preds_df.over_under_pred>preds_df.total_line)])/(len(preds_df[(preds_df.over_under_result==2) & (preds_df.total_line==line)])+1)
    pred_p_given_p = len(preds_df[(preds_df.over_under_result==2) & (preds_df.total_line==line) & (preds_df.over_under_pred==preds_df.total_line)])/(len(preds_df[(preds_df.over_under_result==2) & (preds_df.total_line==line)])+1)
    pred_u_given_p = len(preds_df[(preds_df.over_under_result==2) & (preds_df.total_line==line) & (preds_df.over_under_pred<preds_df.total_line)])/(len(preds_df[(preds_df.over_under_result==2) & (preds_df.total_line==line)])+1)

    pred_o_given_u = len(preds_df[(preds_df.over_under_result==0) & (preds_df.total_line==line) & (preds_df.over_under_pred>preds_df.total_line)])/len(preds_df[(preds_df.over_under_result==0) & (preds_df.total_line==line)])
    pred_p_given_u = len(preds_df[(preds_df.over_under_result==0) & (preds_df.total_line==line) & (preds_df.over_under_pred==preds_df.total_line)])/len(preds_df[(preds_df.over_under_result==0) & (preds_df.total_line==line)])
    pred_u_given_u = len(preds_df[(preds_df.over_under_result==0) & (preds_df.total_line==line) & (preds_df.over_under_pred<preds_df.total_line)])/len(preds_df[(preds_df.over_under_result==0) & (preds_df.total_line==line)])

    if our_prediction < line:
        joint1 = pred_o_given_u * under
        joint2 = pred_p_given_u * under
        joint3 = pred_u_given_u * under
        normalizer = joint1 + joint2 + joint3
        return (joint3/normalizer)
    elif our_prediction > line:
        joint1 = pred_o_given_o * over
        joint2 = pred_p_given_o * over
        joint3 = pred_u_given_o * over
        normalizer = joint1 + joint2 + joint3
        return (joint1/normalizer)
    elif our_prediction == line:
        joint1 = pred_o_given_p * push
        joint2 = pred_p_given_p * push
        joint3 = pred_u_given_p * push
        normalizer = joint1 + joint2 + joint3
        return (joint2/normalizer)


def model_output(week, insert=None):
    if insert==None:
        reg_w1 = week_lines(week)
    else:
        reg_w1 = week_lines(week, insert='')
    bayes_probs = []
    for index, row in reg_w1.iterrows():
        bayes_probs.append(np.round(bayes(row.total_line, row.prediction), 3))
    
    reg_w1['prob_correct_pred'] = bayes_probs
    return reg_w1



def analyze_predictions():
    data = get_testing()
    preds = predictions_df()
    push = len(data[data['over_under_result'] == 2])
    len_data = len(data) - push
    overs = data['over_under_result'].sum()
    unders = (len(data) - data['over_under_result'].sum()) 
    real_over = overs / len_data
    real_under = unders / len_data
    under_model = len(preds[preds['pred-actual'] < 0])
    over_model = len(preds[preds['pred-actual'] > 0])
    model_under = under_model / len_data
    model_over = over_model / len_data

    print(f'Real data over % is {real_over: .2%}')
    print(f'Real data under % is {real_under: .2%}')
    print(f'Our model over % is {model_over: .2%}')
    print(f'Our model under % is {model_under: .2%}')
    

