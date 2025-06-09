import pandas as pd
from matplotlib import pyplot as plt
import random
import numpy as np
import math
from scipy.stats import multivariate_normal, lognorm
import seaborn as sns
import re
from collections import defaultdict



# HELPER FUNCTIONS
def llk(rate, k):
    return - rate + k * np.log(rate) - np.log(math.factorial(k))

def sample_score(home_strength, away_strength, home_advantage, away_advantage):
    """ 
    sample score, calculate likelihood
    """
    exp_diff = np.exp(home_strength - away_strength)

    home_rate = home_advantage * exp_diff
    away_rate = away_advantage / exp_diff

    home_score = np.random.poisson(home_rate)
    away_score = np.random.poisson(away_rate)

    home_llk = llk(home_rate, home_score)
    away_llk = llk(away_rate, away_score)

    return [[home_score, away_score], home_llk + away_llk]

def score_llk(home_score, away_score, home_strength, away_strength, home_advantage, away_advantage, k = False):
    """ 
    get likelihood of score
    """
    exp_diff = np.exp(home_strength - away_strength)

    home_rate = home_advantage * exp_diff
    away_rate = away_advantage / exp_diff

    home_llk = llk(home_rate, home_score)
    away_llk = llk(away_rate, away_score)
    if k:
        return home_llk + away_llk + np.log(nu(home_score, away_score, k))
    else:
        return home_llk + away_llk

def nu(home_score, away_score, k):
    if home_score == 0:
        return 1.1 + k * away_score
    elif away_score == 0:
        return 1.1 + k * home_score
    else:
        return 1
    
    
def unnorm_joint_density(x, y, lam, mu, k):
    return nu(x, y, k) * (lam**x) * np.exp(-lam) / np.math.factorial(x) * \
                             (mu**y) * np.exp(-mu) / np.math.factorial(y)


def sample_score_rejection(home_strength, away_strength, home_advantage, away_advantage, k, max_trials=1000):
    """
    Sample score using rejection sampling from the enhanced joint Poisson model.
    Returns a tuple: ([home_goals, away_goals], joint_log_likelihood)
    """

    exp_diff = np.exp(home_strength - away_strength)
    home_rate = home_advantage * exp_diff
    away_rate = away_advantage / exp_diff

    # Proposal distribution is independent Poisson(lam) * Poisson(mu)
    for _ in range(max_trials):
        home_score = np.random.poisson(home_rate)
        away_score = np.random.poisson(away_rate)

        # Unnormalised target
        p_bar = score_llk(home_score, away_score, home_rate, away_rate, home_advantage, away_advantage, k)

        # Proposal density
        q = np.exp(score_llk(home_score, away_score, home_strength, away_strength, home_advantage, away_advantage))

        M = 1 + k * max(home_score, away_score) if (home_score == 0 or away_score == 0) else 1.0
        M *= 1.1  

        acceptance_prob = p_bar / (M * q) if q > 0 else 0
        if np.random.uniform(0, 1) < acceptance_prob:
            return [[home_score, away_score], p_bar]
        

def get_teams(season_df):
    """ 
    Get the list of teams from a season

    Inputs:
    season_df - dataframe containing season's results

    Outputs:
    teams - list of team names
    """
    teams = sorted(season_df['Home'][:100].unique())
    return teams

def initialise_strengths(teams, mu_p, sigma_p):
    """
    Initialise the strengths for a list of teams

    Inputs:
    teams - list of teams 
    
    Outputs:
    strengths_df - dataframe of teams and strengths
    """
    team_strengths = np.random.normal(loc=mu_p, scale=sigma_p, size=len(teams))

    strengths_df = pd.DataFrame({"Team Name": teams, "Strength": team_strengths})
    
    return strengths_df

def initialise_parameters(k = False):
    initial_dict =  {'Home Advantage': [np.random.uniform(1.4, 1.6)], #mean = shape * scale
            'Away Advantage': [np.random.uniform(1.1, 1.3)],
            'Eta': [np.random.uniform(0.75, 1.25)],
            'Sigma_S': [np.random.uniform(0.2, 0.4)],
            'Mu_P': [np.random.uniform(-0.25, 0)],
            'Sigma_P': [np.random.uniform(0.1, 0.4)]}
    
    if k:
        initial_dict['k'] = [np.random.uniform(-0.5, 0.5)]
    return initial_dict


def initial_prior_strengths(strengths_df, mu_p, sigma_p):
    """ 
    Prior log likelihood for initial strengths

    Inputs:
    strengths_df - dataframe of team strengths
    mu_p - promoted team mean
    sigma_p - promoted team standard deviation
    
    Outputs:
    llk - log likelihood for initial strengths
    """

    strengths = strengths_df['Strength']

    llk = multivariate_normal.logpdf(
        strengths,
        mean = np.full(len(strengths), mu_p),
        cov=np.eye(len(strengths)) * (sigma_p**2)
    )

    return llk


def strength_llk(strengths, scores, home_advantage, away_advantage, k = False):
    #To do: Add parameter sampling
    """ 
    Inputs:
    Strengths - Dataframe containing strengths for each team
    Scores - Dataframe containing scores for season 

    Returns:
    Log likelihood of scores given parameters
    """

    home_join = pd.merge(scores, strengths, left_on = "Home", right_on = "Team Name").rename(columns = {"Strength": "HomeStrength"})
    away_join = pd.merge(home_join, strengths, left_on = "Away", right_on = "Team Name").rename(columns = {"Strength": "AwayStrength"})

    away_join['Llk'] = away_join.apply(
    lambda row: pd.Series(score_llk(row["HomeGoals"], row["AwayGoals"], row["HomeStrength"], row["AwayStrength"],
                                    home_advantage, away_advantage, k)), axis = 1)

    return away_join["Llk"].sum()

def log_prior_strengths(current, previous, mu_p, sigma_p, eta, sigma_s):

    if previous is None:
        return initial_prior_strengths(current, mu_p, sigma_p)
    else:
        #prev_teams = list(previous.values())
        #curr_teams = list(current.values()) #lists of teams in previous and current season

        prev_teams = list(previous['Team Name'])
        curr_teams = list(current['Team Name']) #lists of teams in previous and current season

        promoted_teams = [team for team in curr_teams if team not in prev_teams] 
        remaining_teams = [team for team in curr_teams if team in prev_teams] #list of promoted and remaining teams
        prev_values = previous['Strength'].to_numpy()  # All previous strengths as numpy array
        curr_promoted_values = current[current['Team Name'].isin(promoted_teams)]['Strength'].to_numpy()
        curr_remaining_values = current[current['Team Name'].isin(remaining_teams)]['Strength'].to_numpy()
        prev_remaining_values = previous[previous['Team Name'].isin(remaining_teams)]['Strength'].to_numpy()

        avg_strength = np.mean(prev_values)

        promoted_log_likelihood = multivariate_normal.logpdf(
            curr_promoted_values,
            mean=np.full(len(promoted_teams), mu_p),  # vector of mu_p values
            cov=np.eye(len(promoted_teams)) * (sigma_p**2)  
        )

        mean_prior_remaining = eta * (prev_remaining_values - avg_strength)

        remaining_log_likelihood = multivariate_normal.logpdf(
            curr_remaining_values,
            mean=mean_prior_remaining,
            cov=np.eye(len(remaining_teams)) * (sigma_s**2)  # Diagonal covariance matrix
        )

        return promoted_log_likelihood + remaining_log_likelihood

def parameter_llk(season_dict, strength_dict, start_year, end_year, lambda_h, lambda_a, eta, sigma_s, mu_p, sigma_p, k = False):
    """ 
    Get log likelihood over whole season given parameters
    """
    season_df = season_dict.get(start_year)
    strength_df = strength_dict.get(start_year)[-1]
    
    proposed_prob = (
        initial_prior_strengths(strength_df, mu_p, sigma_p)
        + strength_llk(strength_df, season_df, lambda_h, lambda_a, k)
        )
    
    for year in range(start_year+1, end_year+1):
        season_df = season_dict.get(year)
        strength_df = strength_dict.get(year)[-1]
        prev_strength_df = strength_dict.get(year-1)[-1]

        proposed_prob += (
        log_prior_strengths(strength_df, prev_strength_df, mu_p, sigma_p, eta, sigma_s)
        + strength_llk(strength_df, season_df, lambda_h, lambda_a, k)
        )
    return proposed_prob


def create_season_dict(seasons_df, start_year, end_year, mu_p, sigma_p):
    """ 
    Returns dictionary, keys are years, values are season results
    and dictionary of strengths, all initialised
    """
    season_dict = {}
    strength_dict = {}
    
    for year in range(start_year, end_year + 1):
        season_data = seasons_df[seasons_df['Season_End_Year'] == year]
        season_dict[year] = season_data
        teams = get_teams(season_data)
        strength_dict[year] = [initialise_strengths(teams, mu_p, sigma_p)]
        
    return season_dict, strength_dict

def parameter_track(start_year, end_year, parameter_dict):
    tracking_dict = {}

    for year in range(start_year, end_year + 1):
        tracking_dict[year] = {}
        for parameter in parameter_dict.keys():
            tracking_dict[year][parameter] = []
        tracking_dict[year]['total'] = 0

    return tracking_dict


#MCMCM FUNCTION

def mcmc(seasons_df, start_year, end_year, burn_in = 1000, iterations = 5000, k = False):

    """ 
    Multi season MCMC

    start year and end year inclusive
    """
    
    #initialise parameters
    parameters = initialise_parameters(k)

    parameter_tracking = parameter_track(start_year, end_year, parameters)
    #parameter_names = ['Home Advantage', 'Away Advantage', 'Eta', 'Sigma_S', 'Mu_P', 'Sigma_P']
    lambda_h = parameters['Home Advantage'][0]
    lambda_a = parameters['Away Advantage'][0]
    eta = parameters['Eta'][0]
    sigma_s = parameters['Sigma_S'][0]
    mu_p = parameters['Mu_P'][0]
    sigma_p = parameters['Sigma_P'][0]
    if k:
        k = parameters['k'][0]

    season_dict, strength_dict = create_season_dict(seasons_df, start_year, end_year, mu_p, sigma_p)
    #store season results and initialise strengths in dictionaries
    log_prob_dict_strength = {}
    log_prob_dict_parameter = {}

    log_prob_dict_parameter['Home Advantage'] = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                           lambda_h, lambda_a, eta, sigma_s, mu_p, sigma_p, k)
    log_prob_dict_parameter['Away Advantage'] = log_prob_dict_parameter['Home Advantage']
    log_prob_dict_parameter['Remaining'] = log_prob_dict_parameter['Home Advantage']
    log_prob_dict_parameter['Promoted'] = log_prob_dict_parameter['Home Advantage']
    if k:
        log_prob_dict_parameter['k'] = log_prob_dict_parameter['Home Advantage']

    h_sig = 0.01
    a_sig = 0.01
    e_sig = 0.1
    sig_s_sig = np.sqrt(0.005)
    mu_p_sig = np.sqrt(0.0002)
    sig_p_sig = np.sqrt(0.002)
    k_sig = 0.1
    
    # Acceptance tracking
    acceptance_counters = {
        'Strength': {'accepted': 0, 'total': 0},
        'Home Advantage': {'accepted': 0, 'total': 0},
        'Away Advantage': {'accepted': 0, 'total': 0},
        'Mu_P': {'accepted': 0, 'total': 0},
        'Sigma_P': {'accepted': 0, 'total': 0},
        'Eta': {'accepted': 0, 'total': 0},
        'Sigma_S': {'accepted': 0, 'total': 0}
    }

    if k:
        acceptance_counters['k'] = {'accepted':0, 'total':0}
    
    num_teams = 20 #get total number of teams
    #initialise team strengths
    #parameters['Strength'] = [strength_df] #store initial strengths in parameter dataframe
    cov_matrix = 0.0002 * np.eye(num_teams) #can adjust
    num_indices = 5
    if k:
        num_indices += 1

    for i in range(burn_in + iterations):
        if np.random.uniform(0, 1) < 0.8: #strength proposal
            year = np.random.randint(start_year, end_year + 1) #select random season
            if i >= burn_in:
                acceptance_counters['Strength']['total'] += 1 #add to total iterations
                parameter_tracking[year]['total'] += 1
                for parameter in parameters.keys():
                    parameter_tracking[year][parameter].append(parameters[parameter][-1])
            season_df = season_dict.get(year) #get season results for that year
            strength_df = strength_dict.get(year)[-1]
            #get strengths for season
            proposed_strength = np.random.multivariate_normal(strength_df['Strength'], cov_matrix)
            proposed_strength_df = strength_df.copy()
            proposed_strength_df['Strength'] = proposed_strength  # Update strength
            if year == start_year:
                proposed_log_prob = (
                    initial_prior_strengths(proposed_strength_df, mu_p, sigma_p)
                    + strength_llk(proposed_strength_df, season_df, lambda_h, lambda_a)
                )
            else:
                prev_strength_df = strength_dict.get(year - 1, None)[-1]
                proposed_log_prob = (
                log_prior_strengths(proposed_strength_df, prev_strength_df, mu_p, sigma_p, eta, sigma_s)
                + strength_llk(proposed_strength_df, season_df, lambda_h, lambda_a)
                )

            prob_acceptance = np.exp(min(0, proposed_log_prob - log_prob_dict_strength.get(year, float('-inf'))))
            if np.random.uniform(0, 1) < prob_acceptance:
                strength_df = proposed_strength_df
                log_prob_dict_strength[year] = proposed_log_prob
                if i >= burn_in:
                    acceptance_counters['Strength']['accepted'] += 1
            strength_dict[year].append(strength_df)
            
        else:
            index = np.random.randint(1, num_indices)
            if index == 1: #update home advantage
                if i >= burn_in:
                    acceptance_counters['Home Advantage']['total'] += 1
                proposed_lambda_h = np.random.lognormal(np.log(lambda_h), h_sig) #maybe adjust sigma here
                
                proposed_lambda_h_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                       proposed_lambda_h, lambda_a, eta, sigma_s, mu_p, sigma_p)

                #prob_acceptance = np.exp(min(0, proposed_lambda_h_prob - log_prob_dict_parameter.get('Home Advantage', float('-inf'))))
                q_curr_given_prop = lognorm(s=h_sig, scale=lambda_h).logpdf(proposed_lambda_h)
                q_prop_given_curr = lognorm(s=h_sig, scale=proposed_lambda_h).logpdf(lambda_h)

                log_acceptance_ratio = (
                proposed_lambda_h_prob
                - log_prob_dict_parameter.get('Home Advantage', float('-inf'))
                + q_curr_given_prop
                - q_prop_given_curr
                )
                prob_acceptance = np.exp(min(0, log_acceptance_ratio))
                if np.random.uniform(0, 1) < prob_acceptance:
                    lambda_h = proposed_lambda_h
                    log_prob_dict_parameter['Home Advantage'] = proposed_lambda_h_prob
                    if i >= burn_in:
                        acceptance_counters['Home Advantage']['accepted'] += 1
                parameters['Home Advantage'].append(lambda_h)
            elif index == 2: #update away advantage
                if i >= burn_in:
                    acceptance_counters['Away Advantage']['total'] += 1
                proposed_lambda_a = np.random.lognormal(np.log(lambda_a), a_sig) #maybe adjust sigma here
                proposed_lambda_a_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                       lambda_h, proposed_lambda_a, eta, sigma_s, mu_p, sigma_p)
                #prob_acceptance = np.exp(min(0, proposed_lambda_a_prob - log_prob_dict_parameter.get('Away Advantage', float('-inf'))))
                q_curr_given_prop = lognorm(s=a_sig, scale=lambda_a).logpdf(proposed_lambda_a)
                q_prop_given_curr = lognorm(s=a_sig, scale=proposed_lambda_a).logpdf(lambda_a)

                log_acceptance_ratio = (
                proposed_lambda_a_prob
                - log_prob_dict_parameter.get('Away Advantage', float('-inf'))
                + q_curr_given_prop
                - q_prop_given_curr
                )
                prob_acceptance = np.exp(min(0, log_acceptance_ratio))
                if np.random.uniform(0, 1) < prob_acceptance:
                    lambda_a = proposed_lambda_a
                    log_prob_dict_parameter['Away Advantage'] = proposed_lambda_a_prob
                    if i >= burn_in:
                        acceptance_counters['Away Advantage']['accepted'] += 1
                parameters['Away Advantage'].append(lambda_a)

            elif index == 3: #update promoting parameters
                if i >= burn_in:
                    acceptance_counters['Eta']['total'] += 1
                    acceptance_counters['Sigma_S']['total'] += 1
                proposed_eta = np.random.normal(eta, e_sig)
                proposed_sigma_s = np.random.lognormal(np.log(sigma_s), sig_s_sig)
                proposed_s_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                lambda_h, lambda_a, proposed_eta, proposed_sigma_s, mu_p, sigma_p)
                #prob_acceptance = np.exp(min(0, proposed_s_prob - log_prob_dict_parameter.get('Remaining', float('-inf'))))
                q_curr_given_prop = lognorm(s=sig_s_sig, scale=proposed_sigma_s).logpdf(sigma_s)
                q_prop_given_curr = lognorm(s=sig_s_sig, scale=sigma_s).logpdf(proposed_sigma_s)

                # Log acceptance ratio
                log_acceptance_ratio = (
                    proposed_s_prob
                    - log_prob_dict_parameter.get('Remaining', float('-inf'))
                    + q_curr_given_prop
                    - q_prop_given_curr
                )
                prob_acceptance = np.exp(min(0, log_acceptance_ratio))
                if np.random.uniform(0, 1) < prob_acceptance:
                    eta = proposed_eta
                    sigma_s = proposed_sigma_s
                    log_prob_dict_parameter['Remaining'] = proposed_s_prob
                    if i >= burn_in:
                        acceptance_counters['Eta']['accepted'] += 1
                        acceptance_counters['Sigma_S']['accepted'] += 1
                parameters['Eta'].append(eta)
                parameters['Sigma_S'].append(sigma_s)

            elif index == 4: #update remaining team parameters
                if i >= burn_in:
                    acceptance_counters['Mu_P']['total'] += 1
                    acceptance_counters['Sigma_P']['total'] += 1
                proposed_mu_p = np.random.normal(mu_p, mu_p_sig)
                proposed_sigma_p = np.random.lognormal(np.log(sigma_p), sig_p_sig)
                proposed_p_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                lambda_h, lambda_a, eta, sigma_s, proposed_mu_p, proposed_sigma_p)
                #prob_acceptance = np.exp(min(0, proposed_p_prob - log_prob_dict_parameter.get('Promoted', float('-inf'))))
                q_curr_given_prop = lognorm(s=sig_p_sig, scale=proposed_sigma_p).logpdf(sigma_p)
                q_prop_given_curr = lognorm(s=sig_p_sig, scale=sigma_p).logpdf(proposed_sigma_p)                

                # Log acceptance ratio
                log_acceptance_ratio = (
                    proposed_p_prob
                    - log_prob_dict_parameter.get('Promoted', float('-inf'))
                    + q_curr_given_prop
                    - q_prop_given_curr
                )

                prob_acceptance = np.exp(min(0, log_acceptance_ratio))

                if np.random.uniform(0, 1) < prob_acceptance:
                    mu_p = proposed_mu_p
                    sigma_p = proposed_sigma_p
                    log_prob_dict_parameter['Promoted'] = proposed_p_prob
                    if i >= burn_in:
                        acceptance_counters['Mu_P']['accepted'] += 1
                        acceptance_counters['Sigma_P']['accepted'] += 1
                parameters['Mu_P'].append(mu_p)
                parameters['Sigma_P'].append(sigma_p)

            elif index == 5:
                if i >= burn_in:
                    acceptance_counters['k']['total'] += 1
                proposed_k = np.random.normal(k, k_sig) #maybe adjust sigma here
                
                proposed_k_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                       lambda_h, lambda_a, eta, sigma_s, mu_p, sigma_p, proposed_k)

                prob_acceptance = np.exp(min(0, proposed_k_prob - log_prob_dict_parameter.get('k', float('-inf'))))

                if np.random.uniform(0, 1) < prob_acceptance:
                    k = proposed_k
                    log_prob_dict_parameter['k'] = proposed_k_prob
                    if i >= burn_in:
                        acceptance_counters['k']['accepted'] += 1
                parameters['k'].append(k)
            
    
    strength_dict_original = strength_dict.copy()
    parameters_original = parameters.copy()
                    
    for parameter in acceptance_counters:
        if parameter == 'Strength':
            for year in strength_dict:
                year_total = parameter_tracking[year]['total']
                strength_dict[year] = strength_dict[year][-year_total:]
        else:
            accepted = acceptance_counters[parameter]['accepted']
            total = acceptance_counters[parameter]['total']
            acceptance_counters[parameter]['percentage'] = (accepted / total * 100) #calculate acceptance percentage
            parameters[parameter] = parameters[parameter][-total:] #take off burn in

    return [strength_dict, parameters, acceptance_counters, parameter_tracking, strength_dict_original, parameters_original]


def adaptive_mcmc(seasons_df, start_year, end_year, burn_in = 1000, iterations = 5000):

    """ 
    Multi season MCMC

    start year and end year inclusive
    """
    
    #initialise parameters
    parameters = initialise_parameters()

    parameter_tracking = parameter_track(start_year, end_year, parameters)
    #parameter_names = ['Home Advantage', 'Away Advantage', 'Eta', 'Sigma_S', 'Mu_P', 'Sigma_P']
    lambda_h = parameters['Home Advantage'][0]
    lambda_a = parameters['Away Advantage'][0]
    eta = parameters['Eta'][0]
    sigma_s = parameters['Sigma_S'][0]
    mu_p = parameters['Mu_P'][0]
    sigma_p = parameters['Sigma_P'][0]

    season_dict, strength_dict = create_season_dict(seasons_df, start_year, end_year, mu_p, sigma_p)
    #store season results and initialise strengths in dictionaries
    log_prob_dict_strength = {}
    log_prob_dict_parameter = {}

    log_prob_dict_parameter['Home Advantage'] = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                           lambda_h, lambda_a, eta, sigma_s, mu_p, sigma_p)
    log_prob_dict_parameter['Away Advantage'] = log_prob_dict_parameter['Home Advantage']
    log_prob_dict_parameter['Remaining'] = log_prob_dict_parameter['Home Advantage']
    log_prob_dict_parameter['Promoted'] = log_prob_dict_parameter['Home Advantage']

    h_sig = 0.01
    a_sig = 0.01
    e_sig = 0.1
    sig_s_sig = np.sqrt(0.005)
    mu_p_sig = np.sqrt(0.0002)
    sig_p_sig = np.sqrt(0.002)

    i_h = 0
    i_a = 0
    i_eta = 0
    i_sig_s = 0
    i_mu_p = 0
    i_sig_p = 0

    
    # Acceptance tracking
    acceptance_counters = {
        'Strength': {'accepted': 0, 'total': 0},
        'Home Advantage': {'accepted': 0, 'total': 0},
        'Away Advantage': {'accepted': 0, 'total': 0},
        'Mu_P': {'accepted': 0, 'total': 0},
        'Sigma_P': {'accepted': 0, 'total': 0},
        'Eta': {'accepted': 0, 'total': 0},
        'Sigma_S': {'accepted': 0, 'total': 0}
    }
    
    num_teams = 20 #get total number of teams
    #initialise team strengths
    #parameters['Strength'] = [strength_df] #store initial strengths in parameter dataframe
    cov_matrix = 0.0005 * np.eye(num_teams) #can adjust
    gamma = 0.75
    target_prob = 0.2

    for i in range(burn_in + iterations):
        if np.random.uniform(0, 1) < 0.8: #strength proposal
            year = np.random.randint(start_year, end_year + 1) #select random season
            if i >= burn_in:
                acceptance_counters['Strength']['total'] += 1 #add to total iterations
                parameter_tracking[year]['total'] += 1
                for parameter in parameters.keys():
                    parameter_tracking[year][parameter].append(parameters[parameter][-1])
            season_df = season_dict.get(year) #get season results for that year
            strength_df = strength_dict.get(year)[-1]
            #get strengths for season
            proposed_strength = np.random.multivariate_normal(strength_df['Strength'], cov_matrix)
            proposed_strength_df = strength_df.copy()
            proposed_strength_df['Strength'] = proposed_strength  # Update strength
            if year == start_year:
                proposed_log_prob = (
                    initial_prior_strengths(proposed_strength_df, mu_p, sigma_p)
                    + strength_llk(proposed_strength_df, season_df, lambda_h, lambda_a)
                )
            else:
                prev_strength_df = strength_dict.get(year - 1, None)[-1]
                proposed_log_prob = (
                log_prior_strengths(proposed_strength_df, prev_strength_df, mu_p, sigma_p, eta, sigma_s)
                + strength_llk(proposed_strength_df, season_df, lambda_h, lambda_a)
                )

            prob_acceptance = np.exp(min(0, proposed_log_prob - log_prob_dict_strength.get(year, float('-inf'))))
            if np.random.uniform(0, 1) < prob_acceptance:
                strength_df = proposed_strength_df
                log_prob_dict_strength[year] = proposed_log_prob
                if i >= burn_in:
                    acceptance_counters['Strength']['accepted'] += 1
            strength_dict[year].append(strength_df)
            
        else:
            index = np.random.randint(1, 5)
            if index == 1: #update home advantage
                if i >= burn_in:
                    acceptance_counters['Home Advantage']['total'] += 1
                proposed_lambda_h = np.random.normal(lambda_h, h_sig)
                
                proposed_lambda_h_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                       proposed_lambda_h, lambda_a, eta, sigma_s, mu_p, sigma_p)

                prob_acceptance = np.exp(min(0, proposed_lambda_h_prob - log_prob_dict_parameter.get('Home Advantage', float('-inf'))))
                if np.random.uniform(0, 1) < prob_acceptance:
                    lambda_h = proposed_lambda_h
                    log_prob_dict_parameter['Home Advantage'] = proposed_lambda_h_prob
                    if i >= burn_in:
                        acceptance_counters['Home Advantage']['accepted'] += 1
                parameters['Home Advantage'].append(lambda_h)
                i_h += 1
                h_sig = h_sig * np.sqrt(1 + (i_h**(-gamma)) * (prob_acceptance - target_prob))
            elif index == 2: #update away advantage
                if i >= burn_in:
                    acceptance_counters['Away Advantage']['total'] += 1
                proposed_lambda_a = np.random.normal(lambda_a, a_sig)
                proposed_lambda_a_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                       lambda_h, proposed_lambda_a, eta, sigma_s, mu_p, sigma_p)
                prob_acceptance = np.exp(min(0, proposed_lambda_a_prob - log_prob_dict_parameter.get('Away Advantage', float('-inf'))))

                if np.random.uniform(0, 1) < prob_acceptance:
                    lambda_a = proposed_lambda_a
                    log_prob_dict_parameter['Away Advantage'] = proposed_lambda_a_prob
                    if i >= burn_in:
                        acceptance_counters['Away Advantage']['accepted'] += 1
                parameters['Away Advantage'].append(lambda_a)
                i_a += 1
                a_sig = a_sig * np.sqrt(1 + (i_a**(-gamma)) * (prob_acceptance - target_prob))

            elif index == 3: #update promoting parameters
                if i >= burn_in:
                    acceptance_counters['Eta']['total'] += 1
                    acceptance_counters['Sigma_S']['total'] += 1
                proposed_eta = np.random.normal(eta, e_sig)
                proposed_sigma_s = np.random.normal(sigma_s, e_sig)
                proposed_s_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                lambda_h, lambda_a, proposed_eta, proposed_sigma_s, mu_p, sigma_p)
                prob_acceptance = np.exp(min(0, proposed_s_prob - log_prob_dict_parameter.get('Remaining', float('-inf'))))

                if np.random.uniform(0, 1) < prob_acceptance:
                    eta = proposed_eta
                    sigma_s = proposed_sigma_s
                    log_prob_dict_parameter['Remaining'] = proposed_s_prob
                    if i >= burn_in:
                        acceptance_counters['Eta']['accepted'] += 1
                        acceptance_counters['Sigma_S']['accepted'] += 1
                parameters['Eta'].append(eta)
                parameters['Sigma_S'].append(sigma_s)
                i_eta += 1
                e_sig = e_sig * np.sqrt(1 + (i_eta**(-gamma)) * (prob_acceptance - target_prob))

            elif index == 4: #update remaining team parameters
                if i >= burn_in:
                    acceptance_counters['Mu_P']['total'] += 1
                    acceptance_counters['Sigma_P']['total'] += 1
                proposed_mu_p = np.random.normal(mu_p, mu_p_sig)
                proposed_sigma_p = np.random.normal(sigma_p, mu_p_sig)
                proposed_p_prob = parameter_llk(season_dict, strength_dict, start_year, end_year,
                                                lambda_h, lambda_a, eta, sigma_s, proposed_mu_p, proposed_sigma_p)
                prob_acceptance = np.exp(min(0, proposed_p_prob - log_prob_dict_parameter.get('Promoted', float('-inf'))))
                if np.random.uniform(0, 1) < prob_acceptance:
                    mu_p = proposed_mu_p
                    sigma_p = proposed_sigma_p
                    log_prob_dict_parameter['Promoted'] = proposed_p_prob
                    if i >= burn_in:
                        acceptance_counters['Mu_P']['accepted'] += 1
                        acceptance_counters['Sigma_P']['accepted'] += 1
                parameters['Mu_P'].append(mu_p)
                parameters['Sigma_P'].append(sigma_p)
                i_mu_p += 1
                mu_p_sig = mu_p_sig * np.sqrt(1 + (i_mu_p**(-gamma)) * (prob_acceptance - target_prob))
                print(i_mu_p**(-gamma))
                print(mu_p_sig)
            
    
    strength_dict_original = strength_dict.copy()
    parameters_original = parameters.copy()
                    
    for parameter in acceptance_counters:
        if parameter == 'Strength':
            for year in strength_dict:
                year_total = parameter_tracking[year]['total']
                strength_dict[year] = strength_dict[year][-year_total:]
        else:
            accepted = acceptance_counters[parameter]['accepted']
            total = acceptance_counters[parameter]['total']
            acceptance_counters[parameter]['percentage'] = (accepted / total * 100) #calculate acceptance percentage
            parameters[parameter] = parameters[parameter][-total:] #take off burn in

    return [strength_dict, parameters, acceptance_counters, parameter_tracking, strength_dict_original, parameters_original]

#VISUALISATION FUNCTIONS

def strengths_by_team(mcmc_results):
    mcmc_results = [df.set_index('Team Name') for df in mcmc_results]
    joined_df = pd.concat(mcmc_results, axis = 1)
    joined_df['Strength History'] = joined_df.values.tolist()
    return joined_df[['Strength History']].reset_index()

def get_season_strengths(strength_dict, year):
    season = strength_dict[year]
    strength_histories = strengths_by_team(season)
    return strength_histories

def trace_plot_strength(strength_dict, teams, year, true_strengths=False):
    # Create a dictionary with team names as keys and strength history as values

    strength_histories = get_season_strengths(strength_dict, year)
    strength_dict = {
        team: strength_histories.loc[
            strength_histories['Team Name'] == team, 'Strength History'
        ].values[0]
        for team in teams
    }

    plt.figure(figsize=(10, 5))

    for team, strengths in strength_dict.items():
        plt.plot(strengths, label=team)

        if true_strengths is not False:
            plt.axhline(
                y=true_strengths.loc[strength_histories['Team Name'] == team, 'Strength'].values[0],
                linestyle='dotted',
                color=plt.gca().lines[-1].get_color()
            )

    plt.xlabel("Iteration")
    plt.ylabel("Strength")
    plt.title(f"Trace Plot of Team Strengths Over Iterations, Season {year-1}-{year}")
    plt.legend()
    plt.grid(True)
    plt.show()

def trace_plot_strength_with_burnin(strength_dict, strength_dict_original, teams, year, true_strengths=False):
    """
    Trace plot of strengths showing both burn-in and post-burn-in samples.
    
    Parameters:
    - strength_dict: strengths after burn-in
    - strength_dict_original: full MCMC trace including burn-in
    - teams: list of team names to plot
    - year: season to plot
    - burn_in: number of burn-in iterations
    - true_strengths: optional DataFrame of true strengths (with 'Team Name' and 'Strength')
    """
    burn_in = len(strength_dict_original[year]) - len(strength_dict[year])
    full_histories = get_season_strengths(strength_dict_original, year)
    post_burnin_histories = get_season_strengths(strength_dict, year)


    plt.figure(figsize=(12, 6))

    for team in teams:
        full_trace = full_histories.loc[full_histories['Team Name'] == team, 'Strength History'].values[0]
        plt.plot(full_trace, label=f"{team}")

        if true_strengths is not False:
            true_val = true_strengths.loc[true_strengths['Team Name'] == team, 'Strength'].values[0]
            plt.axhline(y=true_val, linestyle='dotted', color=plt.gca().lines[-1].get_color())

    # Add vertical line for burn-in boundary
    plt.axvline(x=burn_in, color='black', linestyle='dotted', label="End of Burn-in")

    plt.xlabel("Iteration")
    plt.ylabel("Strength")
    plt.title(f"Trace Plot of Team Strengths for {', '.join(teams)}, Season {year-1}-{year}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def trace_plot_parameter(parameter_dict):
    title_dict = {
        'Home Advantage': r'$\lambda_H$',
        'Away Advantage': r'$\lambda_A$',
        'Eta': r'$\eta$',
        'Sigma_S': r'$\sigma_s$',
        'Mu_P': r'$\mu_p$',
        'Sigma_P': r'$\sigma_p$'
    }

    num_params = len(parameter_dict)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns
    axes = axes.flatten()

    for ax, (parameter, values) in zip(axes, parameter_dict.items()):
        ax.plot(values)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(parameter)
        label = title_dict.get(parameter, parameter)
        ax.set_title(f"Trace Plot of {label}")
        ax.grid(True)

    for i in range(num_params, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def trace_plot_k(parameter_dict):
    values = parameter_dict['k']
    plt.plot(values)
    plt.xlabel("Iteration")
    plt.ylabel('k')
    plt.title("Trace Plot of k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def trace_plot_parameter_with_burnin(parameter_dict, parameter_dict_original):
    """
    Trace plots of scalar parameters showing both burn-in and post-burn-in samples.

    Parameters:
    - parameter_dict: dict of post-burn-in samples
    - parameter_dict_original: dict of full MCMC samples (including burn-in)
    """
    title_dict = {
        'Home Advantage': r'$\lambda_H$',
        'Away Advantage': r'$\lambda_A$',
        'Eta': r'$\eta$',
        'Sigma_S': r'$\sigma_s$',
        'Mu_P': r'$\mu_p$',
        'Sigma_P': r'$\sigma_p$'
    }

    num_params = len(parameter_dict)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (parameter, post_values) in zip(axes, parameter_dict.items()):
        full_values = parameter_dict_original[parameter]
        burn_in = len(full_values) - len(post_values)

        # Plot full trace
        ax.plot(full_values, color='blue',)

        # Highlight post-burn-in in a darker color
        ax.plot(range(burn_in, len(full_values)), post_values, color='blue')

        # Vertical burn-in divider
        ax.axvline(x=burn_in, color='black', linestyle='dotted', label='End of Burn-in')

        label = title_dict.get(parameter, parameter)
        ax.set_title(f"Trace Plot of {label}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    for i in range(num_params, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    #plt.legend()
    plt.show()

def comparison_plot(strength_dict, team1, team2, year):
    strength_histories = get_season_strengths(strength_dict, year)
    team1_strengths = strength_histories.loc[strength_histories['Team Name'] == team1, 'Strength History'].values[0]
    team2_strengths = strength_histories.loc[strength_histories['Team Name'] == team2, 'Strength History'].values[0]

    team1_wins = sum(t1 > t2 for t1, t2 in zip(team1_strengths, team2_strengths))
    total_iterations = len(team1_strengths)
    percentage_team1_stronger = (team1_wins / total_iterations) * 100

    print(f"{team1}'s strength was greater than {team2}'s strength in {percentage_team1_stronger}% of the iterations")

    colors = ['blue' if t1 > t2 else 'red' for t1, t2 in zip(team1_strengths, team2_strengths)]

    # Create scatter plot
    plt.figure(figsize=(6, 6))
    sns.kdeplot(x = team1_strengths, y = team2_strengths, fill=True, cmap="Blues", levels=20, alpha=1)

    # Plot the diagonal line (y = x) for reference
    min_val = min(min(team1_strengths), min(team2_strengths))
    max_val = max(max(team1_strengths), max(team2_strengths))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=1)

    # Labels and title
    plt.xlabel(f"{team1} Strength")
    plt.ylabel(f"{team2} Strength")
    plt.title(f"Strength Comparison: {team1} vs {team2}")

    # Legend
    plt.scatter([], [], color='blue', label=f"{team1} stronger")
    plt.scatter([], [], color='red', label=f"{team2} stronger")
    #plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()


#https://en.wikipedia.org/wiki/Credible_interval#/media/File:Highest_posterior_density_interval.svg

def strength_histogram(strength_dict, teams, year, colours, bins=80):
    strength_histories = get_season_strengths(strength_dict, year)
    all_strengths = []
    team_strength_dict = {}

    plt.figure(figsize=(8, 5))
    
    for team in teams:
        team_strengths = strength_histories.loc[
            strength_histories['Team Name'] == team, 'Strength History'
        ].values[0]
        team_strength_dict[team] = team_strengths
        all_strengths += team_strengths

    bin_edges = np.histogram_bin_edges(all_strengths, bins=bins)
    
    for i in range(len(teams)):
        plt.hist(team_strength_dict[teams[i]], bins=bin_edges, alpha = 0.5, color = colours[i], label=teams[i], edgecolor='black', density=True)

    plt.xlabel("Strength")
    plt.ylabel("Density")
    plt.title(f"Histogram of Strengths for {', '.join(teams)}: Season {year-1}-{year}")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def autocorrelation(samples, lag):
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples)
    cov = np.sum((samples[:n-lag] - mean) * (samples[lag:] - mean)) / n #covariance calculation
    return cov / var    #autocorrelation = covariance/variance

def find_neg_index(arr):
    for i, num in enumerate(arr):
        if num < 0:
            return i
    return -1  # Return last index if no negative

def autocorrelation_plot(strength_dict, team, year, max_lag=False):
    strength_histories = get_season_strengths(strength_dict, year)
    team_strengths = strength_histories.loc[strength_histories['Team Name'] == team, 'Strength History'].values[0]

    acf_values_all = []
    prev_acf = 1  # autocorrelation at lag 0 is always 1
    for lag in range(len(team_strengths)):
        acf = autocorrelation(team_strengths, lag)
        if lag > 0 and acf > prev_acf:
            break
        acf_values_all.append(acf)
        prev_acf = acf

    N = len(team_strengths)
    sum_acf = np.sum(acf_values_all)
    ess = N / (1 + 2 * sum_acf)
    if max_lag:
        acf_values_all = acf_values_all[:max_lag]
    
    print(f"Effective Sample Size for {team}: {ess:.2f} out of {N}")

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(acf_values_all)), acf_values_all, width=1.0, color='blue', alpha=0.6)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation Diagnostic Plot for {team}, Season {year-1}-{year}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return acf_values_all

def eff_sample_size_parameter(parameters, parameter, max_lag = False):
    values = parameters[parameter]
    acf_values_all = []
    prev_acf = 1  # autocorrelation at lag 0 is always 1
    for lag in range(len(values)):
        acf = autocorrelation(values, lag)
        if lag > 0 and acf > prev_acf and lag > max_lag:
            break
        acf_values_all.append(acf)
        prev_acf = acf

    N = len(values)
    sum_acf = np.sum(acf_values_all)
    ess = N / (1 + 2 * sum_acf)
    if max_lag:
        acf_values_all = acf_values_all[:max_lag]
    
    print(f"Effective Sample Size for {parameter}: {ess:.2f} out of {N}")

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(acf_values_all)), acf_values_all, width=1.0, color='blue', alpha=0.6)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation Diagnostic Plot for {parameter}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



# SYNTHETIC DATA FUNCTIONS

def sample_score(home_strength, away_strength, lambda_h, lambda_a):
    exp_diff = np.exp(home_strength - away_strength)
    home_rate = lambda_h * exp_diff
    away_rate = lambda_a / exp_diff
    home_score = np.random.poisson(home_rate)
    away_score = np.random.poisson(away_rate)
    home_llk = llk(home_rate, home_score)
    away_llk = llk(away_rate, away_score)
    return [[home_score, away_score], home_llk + away_llk]

def sample_initial_strengths(df, mu_p = 0, sigma_p = 0.1):
    # Define the number of teams
    # Create team names
    teams = sorted(df['Home'][:100].unique())

    # Generate Gaussian random variables for title strength
    # Mean = 0, Variance = 0.04 (std = sqrt(variance))
    team_strengths = np.random.normal(loc=mu_p, scale=sigma_p, size=len(teams))

    # Create the DataFrame
    sampled_df = pd.DataFrame({
        "Team Name": teams,
        "Strength": team_strengths
    })

    home_join = pd.merge(df, sampled_df, left_on = "Home", right_on = "Team Name").rename(columns = {"Strength": "HomeStrength"})
    away_join = pd.merge(home_join, sampled_df, left_on = "Away", right_on = "Team Name").rename(columns = {"Strength": "AwayStrength"})

    return away_join, sampled_df

def add_scores(games_df, lambda_h, lambda_a):
    # Apply function and expand results into two columns
    games_df[['Sampled Score', 'LogLikelihood']] = games_df.apply(
        lambda row: pd.Series(sample_score(row["HomeStrength"], row["AwayStrength"], lambda_h, lambda_a)), axis=1)

    return games_df

def sample_prior_strengths(df, previous, curr_team_names, mu_p = 0, sigma_p = 0.1, eta = 1, sigma_s = 0.1):
    # Separate promoted and remaining teams
    prev_team_names = list(previous['Team Name'])
    
    promoted_teams = [team for team in curr_team_names if team not in prev_team_names]
    remaining_teams = [team for team in curr_team_names if team in prev_team_names]
    
    # Get previous strengths
    avg_strength = np.mean(previous['Strength'].to_numpy())
    prev_remaining_strengths = previous[previous['Team Name'].isin(remaining_teams)]

    # Sample strengths
    # Promoted teams: N(mu_p, sigma_p^2)
    promoted_strengths = np.random.normal(loc=mu_p, scale=sigma_p, size=len(promoted_teams))

    # Remaining teams: N(mean_adjusted, sigma_s^2)
    mean_adjusted = avg_strength + eta * (prev_remaining_strengths['Strength'].to_numpy() - avg_strength)
    remaining_strengths = np.random.normal(loc=mean_adjusted, scale=sigma_s)

    # Build DataFrame
    all_teams = promoted_teams + remaining_teams
    all_strengths = np.concatenate([promoted_strengths, remaining_strengths])

    sampled_df = pd.DataFrame({
        'Team Name': all_teams,
        'Strength': all_strengths
    })

    home_join = pd.merge(df, sampled_df, left_on = "Home", right_on = "Team Name").rename(columns = {"Strength": "HomeStrength"})
    away_join = pd.merge(home_join, sampled_df, left_on = "Away", right_on = "Team Name").rename(columns = {"Strength": "AwayStrength"})

    return away_join, sampled_df

def compute_league_table(df):
    teams = pd.unique(df[['Home', 'Away']].values.ravel())

    table = pd.DataFrame(index=teams, columns=[
        'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'PTS'
    ]).fillna(0)

    for _, row in df.iterrows():
        home, away = row['Home'], row['Away']
        home_goals, away_goals = row['HomeGoals'], row['AwayGoals']

        # Matches played
        table.at[home, 'MP'] += 1
        table.at[away, 'MP'] += 1

        # Goals for and against
        table.at[home, 'GF'] += home_goals
        table.at[home, 'GA'] += away_goals
        table.at[away, 'GF'] += away_goals
        table.at[away, 'GA'] += home_goals

        # Match result
        if home_goals > away_goals:  # Home win
            table.at[home, 'W'] += 1
            table.at[away, 'L'] += 1
            table.at[home, 'PTS'] += 3
        elif home_goals < away_goals:  # Away win
            table.at[away, 'W'] += 1
            table.at[home, 'L'] += 1
            table.at[away, 'PTS'] += 3
        else:  # Draw
            table.at[home, 'D'] += 1
            table.at[away, 'D'] += 1
            table.at[home, 'PTS'] += 1
            table.at[away, 'PTS'] += 1

    # Goal difference
    table['GD'] = table['GF'] - table['GA']

    # Sort by points, then goal difference, then goals for
    table = table.sort_values(by=['PTS', 'GD', 'GF'], ascending=False)
    table = table.reset_index().rename(columns={'index': 'Team'})

    return table


def convert_match_data(df):
    df_converted = pd.DataFrame(index=df.index)

    df_converted['Season_End_Year'] = 2024
    df_converted['Wk'] = df['Round'].str.extract(r'(\d+)').astype(int)
    df_converted['Date'] = df['Date']
    df_converted['Home'] = df['Team']
    df_converted['HomeGoals'] = df['GF']
    df_converted['AwayGoals'] = df['GA']
    df_converted['Away'] = df['Opponent']

    def get_ftr(row):
        if row['HomeGoals'] > row['AwayGoals']:
            return 'H'
        elif row['HomeGoals'] < row['AwayGoals']:
            return 'A'
        else:
            return 'D'

    df_converted['FTR'] = df_converted.apply(get_ftr, axis=1)

    return df_converted

def sample_season(new_season, strength_dict, year, parameters):
    parameter_names = ['Home Advantage', 'Away Advantage', 'Eta', 'Sigma_S', 'Mu_P', 'Sigma_P']
    parameter_dict = {}
    strengths = random.choice(strength_dict[year])
    teams = sorted(new_season['Home'][:100].unique())

    for name in parameter_names:
        parameter_dict[name] = random.choice(parameters[name])

    strength_season, strength_df = sample_prior_strengths(
        new_season, strengths, teams, 
        mu_p=parameter_dict['Mu_P'], sigma_p=parameter_dict['Sigma_P'],
        eta=parameter_dict['Eta'], sigma_s=parameter_dict['Sigma_S']
    )

    sampled_score = add_scores(strength_season, parameter_dict['Home Advantage'], parameter_dict['Away Advantage'])

    sampled_score[['HomeGoals', 'AwayGoals']] = pd.DataFrame(
        sampled_score['Sampled Score'].tolist(),
        index=sampled_score.index
    )

    sampled_score = sampled_score[['Season_End_Year', 'Wk', 'Date', 'Home', 'HomeGoals', 'AwayGoals', 'Away', 'FTR']]
    sampled_score = sampled_score.reset_index(drop=True)
    sampled_score.index += 1

    league_table = compute_league_table(sampled_score)
    league_table = league_table.reset_index(drop=True)
    league_table.insert(0, 'Position', league_table.index + 1)

    return sampled_score, league_table, strength_df


def simulate_league_outcomes(new_season, strength_dict, year, parameters, num_simulations=1000, truncate=False):

    position_counts = defaultdict(lambda: defaultdict(int))

    for _ in range(num_simulations):
        _, league_table, _ = sample_season(new_season, strength_dict, year, parameters)

        for _, row in league_table.iterrows():
            team = row['Team']
            pos = row['Position']
            position_counts[team][pos] += 1

    # Build full probability table
    teams = sorted(position_counts.keys())
    positions = sorted({pos for counts in position_counts.values() for pos in counts})
    
    prob_df = pd.DataFrame(index=teams, columns=positions)

    for team in teams:
        total = sum(position_counts[team].values())
        for pos in positions:
            prob = position_counts[team].get(pos, 0) / total
            prob_df.loc[team, pos] = round(prob, 4)

    # Apply truncation if specified
    if truncate:
        top_probs = prob_df.loc[:, :truncate].sum(axis=1)
        top_teams = top_probs.sort_values(ascending=False).head(truncate).index
        prob_df = prob_df.loc[top_teams, :truncate]

    prob_df.columns.name = "Final Position"
    prob_df.index.name = "Team"

    return prob_df

def simulate_match_outcomes(home_team, away_team, strength_dict, parameters, year, num_simulations, max_goals = 4):

    strength_histories = get_season_strengths(strength_dict, year)

    home_strengths = strength_histories.loc[
        strength_histories['Team Name'] == home_team, 'Strength History'
    ].values[0]
    away_strengths = strength_histories.loc[
        strength_histories['Team Name'] == away_team, 'Strength History'
    ].values[0]

    score_counts = defaultdict(lambda: defaultdict(int))

    for _ in range(num_simulations):
        home_strength = np.random.choice(home_strengths)
        away_strength = np.random.choice(away_strengths)
        home_adv = np.random.choice(parameters['Home Advantage'])
        away_adv = np.random.choice(parameters['Away Advantage'])
        home_goals, away_goals = sample_score(home_strength, away_strength, home_adv, away_adv)[0]

        if home_goals <= max_goals and away_goals <= max_goals:
            score_counts[home_goals][away_goals] += 1

    # Create base index/columns
    index = list(range(max_goals + 1))
    data = []

    for h in index:
        row = []
        for a in index:
            prob = score_counts[h][a] / num_simulations
            row.append(round(prob, 2))
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, index=index, columns=index)

    # Add labels with MultiIndex
    df.columns = pd.MultiIndex.from_product([['Away Goals'], df.columns])
    df.index.name = 'Home Goals'

    return df

    


def insert_space(name):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name)


def goal_summary(df, start_year, end_year, max_goals=4):
    df_all = df[(df['Season_End_Year'] >= start_year) & (df['Season_End_Year'] <= end_year)]
    total_matches = len(df_all)

    # Print average goals
    avg_home_goals = df_all['HomeGoals'].mean()
    avg_away_goals = df_all['AwayGoals'].mean()
    print(f"Average Home Goals: {avg_home_goals:.2f}")
    print(f"Average Away Goals: {avg_away_goals:.2f}")

    # Compute marginals over full data
    away_dist = df_all['AwayGoals'].value_counts(normalize=True).sort_index()
    home_dist = df_all['HomeGoals'].value_counts(normalize=True).sort_index()

    # Build the display subset: only goals within range for table output
    table_data = []
    ratio_data = []

    for h in range(max_goals + 1):
        row = []
        ratio_row = []
        for a in range(max_goals + 1):
            p_joint = ((df_all['HomeGoals'] == h) & (df_all['AwayGoals'] == a)).sum() / total_matches
            se_joint = np.sqrt(p_joint * (1 - p_joint) / total_matches)
            row.append(f"{p_joint * 100:.1f} ({se_joint * 100:.2f})")

            p_expected = home_dist.get(h, 0) * away_dist.get(a, 0)
            ratio = p_joint / p_expected if p_expected > 0 else np.nan
            se_ratio = se_joint / p_expected if p_expected > 0 else np.nan

            ratio_row.append(f"{ratio:.2f} ({se_ratio:.2f})" if p_expected > 0 else "NA")
        table_data.append(row)
        ratio_data.append(ratio_row)

    # Marginals
    home_marginals = [
        f"{home_dist.get(h, 0) * 100:.1f} ({np.sqrt(home_dist.get(h, 0) * (1 - home_dist.get(h, 0)) / total_matches) * 100:.2f})"
        for h in range(max_goals + 1)
    ]
    away_marginals = [
        f"{away_dist.get(a, 0) * 100:.1f} ({np.sqrt(away_dist.get(a, 0) * (1 - away_dist.get(a, 0)) / total_matches) * 100:.2f})"
        for a in range(max_goals + 1)
    ]

    # Create main DataFrame
    prob_df = pd.DataFrame(table_data, index=range(max_goals + 1), columns=range(max_goals + 1))
    prob_df.insert(0, "", home_marginals)
    prob_df.columns = pd.MultiIndex.from_arrays([
        ["", "Away Goals"] + [""] * max_goals,
        ["Home Goals"] + list(range(max_goals + 1))
    ])
    prob_df = pd.concat([pd.DataFrame([[""] + away_marginals], columns=prob_df.columns), prob_df], ignore_index=True)
    prob_df.index = [""] + list(range(max_goals + 1))

    # Ratio DataFrame
    ratio_df = pd.DataFrame(ratio_data, index=range(max_goals + 1), columns=range(max_goals + 1))
    ratio_df.insert(0, "", home_marginals)
    ratio_df.columns = pd.MultiIndex.from_arrays([
        ["", "Away Goals"] + [""] * max_goals,
        ["Home Goals"] + list(range(max_goals + 1))
    ])
    ratio_df = pd.concat([pd.DataFrame([[""] + away_marginals], columns=ratio_df.columns), ratio_df], ignore_index=True)
    ratio_df.index = [""] + list(range(max_goals + 1))

    return prob_df, ratio_df


def subtract_prob_dfs(prob_df1, prob_df2):
    # Extract the numeric body only (exclude marginal rows and columns)
    df1_body = prob_df1.iloc[1:, 1:].copy()
    df2_body = prob_df2.iloc[1:, 1:].copy()

    # Extract only the numeric part before the space (discard SE)
    df1_numeric = df1_body.applymap(lambda s: float(s.split()[0]) if isinstance(s, str) else np.nan)
    df2_numeric = df2_body.applymap(lambda s: float(s.split()[0]) if isinstance(s, str) else np.nan)

    # Subtract and return a DataFrame
    diff = df1_numeric - df2_numeric
    diff.index = df1_body.index
    diff.columns = df1_body.columns
    return diff

def summarise_parameters(full_parameters):

    summary = []

    for param, samples in full_parameters.items():
        samples = np.array(samples)
        mean = np.mean(samples)
        std_err = np.std(samples, ddof=1) / np.sqrt(len(samples))  # Standard error
        summary.append((param, f"{mean:.4f} ({std_err:.4f})"))

    df = pd.DataFrame(summary, columns=["Parameter", "Mean (Std. Error)"])
    return df
