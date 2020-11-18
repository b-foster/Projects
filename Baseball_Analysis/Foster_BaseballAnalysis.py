# Final Project
#
# Baseball
#
# Name: Brett Foster
#

from openpyxl import load_workbook
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as st



f20, f19, f18, f17 = pd.read_excel("data/Fielding20.xlsx"), pd.read_excel("data/Fielding19.xlsx"), pd.read_excel("data/Fielding18.xlsx"), pd.read_excel("data/Fielding17.xlsx")

p20, p19, p18, p17 = pd.read_excel("data/Pitching20.xlsx"), pd.read_excel("data/Pitching19.xlsx"), pd.read_excel("data/Pitching19.xlsx"), pd.read_excel("data/Pitching17.xlsx")

b20, b19, b18, b17 = pd.read_excel("data/Batting20.xlsx"), pd.read_excel("data/Batting19.xlsx"), pd.read_excel("data/Batting18.xlsx"), pd.read_excel("data/Batting17.xlsx")

s20, s19, s18, s17 = pd.read_excel("data/Standings20.xlsx"), pd.read_excel("data/Standings19.xlsx"), pd.read_excel("data/Standings18.xlsx"), pd.read_excel("data/Standings17.xlsx")
    

year20 = s20.iloc[:,0:11].merge(
    b20, on = "Tm", how="left").merge(
        p20, on='Tm', how='left').merge(
            f20, on='Tm', how='left')
            
year19 = s19.iloc[:,0:11].merge(
    b19, on = "Tm", how="left").merge(
        p19, on='Tm', how='left').merge(
            f19, on='Tm', how='left')

year18 = s18.iloc[:,0:11].merge(
    b18, on = "Tm", how="left").merge(
        p18, on='Tm', how='left').merge(
            f18, on='Tm', how='left') 
            
year17 = s17.iloc[:,0:11].merge(
    b17, on = "Tm", how="left").merge(
        p17, on='Tm', how='left').merge(
            f17, on='Tm', how='left') 
            
# Function for Changing Column names in databases
def renaming(df):
    df.rename(columns={'Fld%': 'FldPercent',
                       'R_x': 'RunsPGame',
                       'R_y': 'TotalRuns',
                       'SO/W': 'SOperW',
                       'OPS+': 'OPSplus',
                       'R/G': 'RunsPerGame',
                       '2B': 'Double',
                       '3B': 'Triple',
                       'W-L%_x': 'W_L_percent'},
              inplace=True)    
    df.drop(df.index[30])
    
 
renaming(year20)
renaming(year19)
renaming(year18)
renaming(year17)

# Creating Variables for the AL and NL leagues for each year, 2017 - 2019
NL20, NL19, NL18, NL17 = pd.DataFrame(year20[year20['Lg'] == 'NL']), year19[year19['Lg'] == 'NL'], year18[year18['Lg'] == 'NL'], year17[year17['Lg'] == 'NL']

AL20, AL19, AL18, AL17 = pd.DataFrame(year20[year20['Lg'] == 'AL']), year19[year19['Lg'] == 'AL'], year18[year18['Lg'] == 'AL'], year17[year17['Lg'] == 'AL']


    
    
def HistoPlot(variable, bins, xlab, title, density, b, t):
    plt.hist(variable, bins=bins, label='data', 
             alpha=0.5, edgecolor='black', density=density)
    plt.axvline(x=np.mean(variable), color='red', 
                linewidth=2, label='League Average ({:.2f})'.format(np.mean(variable)))
    if np.mean(variable) < 1:
        mn, mx = min(variable) - 0.05*(np.mean(variable)), max(variable) + 0.05*(np.mean(variable))
    elif np.mean(variable >= 1):
        mn, mx = min(variable) - 0.1*(np.mean(variable)), max(variable) + 0.1*(np.mean(variable))
    plt.xlim(mn, mx)
    if density==True:
        kde_vs = np.linspace(mn, mx, 301)
        kde = st.gaussian_kde(variable.dropna())
        plt.plot(kde_vs, kde.pdf(kde_vs), label='PDF', color='midnightblue')

    plt.ylim(b, t)
    plt.ylabel('Quantity')
    plt.legend(loc="best", fontsize=8)
    plt.xlabel(xlab)
    plt.title(title)
    plt.show()
    
    plt.close()
    

def description(variable):
    print(variable.describe())
    
    print('Mode: ' + str(st.mode(variable)))
    print('Variance: ' + str(np.var(variable)))
    print('Standard Deviation: ' + str(np.std(variable)))
    print('\n')
    
    
def plt_regression(x, y, label_1, label_2, title):

    sns.regplot(x=x, y=y, fit_reg=True)
    plt.xlabel(label_1)
    plt.ylabel(label_2)
    plt.title(title)
    plt.show()
    print(st.pearsonr(x, y))
    

def ALvsNL(variable):
    
    AL = AL19[variable].append(
        AL18[variable]).append(AL17[variable])
    
    NL = NL19[variable].append(
        NL18[variable]).append(NL17[variable])
    
    t_test = st.ttest_ind(AL, NL, equal_var=False)
    
    print('Test for {}:'.format(variable))
    print(t_test)
    print('\n')
    print('AL: {}  vs  NL: {}'.format(np.mean(AL), np.mean(NL)))
    print(st.pearsonr(AL, NL))
    print('\n')
    
    
def ComparingPlot(variable, bins, title, labeling, xlabel, minPly, maxPly):
    plt.hist([year20[variable], year19[variable], year18[variable], year17[variable]], bins=bins, align='right', 
             linewidth=2, density=True, alpha=0.25, edgecolor='black', 
             label=['2020 {}'.format(labeling), '2019 {}'.format(labeling), '2018 {}'.format(labeling), '2017 {}'.format(labeling)])
    mn, mx = min(year19[variable]) - minPly*(np.mean(year19[variable])), max(year19[variable]) + maxPly*(np.mean(year19[variable]))
    plt.xlim(mn, mx)
    kde_vs = np.linspace(mn, mx, 301)
    
    kde20 = st.gaussian_kde(year20[variable].dropna())
    kde19 = st.gaussian_kde(year19[variable].dropna())
    kde18 = st.gaussian_kde(year18[variable].dropna())
    kde17 = st.gaussian_kde(year17[variable].dropna())
    
    plt.plot(kde_vs, kde20.pdf(kde_vs), 
             color='blue', linewidth=2, alpha=0.8)
    plt.plot(kde_vs, kde19.pdf(kde_vs), 
             color='orange', linewidth=2.5, alpha=0.95)
    plt.plot(kde_vs, kde18.pdf(kde_vs), 
             color='green', linewidth=2, alpha=0.8)
    plt.plot(kde_vs, kde17.pdf(kde_vs), 
             color='red', linewidth=2, alpha=0.8)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend(loc='best', fontsize=8)
    plt.show()
    
    plt.close()


def CorrelationMatrix(df, index):
    corr_df = df.iloc[:, index]
    corrMatrix = corr_df.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.title('Correlation Graph of Different Variables')
    plt.show()
    
    plt.close()
    

def ALvsNLcomp20():
    fig = plt.figure()
    ax1 = fig.add_axes([0, 0, 0.5, 0.9], 
                       xlim=(0.35, 0.51))
    ax2 = fig.add_axes([0.6, 0, 0.5, 0.9],
                       xlim=(3, 6))
    
    fig.suptitle('American vs National League Comparisons, 2020 Season')
    ax1.set_xlabel('AL vs NL Slugging Percentage (%)')
    ax2.set_xlabel('AL vs NL Earned Run Average')
    ax1.hist([AL20['SLG'], NL20['SLG']], bins=8, label=['AL', 'NL'], linewidth=1, density=True, alpha=0.4, edgecolor='black', align='right')
    ax2.hist([AL20['ERA'], NL20['ERA']], bins=8, label=['AL', 'NL'], linewidth=1, density=True, alpha=0.4, edgecolor='black', align='right')
    
    En, Ex = min(AL20['ERA']) - 0.1*(np.mean(AL20['ERA'])), max(AL20['ERA']) + 0.1*(np.mean(AL20['ERA']))
    Bn, Bx = min(AL20['SLG']) - 0.2*(np.mean(AL20['SLG'])), max(AL20['SLG']) + 0.2*(np.mean(AL20['SLG']))
    kde_BA = np.linspace(Bn, Bx, 301)
    kde_ERA = np.linspace(En, Ex, 301)
    
    AL_BA_kde20 = st.gaussian_kde(AL20['SLG'].dropna())
    NL_BA_kde20 = st.gaussian_kde(NL20['SLG'].dropna())
    AL_ERA_kde20 = st.gaussian_kde(AL20['ERA'].dropna())
    NL_ERA_kde20 = st.gaussian_kde(NL20['ERA'].dropna())
    
    ax1.plot(kde_BA, AL_BA_kde20.pdf(kde_BA), 
             color='blue', linewidth=2, alpha=0.8)
    ax1.plot(kde_BA, NL_BA_kde20.pdf(kde_BA), 
             color='orange', linewidth=2.5, alpha=0.95)
    ax2.plot(kde_ERA, AL_ERA_kde20.pdf(kde_ERA), 
             color='blue', linewidth=2, alpha=0.8)
    ax2.plot(kde_ERA, NL_ERA_kde20.pdf(kde_ERA), 
             color='orange', linewidth=2, alpha=0.8)
    ax1.xaxis.set_ticks(np.arange(0.36, 0.50, 0.02))
    ax2.xaxis.set_ticks(np.arange(3.0, 6.0, 0.5))
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.show()
    
    plt.close()
    
    
def ALvsNLcomp19():
    fig = plt.figure()
    ax1 = fig.add_axes([0, 0, 0.5, 0.9],
                       xlim=(0.35, 0.51))
    ax2 = fig.add_axes([0.6, 0, 0.5, 0.9],
                       xlim=(3, 6))
    
    fig.suptitle('American vs National League Comparisons, 2019 Season')
    ax1.set_xlabel('AL vs NL Slugging Percentage (%)')
    ax2.set_xlabel('AL vs NL Earned Run Average')
    ax1.hist([AL19['SLG'], NL19['SLG']], bins=8, label=['AL', 'NL'], linewidth=1, density=True, alpha=0.4, edgecolor='black', align='right')
    ax2.hist([AL19['ERA'], NL19['ERA']], bins=8, label=['AL', 'NL'], linewidth=1, density=True, alpha=0.4, edgecolor='black', align='right')
    
    En, Ex = min(AL19['ERA']) - 0.2*(np.mean(AL19['ERA'])), max(AL19['ERA']) + 0.1*(np.mean(AL19['ERA']))
    Bn, Bx = min(AL19['SLG']) - 0.2*(np.mean(AL19['SLG'])), max(AL19['SLG']) + 0.2*(np.mean(AL19['SLG']))
    kde_BA = np.linspace(Bn, Bx, 301)
    kde_ERA = np.linspace(En, Ex, 301)
    
    AL_BA_kde19 = st.gaussian_kde(AL19['SLG'].dropna())
    NL_BA_kde19 = st.gaussian_kde(NL19['SLG'].dropna())
    AL_ERA_kde19 = st.gaussian_kde(AL19['ERA'].dropna())
    NL_ERA_kde19 = st.gaussian_kde(NL19['ERA'].dropna())
    
    ax1.plot(kde_BA, AL_BA_kde19.pdf(kde_BA), 
             color='blue', linewidth=2, alpha=0.8)
    ax1.plot(kde_BA, NL_BA_kde19.pdf(kde_BA), 
             color='orange', linewidth=2.5, alpha=0.95)
    ax2.plot(kde_ERA, AL_ERA_kde19.pdf(kde_ERA), 
             color='blue', linewidth=2, alpha=0.8)
    ax2.plot(kde_ERA, NL_ERA_kde19.pdf(kde_ERA), 
             color='orange', linewidth=2, alpha=0.8)
    ax1.xaxis.set_ticks(np.arange(0.36, 0.50, 0.02))
    ax2.xaxis.set_ticks(np.arange(3.0, 6.0, 0.5))
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.show()
    
    plt.close()
    
    
def Comparing2020(variable):
    
    AL = AL19[variable].append(
        AL18[variable]).append(AL17[variable])
    
    NL = NL19[variable].append(
        NL18[variable]).append(NL17[variable])
    
    nl_ttest = st.ttest_ind(NL20[variable], NL)
    nl_ftest = st.f_oneway(NL20[variable], NL)

    al_ttest = st.ttest_ind(AL20[variable], AL)
    al_ftest = st.f_oneway(AL20[variable], AL)
    
    print('Test for {}:'.format(variable))
    print(nl_ttest)
    print(nl_ftest)
    print('2020: {} vs 2017-2019: {}'.format(np.mean([NL20[variable]]), np.mean(NL)))
    print('\n')
    print(al_ttest)
    print(al_ftest)
    print('2020: {} vs 2017-2019: {}'.format(np.mean(AL20[variable]), np.mean(AL)))
    print('\n') 
    
    print('AL: {}  vs  NL: {}'.format(np.mean(AL), np.mean(NL)))
    
    
def OPS_cdf(log=False):
    mean = year20['TotalRuns'].mean()
    std = year20['TotalRuns'].std()
    
    xs = np.linspace(-3, 3, 100)
    y = std * xs + mean
    
    n = len(year20['TotalRuns'])
    x2 = np.random.normal(0,1,n)
    x2.sort()
    
    y2 = np.array(year20['TotalRuns'])
    y2.sort()
    
    if log == False:
        plt.plot(xs, y, color='r', alpha=0.3, label='model')
        plt.plot(x2, y2, color='b', label='Total Runs')
        plt.ylabel('Total Runs Scored per Team')
        plt.title('Normal Probability Plot For Total Runs Scored, 2020 Season')
        
    else:
        plt.plot(xs, np.log(y), color='r', alpha=0.3, label='model')
        plt.plot(x2, np.log(y2), label='Total Runs', color='b')
        plt.ylabel('Total Runs Scored (log 10)')
        plt.title('Normal Probability Plot For Log Scale Total Runs, 2020 Season')

    plt.xlabel('z')
    plt.legend(loc='best')
    plt.show()


def ScatterPlot(var1, var2, title, xlab, ylab):
    one = year20[var1].append(
        year19[var1]).append(
            year18[var1]).append(
                year17[var1])
                
    two = year20[var2].append(
        year19[var2]).append(
            year18[var2]).append(
                year17[var2])
    
    sns.regplot(x=one, y=two, fit_reg=True)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    
    print('{} vs {}:'.format(var1, var2))
    print('Pearson Correlation: ' + str(st.pearsonr(one, two)))
    print('Spearman Correlation: ' + str(st.spearmanr(one,two)))
    print('Covariance Matrix: ' + '\n' +
          str(np.cov(one, two)))
    print('\n')
    
    plt.show()
    
    plt.close()
    
    
def TestCDF(variable, xlab, title):
    v_sorted = sorted(year20[variable])
    testing = pd.Series(year19[variable]).append(
        year18[variable]).append(year17[variable])
    
    n, nbins, npatches = plt.hist(v_sorted, 30, density=True, linewidth=3,
                                histtype='step', cumulative=True, 
                                alpha=0.1, color='darkslateblue')
    
    m, mbins, mpatches = plt.hist(testing, 30, density=True, linewidth=3,
                                histtype='step', cumulative=True, 
                                alpha=0.1, color='darkgreen')
    
    y = ((1 / (np.sqrt(2 * np.pi) * np.std(v_sorted))) *
          np.exp(-0.5 * (1 / np.std(v_sorted) * (nbins - np.mean(v_sorted))) ** 2)).cumsum()
    y /= y[-1]
    
    z = ((1 / (np.sqrt(2 * np.pi) * np.std(testing))) *
      np.exp(-0.5 * (1 / np.std(testing) * (nbins - np.mean(testing))) ** 2)).cumsum()
    z /= z[-1]
    
    plt.plot(nbins, y, linewidth=2, ls='solid', 
             color='darkslateblue', label='2020', alpha=0.7)
    plt.plot(mbins, z, linewidth=2, ls='solid',
             color='darkgreen', label='2017-2019', alpha=0.7)
    plt.axvline([year20[year20.Tm == 'LAD'][variable]], ls=':',
                color='blue', lw=2, label='2020 WS Winner')
    plt.axvline([year19[year19.Tm == 'WSN'][variable]], ls=':',
                color='teal', lw=2, label='2019 WS Winner')
    plt.axvline([year18[year18.Tm == 'BOS'][variable]], ls=':',
                color='red', lw=2, label='2018 WS Winner')
    plt.axvline([year17[year17.Tm == 'HOU'][variable]], ls=':',
                color='darkorange', lw=2, label='2017 WS Winner')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('CDF')
    plt.legend(loc='best', fontsize=8)
    plt.show()
    plt.close()
    
           
def main():

        
    HistoPlot(year20['W_L_percent'], 10, 'Winning Percentage by Team', 'Distribution of Team Winning Percentage', False,0, 7)

    HistoPlot(year20['ERA'], 10, 'Earned Run Average', 'Distribution of 2020 Team Earned Run Average', False, 0, 7)
    
    HistoPlot(year20['RBI'], 10, 'Runs Batted In', 'Distribution of 2020 Team RBIs', False, 0, 6)
    
    HistoPlot(year20['TB'], 10, 'Total Bases Achieved', 'Distribution of 2020 Total Bases Achieved by Team', False, 0, 7)
    
    HistoPlot(year20['TotalRuns'], 10, 'Total Runs Scored by a Team', 'Total Runs Scored per Team 2020', False, 0, 6)
    
    HistoPlot(year20['BA'], 10, 'Team Batting Average', 'Distribution of Team Batting Average for 2020', False, 0, 8)
    
    HistoPlot(year20['SLG'], 10, 'Total Slugging Percentage', 'Distribution of Team Slugging Percentage for 2020', False, 0, 6)
    
    HistoPlot(year20['WHIP'], 10, 'Walks & Hits Per Innining', 'Walks and Hits per Inning Pitched 2020 Distribution', False, 0, 7)
    
    HistoPlot(year20['OBP'], 10, 'On-Base Percentage', '2020 On-Base Percentage of Teams Distribution', False, 0, 8)


    # Representation of some Outliers in the Data
    print('2020 Team(s) with an ERA greater than 5.2:')
    print(year20[year20.ERA > 5.2][['Rk', 'Tm']])
    print('\n')
    
    print('2020 Team(s) with on On-Base Percentage less than .300:')
    print(year20[year20.OBP < .3][['Rk', 'Tm']])
    print('\n')
    
    print('2020 Team(s) with less than 225 RBIs:')
    print(year20[year20.RBI < 225][['Rk', 'Tm']])
    print('\n')
    
    print('2020 Team(s) with more than 325 RBIs:')
    print(year20[year20.RBI > 325][['Rk', 'Tm']])
    print('\n')
    
    print('2020 Team(s) with more than 330 Total Runs Scored:')
    print(year20[year20.TotalRuns > 330][['Rk', 'Tm']])
    print('\n')
    
    print('2020 Team(s) with less than 1.19 Walks and Hits per Inning Pitched:')
    print(year20[year20.WHIP < 1.19][['Rk', 'Tm']])
    print('\n')
    
    print('2020 Team(s) with more than 1.5 ')
    print(year20[year20.WHIP > 1.5][['Rk', 'Tm']])
    print('\n')
    
    print('2020 Team(s) with more than 950 Total Bases Achieved:')
    print(year20[year20.TB > 950][['Rk', 'Tm']])
    
    print('\n')
    
    print('Season - 2020:' + '\n')
    print('League rank for the outliers on the "Home Runs" histogram plot,' + '\n' + 
          'Home Runs by a Team greater than 100: ' + str(year20[year20.HR_x > 100]['Rk'].values) + '\n')
    
    print('League rank for the outliers on the "Runs Batted In" histogram plot' + '\n' + 
          'Total Team RBIs less than 225: ' + str(year20[year20.RBI < 225]['Rk'].values) + '\n')
    
    print('League rank for the outliers on the "Total Bases" histogram plot' + '\n' + 
          'Total Bases Completed by a Team greater than 950: ' + str(year20[year20.TB > 950]['Rk'].values) + '\n')
    
    print('League rank for the outliers on the "Earned Run Average" histogram plot' + '\n' + 
          'Team ERA Average greater than 5.2: ' + str(year20[year20.ERA > 5.2]['Rk'].values))

    print('\n')
    print('Season - 2019:' + '\n')
    print('League rank for the outliers on the "Home Runs" histogram plot,' + '\n' + 
          'Home Runs by a Team less than 100: ' + str(year19[year19.HR_x < 199]['Rk'].values) + '\n')
    
    print('League rank for the outliers on the "Runs Scored" histogram plot,' + '\n' + 
          'Runs Scored by a Team less than 650: ' + str(year19[year19.RunsPerGame  < 650]['Rk'].values) + '\n')

    
    year20['ERA2'] = year20.ERA**2
    year20['ERA3'] = year20.ERA**3
    
    year20_model = smf.ols('W_L_percent ~ TB + TotalRuns + ERA2 + ERA3', data=year20)
    
    year20_results = year20_model.fit()
    
    print(year20_results.summary())
    print('\n')
    
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('Wonders of Rank 14?:' + '\n' + 
          str(year19[year19.Rk == 14][['Tm', 'ERA', 'TB', 'WHIP', 'RBI', 'OBP']]))
    
    print('\n')
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(year19[year19.Tm == 'LgAvg'][['Rk', 'Tm', 'ERA', 'TB', 'WHIP', 'RBI', 'OBP']])
    
    
    print('\n')
    
    print('2020 Variables Averages between the American and National Leagues:')
    print(year20[['Lg', 'TotalRuns', 'BA', 'ERA', 'OBP']].groupby('Lg').mean())
    
    
    iloc_numbers1 = [6,8,9,20,21,24,26]

    iloc_numbers2 = [6,27,28,29,30,33,44]
    
    iloc_numbers3 = [6,52,67,71,72,76,84,86]
        
    
    ComparingPlot('W_L_percent', 10, 
                  'Distribution of Winning Percentage from the 2017-2020 Seasons', 
                 'Winning %', 'Winning Percentage (%)', 0.05, 0.2)
    
    ComparingPlot('ERA', 10, 
                  'Distribution of Earned Run Average, 2017-2020 Seasons', 
                 'ERA', 'Earned Run Average', 0.1, 0.1)
    
    ComparingPlot('BA', 10, 
                  'Comparing Team Batting Averages from the 2017-2020 Seasons', 
                 'Batting Average', 'Team Batting Average', 0.1, 0.05)
    
    CorrelationMatrix(year20, iloc_numbers1)
    print('\n')
    
    CorrelationMatrix(year20, iloc_numbers2)
    print('\n')
    
    CorrelationMatrix(year20, iloc_numbers3)
    print('\n')
    
    
    ALvsNLcomp20()
    print('\n')

    print('\n')
    print('2020 AL Slugging Percentage Average: ' + str(round(np.mean(AL20['SLG']), 3)))
    print('2020 NL Slugging Percentage Average: ' + str(round(np.mean(NL20['SLG']), 3)))
    print('\n')
    print('2020 AL Earned Run Average: ' + str(round(np.mean(AL20['ERA']), 3)))
    print('2020 NL Earned Run Average: ' + str(round(np.mean(NL20['ERA']), 3)))
    print('\n')
    
    ALvsNLcomp19()
    print('\n')
    
    print('2019 AL Slugging Percentage Average: ' + str(round(np.mean(AL19['SLG']), 3)))
    print('2019 NL Slugging Percentage Average: ' + str(round(np.mean(NL19['SLG']), 3)))
    print('\n')
    print('2019 AL Earned Run Average: ' + str(round(np.mean(AL19['ERA']), 3)))
    print('2019 NL Earned Run Average: ' + str(round(np.mean(NL19['ERA']), 3)))
    print('\n')
    
    TestCDF('WHIP', 'Walks & Hits (per Inning Pitched)', 'Cumulative Distribution Function of WHIP by Team')

    TestCDF('BA', 'Team Batting Average', 'Cumulative Distribution Function of Team Batting Average')
    
    TestCDF('SLG', 'Slugging Percentage (% per Team)', 'Cumulative Distribution Function of Team Slugging Percentage')
    
    TestCDF('ERA', 'Opponent Earned Runs Average (per 9 Innings)', 'Cumulative Distribution Function of Earned Run Average')
    
    TestCDF('OBP', 'On-Base Percentage (%)', 'Cumulative Distribution Function of On-Base Percentage by Team')
    
    TestCDF('RunsPGame', 'Runs Per Game', 'Cumulative Distribution Function of Runs Per Game')
    
    TestCDF('SO_x', 'Strikeouts', 'Cumulative Distribution Function of Runs Per Game')

    
    OPS_cdf(log=False)
    print('\n')
    OPS_cdf(log=True)
    print('\n')
    
    # Important Scatter Plots
       
    ScatterPlot('OBP', 'W_L_percent', 'Comparison Between On-Base % and Win %',
                'On-Base Percentage (%)', 'Winning Percentage (%)')
    
    ScatterPlot('RunsPGame', 'W_L_percent', 'Comparing Runs per Game and Winning %',
                'Runs Per Game', 'Winning Percentage (%)')
    
    ScatterPlot('BA', 'W_L_percent', 'Comparing Batting Average and Winning %',
                'Batting Average', 'Winning Percentage (%)')
    
    ScatterPlot('ERA', 'W_L_percent', 'Comparison of Earned Run Average and Winning %',
                'Earned Run Average', 'Winning Percentage (%)')
    
    ScatterPlot('WHIP', 'W_L_percent', 'WHIP vs Winning Percentage',
                'Walks & Hits per Inning Pitched', 'Winning Percentage (%)')
    
    ScatterPlot('SO9', 'W_L_percent', 'Importance of Strikeouts to Winning Percentage',
                'Strikeouts per 9 Innings', 'Winning Percentage (%)')
    
    ScatterPlot('RunsPGame', 'OBP', 'On-Base Percentage\'s Influence on Runs',
                'Runs Scored (per Game)', 'On-Base Percentage (%)')
    
    ScatterPlot('ERA', 'SO9', 'Runs Scored vs Batting Average',
                'Runs Scored (per Game)', 'Team Batting Average')
    
    ScatterPlot('WHIP', 'SO9', 'Runs Scored vs Batting Average',
                'Runs Scored (per Game)', 'Team Batting Average')
    
    
        
    ALvsNL('ERA')
    print('\n')
    
    ALvsNL('SLG')
    print('\n')
    
    ALvsNL('OBP')
    print('\n')
    
    ALvsNL('BA')
    print('\n')
    
    ALvsNL('WHIP')
    print('\n')
    
    ALvsNL('SO9')
    print('\n')
    
    ALvsNL('TB')
    print('\n')
    
    ALvsNL('SO_x')
    print('\n')
    
    era17_19 = year19['ERA'].append(year18['ERA']).append(year17['ERA'])
    
    year_era_test = st.ttest_ind(year20['ERA'], era17_19, equal_var=False)
    
    ba17_19 = year19['BA'].append(year18['BA']).append(year17['BA'])
    
    year_ba_test = st.ttest_ind(year20['BA'], ba17_19, equal_var=False)
    
    slg17_19 = year19['SLG'].append(year18['SLG']).append(year17['SLG'])
    
    year_slg_test = st.ttest_ind(year20['SLG'], slg17_19, equal_var=False)
    
    
    print('BA')
    print(year_ba_test)
    print('\n')
    print('SLG')
    print(year_slg_test)
    print('\n')
    print('ERA')
    print(year_era_test)
    print('\n')
    
      
        
        
    # All Important
    
    Comparing2020('BA')
    print('\n')
    
    Comparing2020('SLG')
    print('\n')
    
    Comparing2020('ERA')
    print('\n')
    
    Comparing2020('WHIP')
    print('\n')
    
    
    # Correlation Plot and Calculations between variables
    ScatterPlot('SO9', 'W_L_percent', 'Importance of Strikeouts and Winning Percentage', 'Strikeouts (per 9 Innings)', 'Winning Percentage (%)')
    print('\n')
    
    ScatterPlot('RunsPGame', 'W_L_percent', 'Score Runs and Winning Games', 'Runs Scored (per Game)', 'Winning Percentage (%)')
    print('\n')
    
    ScatterPlot('OBP', 'W_L_percent', 'Comparison between On-Base Percentage and Winning', 'On-Base Percentage (%)', 'Winning Percentage (%)')
    print('\n')
    
    ScatterPlot('ERA', 'W_L_percent', 'Pitchers Abilites to Win Games (ERA vs Winning)', 'Earned Run Average', 'Winning Percentage (%)')
    print('\n')
    
    ScatterPlot('ERA', 'SO9', 'Pitchers Abilites to Win Games (ERA vs Winning)', 'Earned Run Average', 'Winning Percentage (%)')
    print('\n')
    
    ScatterPlot('ERA', 'WHIP', 'Pitchers Abilites to Win Games (ERA vs Winning)', 'Earned Run Average', 'Winning Percentage (%)')
    print('\n')
    
    ScatterPlot('WHIP', 'SO9', 'Pitchers Abilites to Win Games (ERA vs Winning)', 'Earned Run Average', 'Winning Percentage (%)')
    print('\n')
    
    ScatterPlot('TB', 'W_L_percent', 'Pitchers Abilites to Win Games (ERA vs Winning)', 'Earned Run Average', 'Winning Percentage (%)')


if __name__ == '__main__':
    main()
    




