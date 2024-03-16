import csv
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

FILENAME = os.path.join(os.path.dirname(__file__), 'significance_test_data.csv')

def display_comparision(comparision_header: str, rhyme_data: np.ndarray = None, meter_data: np.ndarray = None, year_data:np.ndarray =None, 
                        rhyme_base:float = None, rhyme_improved:float = None, meter_base:float = None, meter_improved:float = None, 
                        year_base:float = None, year_improved:float = None, p_value_checks:list = [1, 5, 25, 50, 75, 95, 99]):
    # CMAP
    _CMAP = plt.get_cmap('Dark2')
    SLICED_CMAP = _CMAP(np.linspace(0, 1, len(p_value_checks) + 1))

    # Plot Rhyme Data
    if not rhyme_data is  None:
        sns.histplot(data=pd.DataFrame({'rhyme_acc': rhyme_data}), x='rhyme_acc').set(title=f'RHYME {comparision_header}')
        rhyme_lows_pos = np.percentile(rhyme_data, p_value_checks)
        for i, (p_value, value) in enumerate(zip(p_value_checks, rhyme_lows_pos)):
            plt.axvline(x=value, color= SLICED_CMAP[i], ls='--', label= f'{p_value} %')
        if rhyme_base != None and rhyme_improved != None:
            percentiles = stats.percentileofscore(rhyme_data, [rhyme_base, rhyme_improved])
            plt.axvline(x=rhyme_base, color='r' , ls=':', label=f'BASE: {percentiles[0]} %', linewidth=5)
            plt.axvline(x=rhyme_improved, color='black' , ls=':', label=f'IMPROVED: {percentiles[1]} %', linewidth=5)
        plt.legend()
        plt.show()
        plt.figure()

    # Plot Meter Data
    if  not meter_data is None:
        sns.histplot(data=pd.DataFrame({'meter_acc': meter_data}), x='meter_acc').set(title=f'METER {comparision_header}')
        meter_lows_pos = np.percentile(meter_data, p_value_checks)
        for i, (p_value, value) in enumerate(zip(p_value_checks, meter_lows_pos)):
            plt.axvline(x=value, color= SLICED_CMAP[i], ls='--', label= f'{p_value} %')
        if meter_base != None and meter_improved != None:
            percentiles = stats.percentileofscore(meter_data, [meter_base, meter_improved])
            plt.axvline(x=meter_base, color='r' , ls=':', label=f'BASE: {percentiles[0]} %', linewidth=5)
            plt.axvline(x=meter_improved, color='black' , ls=':', label=f'IMPROVED: {percentiles[1]} %', linewidth=5)
        plt.legend()
        plt.show()
        plt.figure()

    # Plot Year Data
    if not year_data is None:
        sns.histplot(data=pd.DataFrame({'year_acc': year_data}), x='year_acc').set(title=f'YEAR {comparision_header}')
        year_lows_pos = np.percentile(year_data, p_value_checks)
        for i, (p_value, value) in enumerate(zip(p_value_checks, year_lows_pos)):
            plt.axvline(x=value, color= SLICED_CMAP[i], ls='--', label= f'{p_value} %')
        if year_base != None and year_improved != None:
            percentiles = stats.percentileofscore(year_data, [year_base, year_improved])
            plt.axvline(x=year_base, color='r' , ls=':', label=f'BASE: {percentiles[0]} %', linewidth=5)
            plt.axvline(x=year_improved, color='black' , ls=':', label=f'IMPROVED: {percentiles[1]} %', linewidth=5)
        plt.legend()
        plt.show()
        plt.figure()



with open(FILENAME, 'r') as file:
    rows = []
    spamreader = csv.reader(file, delimiter=',')
    for row in spamreader:
        rows.append(row)
    # Display RHYME Distil RAW to Disil Syllable 
    display_comparision(comparision_header= rows[0][1], 
                        rhyme_data=np.asarray( list(map(float,rows[0][2:])) ), rhyme_base=0.9682, rhyme_improved=0.9689)
    # Display RHYME RobeCzech RAW to RobeCzech Syllable 
    display_comparision(comparision_header= rows[1][1],
                        rhyme_data=np.asarray( list(map(float,rows[1][2:])) ), rhyme_base=0.4806, rhyme_improved=0.9468)
    # Display RHYME RobeCzech SYLLABLE to Disil Syllable 
    display_comparision(comparision_header= rows[2][1],
                        rhyme_data=np.asarray( list(map(float,rows[2][2:])) ), rhyme_base=0.9468, rhyme_improved=0.9689)
    # Display YEAR RobeCzech SYLLABLE to RobeCzech Raw 
    display_comparision(comparision_header= rows[3][1],
                        year_data=np.asarray( list(map(float,rows[3][2:])) ), year_base=0.4255, year_improved=0.4745)
    # Display YEAR Roberta Raw to RobeCzech Raw 
    display_comparision(comparision_header= rows[4][1],
                        year_data=np.asarray( list(map(float,rows[4][2:])) ), year_base=0.4315, year_improved=0.4745)
    # Display RHYME Roberta Raw Unweighted to Roberta Raw Weighted
    display_comparision(comparision_header= rows[5][1],
                        rhyme_data=np.asarray( list(map(float,rows[5][2:])) ), rhyme_base=0.9678, rhyme_improved=0.9693)
    # Display METER Distil RAW to Disil Syllable 
    display_comparision(comparision_header= rows[6][1],
                        meter_data=np.asarray( list(map(float,rows[6][2:])) ), meter_base=0.8987, meter_improved=0.8952)
    # Display METER Distil Syllable to Disil Syllable Context
    display_comparision(comparision_header= rows[7][1],
                        meter_data=np.asarray( list(map(float,rows[7][2:])) ), meter_base=0.8952, meter_improved=0.9494)
    # Display METER Roberta Syllable Context to Disil Syllable Context
    display_comparision(comparision_header= rows[8][1],
                        meter_data=np.asarray( list(map(float,rows[8][2:])) ), meter_base=0.9434, meter_improved=0.9494)
    # Display METER Roberta Raw Unweighted to Roberta Raw Weighted
    display_comparision(comparision_header= rows[9][1],
                        meter_data=np.asarray( list(map(float,rows[9][2:])) ), meter_base=0.8973, meter_improved=0.8928)
        
    # Display RHYME and METER Verse-Par input base and pretarined
    display_comparision(comparision_header= rows[10][1],
                        rhyme_data=np.asarray( list(map(float,rows[10][2:])) ), rhyme_base=0.898, rhyme_improved=0.883, 
                        meter_data=np.asarray( list(map(float,rows[11][2:])) ), meter_base=0.944, meter_improved=0.946)
    # Display RHYME and METER BASIC input base and pretarined
    display_comparision(comparision_header= rows[12][1],
                        rhyme_data=np.asarray( list(map(float,rows[12][2:])) ), rhyme_base=0.496, rhyme_improved=0.411, 
                        meter_data=np.asarray( list(map(float,rows[13][2:])) ), meter_base=0.944, meter_improved=0.922)
    # Display RHYME BASE tok Basic to Forced Generation
    display_comparision(comparision_header= rows[14][1],
                        rhyme_data=np.asarray( list(map(float,rows[14][2:])) ), rhyme_base=0.865, rhyme_improved=0.869,
                        meter_data=np.asarray( list(map(float,rows[15][2:])) ), meter_base=0.945, meter_improved=0.938)
    # Display RHYME OUR tok Basic to Forced Generation
    display_comparision(comparision_header= rows[16][1],
                        rhyme_data=np.asarray( list(map(float,rows[16][2:])) ), rhyme_base=0.806, rhyme_improved=0.806,
                        meter_data=np.asarray( list(map(float,rows[17][2:])) ), meter_base=0.946, meter_improved=0.948)
    # Display RHYME SYLLABLE tok Basic to Forced Generation
    display_comparision(comparision_header= rows[18][1],
                        rhyme_data=np.asarray( list(map(float,rows[18][2:])) ), rhyme_base=0.887, rhyme_improved=0.877)
    # Display RHYME BASIC Input to METER\_VERSE Input
    display_comparision(comparision_header= rows[19][1],
                        rhyme_data=np.asarray( list(map(float,rows[19][2:])) ), rhyme_base=0.496, rhyme_improved=0.868)
    # Display RHYME BASE Tok to OUR Tok
    display_comparision(comparision_header= rows[20][1],
                        rhyme_data=np.asarray( list(map(float,rows[20][2:])) ), rhyme_base=0.869, rhyme_improved=0.806)
    # Display RHYME UNICODE Tok Basic to Forced Gen 
    display_comparision(comparision_header= rows[21][1],
                       rhyme_data=np.asarray( list(map(float,rows[21][2:])) ), rhyme_base=0.738, rhyme_improved=0.940)
    # Display RHYME BASE Tok to SYLLABLE Tok
    display_comparision(comparision_header= rows[22][1],
                        rhyme_data=np.asarray( list(map(float,rows[22][2:])) ), rhyme_base=0.869, rhyme_improved=0.877)
        
    # Display RHYME, METER BASE Tok Base to BASE tok Secondary
    display_comparision(comparision_header= rows[23][1],
                        rhyme_data=np.asarray( list(map(float,rows[23][2:])) ), rhyme_base=0.868, rhyme_improved=0.865,
                        meter_data=np.asarray( list(map(float,rows[24][2:])) ), meter_base=0.946, meter_improved=0.945)
    # Display RHYME, METER OUR Tok Base to OUR tok Secondary
    display_comparision(comparision_header= rows[25][1],
                        rhyme_data=np.asarray( list(map(float,rows[25][2:])) ), rhyme_base=0.815, rhyme_improved=0.806,
                        meter_data=np.asarray( list(map(float,rows[26][2:])) ), meter_base=0.945, meter_improved=0.946)
    # Display RHYME, METER SYLLABLE Tok Base to SYLLABLE tok Secondary
    display_comparision(comparision_header= rows[27][1],
                        rhyme_data=np.asarray( list(map(float,rows[27][2:])) ), rhyme_base=0.902, rhyme_improved=0.887,
                        meter_data=np.asarray( list(map(float,rows[28][2:])) ), meter_base=0.939, meter_improved=0.946)
    # Display RHYME, METER SYLLABLE Tok to UNICODE Tok
    display_comparision(comparision_header= rows[29][1],
                        rhyme_data=np.asarray( list(map(float,rows[29][2:])) ), rhyme_base=0.877, rhyme_improved=0.940,
                        meter_data=np.asarray( list(map(float,rows[30][2:])) ), meter_base=0.942, meter_improved=0.940)
    