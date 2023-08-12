# Basic
import pandas as pd
import numpy as np

# Loading, converting, saving libraries
from aws_to_df import AwsToDf
from newtools import PandasDoggo
from aws_to_df import AwsToDf

# Libraries for Plots/Data Visualisations
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

class b1:
    def eda(self):

        atd = AwsToDf()
        new_df = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'ML_df3.csv', 'csv', has_header=True)
        segment_df = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'segment.csv', 'csv', has_header=True)
        mrkt_hh = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'market_hh.csv', 'csv', has_header=True)
        genre_df = atd.files_to_df('prod_mb/data_source/machine_learning_data', 'genre_df.csv', 'csv', has_header=True)
        pd.set_option('display.max_columns', None)
        print('check')
        # Genre df
        """
        genre_df = df[['genre_id','genre_name']]
        genre_df = genre_df.groupby('genre_id')['genre_name'].first().reset_index()
    
        doggo = PandasDoggo()
        path = "s3://csmediabrain-mediabrain/prod_mb/data_source/machine_learning_data/genre_df.csv"
        doggo.save(genre_df, path)
        """
        print('check 1')

        # Creating watched for EDA
        genre_df = genre_df.drop(['Unnamed: 0'], axis=1)
        watch = new_df[new_df['watched'] == 1]
        g_watch = watch[['hh_id', 'genre_id', 'network_id']]
        g_watch = g_watch.merge(genre_df, on='genre_id', how='left')

        watch['time'] = pd.to_datetime(watch['time'])
        watch['hour_of_day'] = watch['time'].dt.hour
        watch = watch.drop(['time'], axis=1)

        print("DFs created")
        # Network

        network_distribution = watch['network_id'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(network_distribution, labels=network_distribution.index, autopct='%1.1f%%', startangle=140)
        plt.title('Network Distribution')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
        print('Network')
        # Genre
        genre_distribution = g_watch['genre_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(genre_distribution.index, genre_distribution.values)
        plt.title('Genre Distribution')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()
        print('Genre')
        # State
        """
        data = {
            'State Name': ['Montana', 'Michigan', 'Nebraska', 'Texas', 'Alaska'],
            'Latitude': [46.8797, 44.3148, 41.4925, 31.9686, 64.2008],
            'Longitude': [-110.3626, -85.6024, -99.9018, -99.9018, -149.4937]
        }
    
        states_df = pd.DataFrame(data)
    
        print(states_df)
        """
        # Hour of Day

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=watch['hour_of_day'].value_counts().index, y=watch['hour_of_day'].value_counts().values, marker="o")
        plt.title('TV Programme Viewership By Hour ')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count')
        plt.xticks(range(1, 25))
        plt.grid(True)
        plt.show()
        print('hour of day plot')
        # Day

        day_dist = watch['day'].value_counts()
        plt.figure(figsize=(8, 8))

        # Creating an outer circle representing the donut shape
        outer_circle = plt.Circle((0, 0), 0.7, color='white')
        plt.gca().add_artist(outer_circle)
        # Create the donut chart
        plt.pie(day_dist, labels=day_dist.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85,
                wedgeprops={'width': 0.4})
        plt.title('TV Viewership By Day ')
        plt.axis('equal')
        plt.show()
        print("Day")
        # Genre per network

        # 33
        n_33 = g_watch[g_watch['network_id'] == 33]

        g_33 = n_33['genre_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(g_33.index, g_33.values)
        plt.title('Genre Distribution For Network 33')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Genre: Network 33")
        # 4
        n_4 = g_watch[g_watch['network_id'] == 4]

        g_4 = n_4['genre_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(g_4.index, g_4.values)
        plt.title('Genre Distribution For Network 4')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        print("Genre: Network 4")
        # 5
        n_5 = g_watch[g_watch['network_id'] == 5]

        g_5 = n_5['genre_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(g_5.index, g_5.values)
        plt.title('Genre Distribution For Network 5')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Genre: Network 5")
        # Network Comb

        n_33 = g_watch[g_watch['network_id'] == 33]
        n_4 = g_watch[g_watch['network_id'] == 4]
        n_5 = g_watch[g_watch['network_id'] == 5]

        g_33 = n_33['genre_name'].value_counts()
        g_4 = n_4['genre_name'].value_counts()
        g_5 = n_5['genre_name'].value_counts()

        # Create an array of genre names
        genres = np.unique(np.concatenate((g_33.index, g_4.index, g_5.index)))

        # Create arrays for genre counts for each network, filling with 0 for missing genres
        g33_counts = np.array([g_33.get(genre, 0) for genre in genres])
        g4_counts = np.array([g_4.get(genre, 0) for genre in genres])
        g5_counts = np.array([g_5.get(genre, 0) for genre in genres])

        # Set up the positions for the bars
        bar_width = 0.25
        r1 = np.arange(len(genres))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        plt.figure(figsize=(12, 6))
        plt.bar(r1, g33_counts, color='dodgerblue', width=bar_width, edgecolor='grey', label='Network 33')
        plt.bar(r2, g4_counts, color='green', width=bar_width, edgecolor='grey', label='Network 4')
        plt.bar(r3, g5_counts, color='orange', width=bar_width, edgecolor='grey', label='Network 5')

        plt.title('Genre Distribution by Network')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks([r + bar_width for r in range(len(genres))], genres, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Network Comb")

        def cat(ser, g_watch):
            ser = ser['hh_id']
            ser = ser.to_frame()
            c = ser.merge(g_watch, on='hh_id', how='left')
            return c

        # 1 Male only (585)
        male_df = segment_df[segment_df['585'] == 1]
        male_df = cat(male_df, g_watch)

        m_g = male_df['genre_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(m_g.index, m_g.values)
        plt.title('TV Viewership by Genre in Single Male Households')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Male:Genre Distribution")

        # 1 Female only (586)
        female_df = segment_df[segment_df['586'] == 1]
        female_df = cat(female_df, g_watch)

        f_g = female_df['genre_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(f_g.index, f_g.values, color='green')
        plt.title('TV Viewership by Genre in Single Female Households')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Female Plot")
        # Comb

        m_g = male_df['genre_name'].value_counts()
        f_g = female_df['genre_name'].value_counts()

        # Create an array of genre names
        genres = np.unique(np.concatenate((m_g.index, f_g.index)))

        # Create arrays for male and female genre counts, filling with 0 for missing genres
        m_counts = np.array([m_g.get(genre, 0) for genre in genres])
        f_counts = np.array([f_g.get(genre, 0) for genre in genres])

        # plot
        bar_width = 0.4
        r1 = np.arange(len(genres))
        r2 = [x + bar_width for x in r1]

        plt.figure(figsize=(12, 6))
        plt.bar(r1, m_counts, color='blue', width=bar_width, edgecolor='grey', label='Male')
        plt.bar(r2, f_counts, color='green', width=bar_width, edgecolor='grey', label='Female')

        plt.title('Genre Distribution by Sex for Single Occupant Households')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks([r + bar_width / 2 for r in range(len(genres))], genres, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Male-female comb")
        # Child in HH
        child_df = segment_df[segment_df['677'] == 1]
        child_df = cat(child_df, g_watch)

        c_g = child_df['genre_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(c_g.index, c_g.values, color='orange')
        plt.title('TV Viewership by Genre for Households with Children')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Genre: Child")
        # Age group 1
        age_1 = segment_df[segment_df['644'] == 1]
        age_1 = cat(age_1, g_watch)

        age_g1 = age_1['genre_name'].value_counts()

        total_viewership = age_g1.sum()
        genre_proportions = age_g1 / total_viewership

        plt.figure(figsize=(10, 6))
        plt.bar(genre_proportions.index, genre_proportions.values, color='purple')
        plt.title('Genre Distribution Proportion for Households having Age Group 18-24')
        plt.xlabel('Genre')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Age: 18-24")
        # Age group 5 (proportion of genre viwership)
        age_5 = segment_df[segment_df['648'] == 1]
        age_5 = cat(age_5, g_watch)

        age_g5 = age_5['genre_name'].value_counts()

        total_viewership = age_g5.sum()
        genre_proportions = age_g5 / total_viewership

        plt.figure(figsize=(10, 6))
        plt.bar(genre_proportions.index, genre_proportions.values, color='violet')
        plt.title('Genre Distribution Proportion for Age Group 65+')
        plt.xlabel('Genre')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Age: 65+")
        # Comb: ages

        total_viewership_g1 = age_g1.sum()
        total_viewership_g5 = age_g5.sum()

        genre_proportions_g1 = age_g1 / total_viewership_g1
        genre_proportions_g5 = age_g5 / total_viewership_g5

        # Create an array of genre names
        genres = np.unique(np.concatenate((genre_proportions_g1.index, genre_proportions_g5.index)))

        # Create arrays for genre proportions for each age group, filling with 0 for missing genres
        g1_proportions = np.array([genre_proportions_g1.get(genre, 0) for genre in genres])
        g5_proportions = np.array([genre_proportions_g5.get(genre, 0) for genre in genres])

        # Set up the positions for the bars
        bar_width = 0.4
        r1 = np.arange(len(genres))
        r2 = [x + bar_width for x in r1]

        plt.figure(figsize=(12, 6))
        plt.bar(r1, g1_proportions, color='purple', width=bar_width, edgecolor='grey', label='Age: 18-24')
        plt.bar(r2, g5_proportions, color='violet', width=bar_width, edgecolor='grey', label='Age: 65+')

        plt.title('Genre Contribution Proportion by Age Group')
        plt.xlabel('Genre')
        plt.ylabel('Proportion')
        plt.xticks([r + bar_width / 2 for r in range(len(genres))], genres, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Age: Comb")
        # Income group-1
        income_1 = segment_df[segment_df['376'] == 1]
        income_1 = cat(income_1, g_watch)

        income_g1 = income_1['genre_name'].value_counts()

        total_viewership = income_g1.sum()
        genre_proportions = income_g1 / total_viewership

        plt.figure(figsize=(10, 6))
        plt.bar(genre_proportions.index, genre_proportions.values, color='lightblue')
        plt.title('Genre Distribution Proportion for Households in the Income Bracket: $20,000- 29,00')
        plt.xlabel('Genre')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Income: $20,000- 29,000")
        # Income Group: 250k+ (proportion of genre viewership)
        income_l = segment_df[segment_df['386'] == 1]
        income_l = cat(income_l, g_watch)

        income_lg = income_l['genre_name'].value_counts()

        total_viewership = income_lg.sum()
        genre_proportions = income_lg / total_viewership

        plt.figure(figsize=(10, 6))
        plt.bar(genre_proportions.index, genre_proportions.values, color='blue')
        plt.title('Genre Distribution Proportion for the Income Bracket: $250,000+')
        plt.xlabel('Genre')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Income Bracket: $250,000+")

        # Comb: Income

        total_viewership_g1 = income_g1.sum()  # Total viewership within income group $20,000-$29,999
        total_viewership_lg = income_lg.sum()  # Total viewership within income group $250,000+

        genre_proportions_g1 = income_g1 / total_viewership_g1
        genre_proportions_lg = income_lg / total_viewership_lg

        # Create an array of genre names
        genres = np.unique(np.concatenate((genre_proportions_g1.index, genre_proportions_lg.index)))

        # Create arrays for genre proportions for each income group, filling with 0 for missing genres
        g1_proportions = np.array([genre_proportions_g1.get(genre, 0) for genre in genres])
        lg_proportions = np.array([genre_proportions_lg.get(genre, 0) for genre in genres])

        # Set up the positions for the bars
        bar_width = 0.4
        r1 = np.arange(len(genres))
        r2 = [x + bar_width for x in r1]

        plt.figure(figsize=(12, 6))
        plt.bar(r1, g1_proportions, color='lightblue', width=bar_width, edgecolor='grey', label='Income: $20,000-$29,999')
        plt.bar(r2, lg_proportions, color='blue', width=bar_width, edgecolor='grey', label='Income: $250,000+')

        plt.title('Genre Viewership Proportion by Income Group')
        plt.xlabel('Genre')
        plt.ylabel('Proportion')
        plt.xticks([r + bar_width / 2 for r in range(len(genres))], genres, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("EDA Done")