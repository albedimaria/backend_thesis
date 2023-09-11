
# Define the colors and labels for each sector
sector_colors = ['orangered', 'orange', 'gold', 'yellow',
                 'yellowgreen', 'limegreen', 'green', 'seagreen',
                 'aquamarine', 'lightblue', 'steelblue', 'blue',
                 'blueviolet', 'violet', 'hotpink', 'red']

sector_labels = ['alert', 'excited', 'elated', 'happy',
                  'content', 'serene', 'relaxed', 'calm',
                  'fatigued', 'lethargic', 'depressed', 'sad',
                  'upset', 'stressed', 'nervous', 'tens']

# Define the tempo range thresholds and labels
tempo_ranges = {
    'very slow': (30, 60),
    'slow': (60, 90),
    'moderate-slow': (90, 105),
    'moderate': (105, 120),
    'moderate-fast': (120, 135),
    'fast': (135, 150),
    'very fast': (150, 256)
}