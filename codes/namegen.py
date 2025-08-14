import pandas as pd
import random

firstname = pd.read_csv('data/raw/firstnames.csv')
lastname = pd.read_csv('data/raw/lastnames.csv')

race = ['whi', 'bla', 'his', 'asi', 'oth']
surefirst = firstname[firstname[race].eq(1).any(axis=1)]
surelast = lastname[lastname[race].eq(1).any(axis=1)]

ambifirst = firstname[(firstname[race] > 0.15).all(axis=1)]
ambilast = lastname[(lastname[race] > 0.15).all(axis=1)]

# Assign the race column where the value is 1.0 for each row in surefirst
surefirst['race'] = surefirst[race].idxmax(axis=1)
surelast['race'] = surelast[race].idxmax(axis=1)

# Create list of names per race
surenames = []
for r in race:
    firstnames = surefirst[surefirst['race'] == r]['name'].tolist()
    lastnames = surelast[surelast['race'] == r]['name'].tolist()
    for i in range(50):
        if firstnames and lastnames:
            surenames.append({
                'name': f"{random.choice(firstnames)} {random.choice(lastnames)}",
                'race': r
            })
surenames = pd.DataFrame(surenames)

surenames.to_csv('data/output/surenames.csv', index=False)
print("Sure names generated and saved to data/output/surenames.csv")
