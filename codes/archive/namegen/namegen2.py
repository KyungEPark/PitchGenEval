import pandas as pd

# Onomastic approach
# US:
# Germany:  Huschka, D., & Wagner, G. G. (2012). Statistical Problems and Solutions in Onomastic Research: Exemplified by a Comparison of Given Name Distributions in Germany throughout the 20th Century. Onoma, 47, 329-365.
# France: INSEE (Institut national de la statistique et des études économiques)
# Italy: Istituto Nazionale di Statistica (ISTAT) - although newborns
# Spain: Instituto Nacional de Estadística (INE)


name = [
    "James Smith", "Mary Smith",            # US
    "Hans Müller", "Maria Müller",          # Germany
    "Gabriel Martin", "Louise Martin",      # France
    "Leonardo Rossi", "Sofia Rossi",        # Italy
    "Antonio García", "María Carmen García",# Spain
    "Oliver Brown", "Emily Brown",          # UK
    "Wei Zhang", "Li Wang",                 # China
    "Takashi Tanaka", "Yuki Sato",          # Japan
    "Aarav Kumar", "Saanvi Kumar",          # India
    "Carlos Silva", "Maria Silva",          # Brazil
    "Ivan Ivanov", "Anna Ivanova",          # Russia
    "Min-jun Kim", "Seo-yeon Kim",          # South Korea
    "Nguyen Van An", "Nguyen Thi Hoa",      # Vietnam
    "Liam Johnson", "Olivia Johnson",       # Canada
    "Jack Wilson", "Charlotte Wilson",      # Australia
    "Ahmed Hassan", "Fatima Hassan",        # Egypt
    "Thabo Nkosi", "Naledi Nkosi",          # South Africa
    "Chinedu Okafor", "Adaeze Okafor",      # Nigeria
    "José Hernández", "Guadalupe Hernández",# Mexico
    "Abdullah Al-Saud", "Aisha Al-Saud",    # Saudi Arabia
    "Juan Pérez", "María González",         # Argentina
    "Mehmet Yılmaz", "Elif Yılmaz",         # Turkey
    "Agus Santoso", "Siti Nurhaliza",       # Indonesia
    "Muhammad Khan", "Ayesha Khan",         # Pakistan
    "Jan Kowalski", "Anna Kowalska",        # Poland
    "Somchai Chaiyaporn", "Suda Chaiyaporn",# Thailand
    "Erik Johansson", "Anna Andersson",     # Sweden
    "John Mwangi", "Grace Wanjiku",         # Kenya
    "Juan Rodríguez", "Catalina Rodríguez", # Colombia
]

country = [
    'US', 'US',
    'DE', 'DE',
    'FR', 'FR',
    'IT', 'IT',
    'ES', 'ES',
    'UK', 'UK',
    'CN', 'CN',
    'JP', 'JP',
    'IN', 'IN',
    'BR', 'BR',
    'RU', 'RU',
    'KR', 'KR',
    'VN', 'VN',
    'CA', 'CA',
    'AU', 'AU',
    'EG', 'EG',
    'ZA', 'ZA',
    'NG', 'NG',
    'MX', 'MX',
    'SA', 'SA',
    'AR', 'AR',
    'TR', 'TR',
    'ID', 'ID',
    'PK', 'PK',
    'PL', 'PL',
    'TH', 'TH',
    'SE', 'SE',
    'KE', 'KE',
    'CO', 'CO',
]

gender = ["m", "f"] * (len(name) // 2)

df = pd.DataFrame({
    "Country": country,
    "Name": name,
    "Gender": gender
})

df.to_csv('data/output/founder_names.csv', index=False)

print(df.head(10))
