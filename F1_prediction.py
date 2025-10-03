"""
F1 Race Winner Predictor using Random Forest
Dataset: Formula 1 World Championship (1950-2020) from Kaggle
+ Nieuwe race voorspellingen
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import pickle
warnings.filterwarnings('ignore')

print("=" * 60)
print("F1 RACE WINNER PREDICTOR - RANDOM FOREST MODEL")
print("=" * 60)

# Laad alle benodigde datasets
print("\n📥 Laden van datasets van Kaggle...")

def load_dataset(file_path):
    """Helper functie om dataset te laden"""
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "rohanrao/formula-1-world-championship-1950-2020",
        file_path
    )

# Laad alle datasets
races = load_dataset("races.csv")
results = load_dataset("results.csv")
drivers = load_dataset("drivers.csv")
constructors = load_dataset("constructors.csv")
qualifying = load_dataset("qualifying.csv")
driver_standings = load_dataset("driver_standings.csv")
constructor_standings = load_dataset("constructor_standings.csv")

print("✅ Datasets geladen!")
print(f"   - Races: {len(races)} records")
print(f"   - Results: {len(results)} records")

# Feature Engineering
print("\n🔧 Feature Engineering...")

def create_features(races_df, results_df, qualifying_df, driver_standings_df, constructor_standings_df):
    """Creëer features voor het ML model"""
    races_df = races_df.sort_values('date').reset_index(drop=True)
    feature_list = []
    
    for idx, race in races_df.iterrows():
        race_id = race['raceId']
        race_results = results_df[results_df['raceId'] == race_id].copy()
        race_qualifying = qualifying_df[qualifying_df['raceId'] == race_id].copy()
        
        if len(race_results) == 0:
            continue
        
        for _, result in race_results.iterrows():
            driver_id = result['driverId']
            constructor_id = result['constructorId']
            
            previous_races = races_df[races_df['raceId'] < race_id]
            previous_results = results_df[results_df['raceId'].isin(previous_races['raceId'])]
            
            driver_history = previous_results[previous_results['driverId'] == driver_id]
            driver_wins = len(driver_history[driver_history['positionOrder'] == 1])
            driver_podiums = len(driver_history[driver_history['positionOrder'] <= 3])
            driver_races = len(driver_history)
            driver_dnf = len(driver_history[driver_history['positionOrder'].isna()])
            
            finished_races = driver_history[driver_history['positionOrder'].notna()]
            driver_avg_pos = finished_races['positionOrder'].mean() if len(finished_races) > 0 else 20
            driver_points = driver_history['points'].sum()
            
            recent_races = driver_history.tail(5)
            recent_avg_pos = recent_races['positionOrder'].mean() if len(recent_races) > 0 else 20
            recent_wins = len(recent_races[recent_races['positionOrder'] == 1])
            
            constructor_history = previous_results[previous_results['constructorId'] == constructor_id]
            constructor_wins = len(constructor_history[constructor_history['positionOrder'] == 1])
            constructor_podiums = len(constructor_history[constructor_history['positionOrder'] <= 3])
            constructor_points = constructor_history['points'].sum()
            
            qual = race_qualifying[race_qualifying['driverId'] == driver_id]
            qualifying_pos = qual['position'].values[0] if len(qual) > 0 and pd.notna(qual['position'].values[0]) else 20
            
            prev_standing = driver_standings_df[
                (driver_standings_df['driverId'] == driver_id) & 
                (driver_standings_df['raceId'] < race_id)
            ].sort_values('raceId', ascending=False)
            
            championship_pos = prev_standing['position'].values[0] if len(prev_standing) > 0 else 20
            championship_points = prev_standing['points'].values[0] if len(prev_standing) > 0 else 0
            
            prev_const_standing = constructor_standings_df[
                (constructor_standings_df['constructorId'] == constructor_id) & 
                (constructor_standings_df['raceId'] < race_id)
            ].sort_values('raceId', ascending=False)
            
            constructor_champ_pos = prev_const_standing['position'].values[0] if len(prev_const_standing) > 0 else 10
            driver_win_rate = (driver_wins / driver_races * 100) if driver_races > 0 else 0
            driver_podium_rate = (driver_podiums / driver_races * 100) if driver_races > 0 else 0
            
            won_race = 1 if result['positionOrder'] == 1 else 0
            
            features = {
                'qualifying_position': qualifying_pos,
                'driver_wins': driver_wins,
                'driver_podiums': driver_podiums,
                'driver_total_races': driver_races,
                'driver_avg_position': driver_avg_pos,
                'driver_total_points': driver_points,
                'driver_win_rate': driver_win_rate,
                'driver_podium_rate': driver_podium_rate,
                'driver_dnf_count': driver_dnf,
                'recent_avg_position': recent_avg_pos,
                'recent_wins': recent_wins,
                'constructor_wins': constructor_wins,
                'constructor_podiums': constructor_podiums,
                'constructor_points': constructor_points,
                'championship_position': championship_pos,
                'championship_points': championship_points,
                'constructor_championship_pos': constructor_champ_pos,
                'won_race': won_race,
                'driver_id': driver_id,
                'race_id': race_id
            }
            
            feature_list.append(features)
    
    return pd.DataFrame(feature_list)

df_features = create_features(races, results, qualifying, driver_standings, constructor_standings)

print(f"✅ Features aangemaakt: {len(df_features)} samples")

df_features = df_features.dropna()
print(f"   - Na cleaning: {len(df_features)} samples")

# ============================================
# TIJDGEBASEERDE SPLIT (BELANGRIJK!)
# ============================================
print("\n🕐 Tijdgebaseerde train/test split...")

# Sorteer op race_id (chronologisch)
df_features = df_features.sort_values('race_id').reset_index(drop=True)

# Gebruik laatste 20% als test set (meest recente races)
split_idx = int(len(df_features) * 0.8)
train_data = df_features.iloc[:split_idx]
test_data = df_features.iloc[split_idx:]

# Bepaal cutoff race voor voorspelling
cutoff_race_id = train_data['race_id'].max()
prediction_race_id = cutoff_race_id + 1  # Voorspel de VOLGENDE race

print(f"   - Training tot race_id: {cutoff_race_id}")
print(f"   - Voorspelling voor race_id: {prediction_race_id}")

# Toon welke races NIET in training zitten
test_race_ids = test_data['race_id'].unique()
test_races_info = races[races['raceId'].isin(test_race_ids)].sort_values('date')

print(f"\n📋 Races NIET in trainingsdata ({len(test_races_info)} races):")
print("=" * 60)
for _, race in test_races_info.iterrows():
    print(f"   Race ID {race['raceId']:4d}: {race['name']:30s} - {race['date']}")
print("=" * 60)

# Split features en labels
X_train = train_data.drop(['won_race', 'driver_id', 'race_id'], axis=1)
y_train = train_data['won_race']
X_test = test_data.drop(['won_race', 'driver_id', 'race_id'], axis=1)
y_test = test_data['won_race']

print(f"\n📊 Dataset splits:")
print(f"   - Training set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")
print(f"   ⚠️  Test set bevat alleen TOEKOMSTIGE races!")

# Random Forest Model
print("\n🌲 Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
print("✅ Model getraind!")

# Evaluatie
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATIE")
print("=" * 60)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 5 belangrijkste features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']:30s} : {row['importance']:.4f}")

# Save model
with open('f1_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\n💾 Model opgeslagen als 'f1_model.pkl'")

# ============================================
# VOORSPEL NIEUWE RACE (zonder data leakage)
# ============================================

def predict_race_winner(model, new_race_data):
    """
    Voorspel winnaar voor nieuwe race
    
    Parameters:
    - model: getraind model
    - new_race_data: DataFrame met driver features
    
    Returns:
    - DataFrame met voorspellingen gesorteerd op kans
    """
    # Verwijder extra kolommen
    feature_cols = X_train.columns.tolist()
    X_new = new_race_data[feature_cols]
    
    # Voorspel kansen
    win_probabilities = model.predict_proba(X_new)[:, 1]
    
    # Maak resultaat DataFrame
    predictions = new_race_data.copy()
    predictions['win_probability'] = win_probabilities * 100
    predictions = predictions.sort_values('win_probability', ascending=False)
    
    return predictions

print("\n" + "=" * 60)
print("🏁 VOORSPEL NIEUWE RACE")
print("=" * 60)

# Toon beschikbare races
available_races = races[races['raceId'] > cutoff_race_id].sort_values('date')
print(f"\n📋 Beschikbare races voor voorspelling (NA training cutoff):")
print("=" * 60)
for _, race in available_races.head(20).iterrows():
    print(f"   ID {race['raceId']:4d}: {race['name']:30s} - {race['date']}")
print("=" * 60)

# Laat gebruiker kiezen
target_race_id = int(input("\n🏎️  Voer race ID in om te voorspellen: "))

# Valideer dat race NA cutoff ligt
if target_race_id <= cutoff_race_id:
    print(f"\n⚠️  WAARSCHUWING: Race {target_race_id} zit in trainingsdata!")
    print(f"   Dit veroorzaakt data leakage. Kies een race > {cutoff_race_id}")
    exit()

# Check of deze race bestaat
if target_race_id in races['raceId'].values:
    target_race = races[races['raceId'] == target_race_id].iloc[0]
    
    print(f"\n📍 Race: {target_race['name']} - {target_race['date']}")
    print(f"   Circuit: {target_race['circuitId']}")
    print(f"   ⚠️  Deze race was NIET in training data!")
    
    # Haal stats op TOT (maar niet inclusief) deze race
    race_results = results[results['raceId'] == target_race_id]
    race_qualifying = qualifying[qualifying['raceId'] == target_race_id]
    
    # Bereken features ALLEEN met data van VOOR deze race
    new_race_features = []
    
    print("\n🏎️  Berekenen van driver kansen (met historische data)...")
    
    for driver_id in race_results['driverId'].unique():
        driver_result = race_results[race_results['driverId'] == driver_id].iloc[0]
        constructor_id = driver_result['constructorId']
        
        # BELANGRIJ: Gebruik alleen data van VOOR deze race (< target_race_id)
        driver_history = results[
            (results['driverId'] == driver_id) & 
            (results['raceId'] < target_race_id)  # ← Strikte ongelijkheid!
        ]
        
        driver_wins = len(driver_history[driver_history['positionOrder'] == 1])
        driver_podiums = len(driver_history[driver_history['positionOrder'] <= 3])
        driver_races = len(driver_history)
        
        finished = driver_history[driver_history['positionOrder'].notna()]
        driver_avg_pos = finished['positionOrder'].mean() if len(finished) > 0 else 20
        driver_points = driver_history['points'].sum()
        
        recent = driver_history.tail(5)
        recent_avg_pos = recent['positionOrder'].mean() if len(recent) > 0 else 20
        recent_wins = len(recent[recent['positionOrder'] == 1])
        
        constructor_history = results[
            (results['constructorId'] == constructor_id) & 
            (results['raceId'] < target_race_id)  # ← Strikte ongelijkheid!
        ]
        constructor_wins = len(constructor_history[constructor_history['positionOrder'] == 1])
        constructor_podiums = len(constructor_history[constructor_history['positionOrder'] <= 3])
        constructor_points = constructor_history['points'].sum()
        
        qual_pos = race_qualifying[race_qualifying['driverId'] == driver_id]['position'].values
        qualifying_pos = qual_pos[0] if len(qual_pos) > 0 else 15
        
        # Standings van VOOR deze race
        standing = driver_standings[
            (driver_standings['driverId'] == driver_id) & 
            (driver_standings['raceId'] < target_race_id)
        ].sort_values('raceId', ascending=False)
        
        championship_pos = standing['position'].values[0] if len(standing) > 0 else 20
        championship_points = standing['points'].values[0] if len(standing) > 0 else 0
        
        const_standing = constructor_standings[
            (constructor_standings['constructorId'] == constructor_id) & 
            (constructor_standings['raceId'] < target_race_id)
        ].sort_values('raceId', ascending=False)
        
        constructor_champ_pos = const_standing['position'].values[0] if len(const_standing) > 0 else 10
        
        driver_win_rate = (driver_wins / driver_races * 100) if driver_races > 0 else 0
        driver_podium_rate = (driver_podiums / driver_races * 100) if driver_races > 0 else 0
        driver_dnf = len(driver_history[driver_history['positionOrder'].isna()])
        
        # Driver info
        driver_info = drivers[drivers['driverId'] == driver_id].iloc[0]
        constructor_info = constructors[constructors['constructorId'] == constructor_id].iloc[0]
        
        new_race_features.append({
            'driver_name': f"{driver_info['forename']} {driver_info['surname']}",
            'constructor': constructor_info['name'],
            'qualifying_position': qualifying_pos,
            'driver_wins': driver_wins,
            'driver_podiums': driver_podiums,
            'driver_total_races': driver_races,
            'driver_avg_position': driver_avg_pos,
            'driver_total_points': driver_points,
            'driver_win_rate': driver_win_rate,
            'driver_podium_rate': driver_podium_rate,
            'driver_dnf_count': driver_dnf,
            'recent_avg_position': recent_avg_pos,
            'recent_wins': recent_wins,
            'constructor_wins': constructor_wins,
            'constructor_podiums': constructor_podiums,
            'constructor_points': constructor_points,
            'championship_position': championship_pos,
            'championship_points': championship_points,
            'constructor_championship_pos': constructor_champ_pos
        })
    
    new_race_df = pd.DataFrame(new_race_features)
    
    # Voorspel!
    predictions = predict_race_winner(rf_model, new_race_df)
    
    print("\n" + "=" * 60)
    print("🏆 RACE VOORSPELLING - TOP 10")
    print("=" * 60)
    
    # Toon ook de werkelijke winnaar als beschikbaar
    actual_winner = results[
        (results['raceId'] == target_race_id) & 
        (results['positionOrder'] == 1)
    ]
    
    if len(actual_winner) > 0:
        winner_id = actual_winner.iloc[0]['driverId']
        winner_info = drivers[drivers['driverId'] == winner_id].iloc[0]
        print(f"\n✅ Werkelijke winnaar: {winner_info['forename']} {winner_info['surname']}")
        print("=" * 60)
    
    for idx, (_, row) in enumerate(predictions.head(10).iterrows(), 1):
        marker = "🏆" if len(actual_winner) > 0 and winner_info['surname'] in row['driver_name'] else "  "
        print(f"\n{marker} {idx}. {row['driver_name']:25s} ({row['constructor']})")
        print(f"   Win kans: {row['win_probability']:.2f}%")
        print(f"   Qualifying: P{int(row['qualifying_position'])} | Championship: P{int(row['championship_position'])}")
        print(f"   Seizoen stats: {int(row['driver_wins'])} wins, {int(row['driver_podiums'])} podiums")
else:
    print(f"\n⚠️  Race {target_race_id} niet gevonden in dataset")
    print("   Probeer een andere race_id of train op meer recente data")