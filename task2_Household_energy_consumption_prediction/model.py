import numpy as np
import pandas as pd
from base_class import BaseMLModel

# IMPLEMENTED MODEL: Ridge Regression with Physics-Based Feature Engineering (Top 10 Strategy)

class Model(BaseMLModel):
    def __init__(self, lambda_reg=10.0):
        super().__init__()
        self.weights = None
        self.lambda_reg = lambda_reg
        self.aux_data = {} # To store homes, weather, etc.

    def _load_aux_data(self):
        """
        Loads auxiliary files (homes, weather, etc.) if not already loaded.
        Assumes files are in the same directory.
        """
        if not self.aux_data:
            try:
                self.aux_data['homes'] = pd.read_csv('homes.csv')
                self.aux_data['rooms'] = pd.read_csv('rooms.csv')
                self.aux_data['persons'] = pd.read_csv('persons.csv')
                self.aux_data['appliances'] = pd.read_csv('appliances.csv')
                self.aux_data['weather'] = pd.read_csv('weather.csv')
            except FileNotFoundError:
                # Fallback: If files are missing, we can't do advanced features.
                # We initialize empty dicts to prevent crashes, but accuracy will drop.
                self.aux_data = {'error': True}

    def _preprocess(self, X):
        """
        Full Pipeline: Merges CSVs and adds Physics Features.
        """
        # 1. Convert Input to DataFrame
        # If X is already the clean numpy matrix (from our notebook), just return it.
        if isinstance(X, np.ndarray) and X.shape[1] > 10: 
            return X
            
        # Otherwise, assume it's the raw dataframe (timestamp, homeid, etc.)
        # If it's a numpy array of objects (raw data passed by auto-grader), convert to DF
        if isinstance(X, np.ndarray):
            # We assume standard column order: timestamp, homeid, roomid, type
            df = pd.DataFrame(X, columns=['timestamp_local', 'homeid', 'roomid', 'type'])
        else:
            df = X.copy()
        
        # 2. Load and Prepare Auxiliary Data
        self._load_aux_data()
        if self.aux_data.get('error'):
            # Emergency Mode: Return simple dummy features if files are missing
            return df.select_dtypes(include=[np.number]).fillna(0).values

        # Prepare Aggregations
        homes = self.aux_data['homes']
        rooms = self.aux_data['rooms']
        
        # Persons Aggregation
        persons = self.aux_data['persons']
        p_agg = persons.groupby('homeid')['personid'].count().reset_index(name='num_residents')
        
        # Appliances Aggregation
        app = self.aux_data['appliances']
        app_agg = app[app['powertype']=='electric'].groupby('homeid')['number'].sum().reset_index(name='num_appliances')
        
        # Weather Aggregation
        weather = self.aux_data['weather']
        w_pivot = weather[weather['weather_type'].isin(['temperature', 'humidity', 'windspeed'])].pivot_table(
            index=['date_hour', 'locationid'], columns='weather_type', values='value', aggfunc='first'
        ).reset_index()
        for c in ['temperature', 'humidity', 'windspeed']:
            w_pivot[c] = pd.to_numeric(w_pivot[c], errors='coerce') * 0.1
        w_pivot['date_hour'] = pd.to_datetime(w_pivot['date_hour'])

        # 3. Merging
        df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
        df = df.merge(homes, on='homeid', how='left')
        df = df.merge(rooms.rename(columns={'type':'room_type'}), on=['homeid','roomid'], how='left')
        df = df.merge(p_agg, on='homeid', how='left')
        df = df.merge(app_agg, on='homeid', how='left')
        
        df['date_hour'] = df['timestamp_local'].dt.round('h')
        df = df.merge(w_pivot, left_on=['date_hour', 'location'], right_on=['date_hour', 'locationid'], how='left')
        
        df['hour'] = df['timestamp_local'].dt.hour
        
        # 4. Physics Features
        df['floorarea'] = pd.to_numeric(df['floorarea'], errors='coerce').fillna(df['floorarea'].median())
        df['temperature'] = df['temperature'].fillna(10)
        df['num_residents'] = df['num_residents'].fillna(0)
        
        # Interaction 1: Heat Loss
        df['physics_heat_loss'] = df['floorarea'] * (20 - df['temperature'])
        
        # Interaction 2: Activity
        df['is_active'] = ((df['hour'] >= 7) & (df['hour'] <= 22)).astype(int)
        df['physics_activity'] = df['num_residents'] * df['is_active']
        
        # 5. Encoding & Cleanup
        cat_cols = ['room_type', 'location', 'hometype']
        df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], dummy_na=True)
        
        # Drop non-numeric
        drop_cols = ['timestamp_local', 'date_hour', 'starttime', 'endtime', 'Predicted', 'Id']
        valid_cols = [c for c in df.columns if c not in drop_cols]
        
        # Force Float
        final_X = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        return final_X

    def fit(self, X, y):
        # 1. Preprocess Data (Merge & Clean)
        X_clean = self._preprocess(X)
        
        X_arr = np.array(X_clean, dtype=float)
        y_arr = np.array(y, dtype=float).flatten()
        
        # 2. Add Bias
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.c_[ones, X_arr]
        
        # 3. Ridge Regression Math
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0
        self.weights = np.linalg.inv(X_b.T @ X_b + self.lambda_reg * I) @ X_b.T @ y_arr
        
        return self

    def predict(self, X):
        # 1. Preprocess Data
        X_clean = self._preprocess(X)
        
        X_arr = np.array(X_clean, dtype=float)
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.c_[ones, X_arr]
        
        if self.weights is None: return np.zeros(len(X))
        
        # 2. Predict
        preds = X_b @ self.weights
        preds[preds < 0] = 0
        return preds
