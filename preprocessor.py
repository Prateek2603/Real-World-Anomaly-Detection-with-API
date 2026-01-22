import numpy as np

def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculates KM distance between customer and merchant."""
    R = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def transform_raw_to_behavioral(raw_data, history):
    """
    Transforms raw CSV-style columns into behavioral features.
    history: dict containing {'avg_amt': float, 'txn_count': int, 'cat_freq': dict}
    """
    # 1. amt_deviation (uses 'amt')
    amt_deviation = abs(raw_data['amt'] - history.get('avg_amt', raw_data['amt']))
    
    # 2. txn_count_cust (uses 'unix_time' simulated via history)
    txn_count_cust = history.get('txn_count', 1)
    
    # 3. cust_category_count (uses 'category')
    cat_stats = history.get('cat_freq', {})
    cust_category_count = cat_stats.get(raw_data['category'], 0)
    
    # 4. distance_from_home (uses lat, long, merch_lat, merch_long)
    distance = calculate_haversine(
        raw_data['lat'], raw_data['long'], 
        raw_data['merch_lat'], raw_data['merch_long']
    )
    
    return {
        "amt_deviation": amt_deviation,
        "txn_count_cust": txn_count_cust,
        "cust_category_count": cust_category_count,
        "distance_from_home": distance
    }