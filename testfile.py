#!/usr/bin/env python3
"""
High-performance Synthetic FMCG Dataset Generator (5 Million Rows)
Preserves all 33 columns, retail domain logic, seasonality, pricing formulas,
and distributions from fmcg_sales_3years_1M_rows.csv.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

def get_base_sku_catalog(source_csv_path):
    """Extracts or defines the 102 SKU catalog with brand, category, subcategory, list price."""
    if os.path.exists(source_csv_path):
        skus = []
        for chunk in pd.read_csv(source_csv_path, chunksize=200000, usecols=['sku_id', 'sku_name', 'category', 'subcategory', 'brand', 'list_price']):
            skus.append(chunk.drop_duplicates())
        sku_df = pd.concat(skus).drop_duplicates().sort_values('sku_id').reset_index(drop=True)
        return sku_df
    else:
        raise FileNotFoundError(f"Source file not found at: {source_csv_path}")

def generate_store_catalog(num_stores=58):
    """Generates store metadata across diverse European markets and channels."""
    cities_data = [
        # Country, City, Latitude, Longitude, Typical Channels
        ("Germany", "Berlin", 52.52586, 13.39071, ["Hypermarket", "Supermarket", "Convenience", "E-commerce"]),
        ("Germany", "Munich", 48.13512, 11.58198, ["Hypermarket", "Supermarket", "E-commerce"]),
        ("Germany", "Hamburg", 53.55108, 9.99368, ["Hypermarket", "Supermarket"]),
        ("Germany", "Frankfurt", 50.11092, 8.68213, ["Supermarket", "Convenience"]),
        ("Germany", "Cologne", 50.93753, 6.96028, ["Supermarket", "Hypermarket"]),
        ("Italy", "Rome", 41.90278, 12.49637, ["Supermarket", "Hypermarket", "Convenience", "E-commerce"]),
        ("Italy", "Milan", 45.46420, 9.19000, ["Supermarket", "Hypermarket", "E-commerce"]),
        ("Italy", "Naples", 40.85177, 14.26812, ["Supermarket", "Convenience"]),
        ("Italy", "Turin", 45.07031, 7.68686, ["Hypermarket", "Supermarket"]),
        ("Italy", "Florence", 43.76956, 11.25581, ["Supermarket", "Convenience"]),
        ("France", "Paris", 48.85661, 2.35222, ["Hypermarket", "Supermarket", "Convenience", "E-commerce"]),
        ("France", "Lyon", 45.76404, 4.83566, ["Hypermarket", "Supermarket"]),
        ("France", "Marseille", 43.29648, 5.36978, ["Hypermarket", "Supermarket"]),
        ("France", "Toulouse", 43.60465, 1.44421, ["Supermarket", "Convenience"]),
        ("France", "Nice", 43.71017, 7.26195, ["Supermarket", "Convenience"]),
        ("Spain", "Madrid", 40.41677, -3.70379, ["Hypermarket", "Supermarket", "E-commerce"]),
        ("Spain", "Barcelona", 41.38792, 2.16992, ["Supermarket", "Hypermarket", "E-commerce"]),
        ("Spain", "Valencia", 39.46991, -0.37629, ["Supermarket", "Hypermarket"]),
        ("Spain", "Seville", 37.38909, -5.98446, ["Supermarket", "Convenience"]),
        ("Poland", "Warsaw", 52.22968, 21.01223, ["Hypermarket", "Supermarket", "Convenience"]),
        ("Poland", "Krakow", 50.06465, 19.94498, ["Hypermarket", "Supermarket"]),
        ("Poland", "Wroclaw", 51.10788, 17.03854, ["Supermarket", "Convenience"]),
        ("Austria", "Vienna", 48.20817, 16.37382, ["Hypermarket", "Supermarket", "E-commerce"]),
        ("Austria", "Salzburg", 47.80949, 13.05501, ["Supermarket", "Convenience"]),
        ("Netherlands", "Amsterdam", 52.36757, 4.90414, ["Supermarket", "Convenience", "E-commerce"]),
        ("Netherlands", "Rotterdam", 51.92442, 4.47773, ["Hypermarket", "Supermarket"]),
        ("Netherlands", "Utrecht", 52.09074, 5.12142, ["Supermarket", "Convenience"]),
        ("Belgium", "Brussels", 50.85034, 4.35171, ["Hypermarket", "Supermarket"]),
        ("Belgium", "Antwerp", 51.21945, 4.40246, ["Supermarket", "Convenience"]),
        ("Sweden", "Stockholm", 59.32932, 18.06858, ["Hypermarket", "Supermarket", "E-commerce"]),
        ("Portugal", "Lisbon", 38.72225, -9.13934, ["Supermarket", "Hypermarket", "Convenience"]),
    ]

    stores = []
    store_idx = 1
    channel_cycle = ["Hypermarket", "Supermarket", "Supermarket", "Hypermarket", "E-commerce", "Convenience"]

    for i in range(num_stores):
        city_info = cities_data[i % len(cities_data)]
        country, city, lat_base, lon_base, available_channels = city_info
        
        # Add slight jitter to lat/lon for distinct store locations in the same city
        lat = round(lat_base + np.random.uniform(-0.04, 0.04), 5)
        lon = round(lon_base + np.random.uniform(-0.04, 0.04), 5)
        channel = channel_cycle[i % len(channel_cycle)]
        if channel not in available_channels:
            channel = available_channels[0]
            
        store_id = f"STORE{store_idx:04d}"
        stores.append({
            'store_id': store_id,
            'country': country,
            'city': city,
            'channel': channel,
            'latitude': lat,
            'longitude': lon
        })
        store_idx += 1

    return pd.DataFrame(stores)

def generate_date_dimension():
    """Generates 3 years of daily calendar dates (2021-01-01 to 2023-12-31 = 1095 days)."""
    date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    
    holidays = {
        '2021-01-01', '2021-05-01', '2021-08-15', '2021-12-25', '2021-12-26',
        '2022-01-01', '2022-05-01', '2022-08-15', '2022-12-25', '2022-12-26',
        '2023-01-01', '2023-05-01', '2023-08-15', '2023-12-25', '2023-12-26'
    }

    date_df = pd.DataFrame({
        'date': date_range.strftime('%Y-%m-%d'),
        'year': date_range.year,
        'month': date_range.month,
        'day': date_range.day,
        'weekofyear': date_range.isocalendar().week.astype(int),
        'weekday': date_range.weekday, # 0=Mon .. 6=Sun
        'is_weekend': (date_range.weekday >= 5).astype(int),
        'is_holiday': [1 if d.strftime('%Y-%m-%d') in holidays else 0 for d in date_range]
    })
    return date_df

def generate_synthetic_dataset(source_csv, output_csv, target_total_rows=5000000, random_seed=42):
    """Generates exactly target_total_rows synthetic FMCG sales records and streams to CSV."""
    np.random.seed(random_seed)
    start_time = time.time()
    
    print(f"Loading base catalog from {source_csv}...")
    sku_df = get_base_sku_catalog(source_csv)
    num_skus = len(sku_df)
    print(f"Loaded {num_skus} base SKUs.")

    date_df = generate_date_dimension()
    num_days = len(date_df)
    print(f"Generated {num_days} dates (2021-01-01 to 2023-12-31).")

    # Determine store-SKU pairings to hit exactly target_total_rows
    # Each store-sku pairing over 1095 days gives 1,095 rows.
    # Total full pairings = 5000000 // 1095 = 4566 full pairings (4,999,770 rows)
    # Remaining rows = 5000000 % 1095 = 230 rows in one partial pairing.
    total_full_pairings = target_total_rows // num_days
    remaining_days = target_total_rows % num_days

    # We distribute these pairings across ~58 stores (~78-80 SKUs per store)
    num_stores = 58
    store_df = generate_store_catalog(num_stores)
    print(f"Configured {num_stores} stores across European retail channels.")

    # Assign SKUs to stores until we reach total_full_pairings
    store_sku_list = []
    pairings_assigned = 0
    store_idx = 0
    
    while pairings_assigned < total_full_pairings:
        store_row = store_df.iloc[store_idx % num_stores]
        # Pick SKUs for this store
        available_skus = list(range(num_skus))
        np.random.shuffle(available_skus)
        
        needed = total_full_pairings - pairings_assigned
        batch_take = min(needed, min(num_skus, 80))
        
        for i in range(batch_take):
            s_idx = available_skus[i]
            store_sku_list.append((store_row, sku_df.iloc[s_idx], num_days))
            pairings_assigned += 1
            
        store_idx += 1

    # Add remaining rows as a partial pairing if any
    if remaining_days > 0:
        store_row = store_df.iloc[store_idx % num_stores]
        sku_row = sku_df.iloc[np.random.randint(0, num_skus)]
        store_sku_list.append((store_row, sku_row, remaining_days))

    total_expected_rows = sum(p[2] for p in store_sku_list)
    print(f"Configured {len(store_sku_list)} store-SKU streams totaling exactly {total_expected_rows:,} rows.")

    # Base demand per SKU
    # In original data, base mean varies from 15 to 130 depending on category/price
    sku_base_demand = {}
    for idx, row in sku_df.iterrows():
        cat = row['category']
        price = row['list_price']
        # Lower price and beverages/snacks have higher base volume
        if cat == 'Beverages':
            base = max(15.0, 140.0 - price * 6.0 + np.random.uniform(-10, 15))
        elif cat == 'Snacks':
            base = max(15.0, 110.0 - price * 5.0 + np.random.uniform(-10, 10))
        elif cat == 'Personal Care':
            base = max(10.0, 90.0 - price * 4.0 + np.random.uniform(-8, 8))
        elif cat == 'Home Care':
            base = max(10.0, 85.0 - price * 4.0 + np.random.uniform(-8, 8))
        else: # Dairy
            base = max(15.0, 100.0 - price * 5.0 + np.random.uniform(-10, 10))
        sku_base_demand[row['sku_id']] = max(10.0, base)

    # Channel multiplier
    channel_mult = {
        'Hypermarket': 1.0,
        'Supermarket': 0.78,
        'E-commerce': 0.65,
        'Convenience': 0.42
    }

    # Discount bucket distribution: [0.0, 0.10, 0.15, 0.20, 0.30]
    discount_choices = np.array([0.0, 0.10, 0.15, 0.20, 0.30])
    discount_probs = np.array([0.9189, 0.0203, 0.0206, 0.0205, 0.0197])

    # Suppliers S001 to S060
    supplier_pool = [f"S{i:03d}" for i in range(1, 61)]

    # Columns
    columns = [
        'date', 'year', 'month', 'day', 'weekofyear', 'weekday', 'is_weekend', 'is_holiday',
        'temperature', 'rain_mm', 'store_id', 'country', 'city', 'channel',
        'latitude', 'longitude', 'sku_id', 'sku_name', 'category', 'subcategory',
        'brand', 'units_sold', 'list_price', 'discount_pct', 'promo_flag',
        'gross_sales', 'net_sales', 'stock_on_hand', 'stock_out_flag', 'lead_time_days',
        'supplier_id', 'purchase_cost', 'margin_pct'
    ]

    print(f"Streaming data generation to {output_csv}...")
    temp_csv = output_csv + ".tmp"
    
    # Pre-generate weather by city/date
    unique_cities = store_df['city'].unique()
    weather_cache = {}
    for city in unique_cities:
        month_temps = {1: 4.5, 2: 6.0, 3: 9.5, 4: 13.0, 5: 17.5, 6: 21.0, 7: 23.5, 8: 23.0, 9: 18.5, 10: 13.5, 11: 8.0, 12: 5.0}
        city_temps = np.random.normal(loc=[month_temps[m] + np.random.uniform(-1, 1) for m in date_df['month']], scale=3.2)
        city_temps = np.round(np.clip(city_temps, 1.5, 25.0), 2)
        
        city_rain = np.random.exponential(scale=2.8, size=num_days)
        city_rain = np.round(np.where(np.random.random(num_days) < 0.35, 0.0, city_rain), 2)
        weather_cache[city] = (city_temps, city_rain)

    # Open CSV writer
    chunk_buffer = []
    chunk_size = 250000
    rows_written = 0
    first_chunk = True

    # Pre-extract numpy arrays from date_df for instant indexing
    dates_arr = date_df['date'].to_numpy()
    years_arr = date_df['year'].to_numpy()
    months_arr = date_df['month'].to_numpy()
    days_arr = date_df['day'].to_numpy()
    weeks_arr = date_df['weekofyear'].to_numpy()
    weekdays_arr = date_df['weekday'].to_numpy()
    weekends_arr = date_df['is_weekend'].to_numpy()
    holidays_arr = date_df['is_holiday'].to_numpy()

    for p_idx, (store_row, sku_row, n_days) in enumerate(store_sku_list):
        store_id = store_row['store_id']
        country = store_row['country']
        city = store_row['city']
        channel = store_row['channel']
        lat = store_row['latitude']
        lon = store_row['longitude']

        sku_id = sku_row['sku_id']
        sku_name = sku_row['sku_name']
        category = sku_row['category']
        subcategory = sku_row['subcategory']
        brand = sku_row['brand']
        list_price = float(sku_row['list_price'])

        city_temps, city_rain = weather_cache[city]
        temps_subset = city_temps[:n_days]
        rain_subset = city_rain[:n_days]

        # Demand factors
        base_demand = sku_base_demand[sku_id] * channel_mult.get(channel, 1.0)
        
        # Discounts & Promos
        discounts = np.random.choice(discount_choices, size=n_days, p=discount_probs)
        promo_flags = (discounts > 0).astype(np.int64)
        
        # Multipliers
        dow_mult = np.where(weekends_arr[:n_days] == 1, 1.25, 1.0)
        holiday_mult = np.where(holidays_arr[:n_days] == 1, 1.30, 1.0)
        promo_mult = np.where(promo_flags == 1, 1.80, 1.0)

        # Expected units
        expected_units = base_demand * dow_mult * holiday_mult * promo_mult
        noise = np.random.normal(loc=1.0, scale=0.18, size=n_days)
        simulated_units = np.clip(np.round(expected_units * noise), 0, 800).astype(np.int64)

        # Stockouts
        stock_outs = (np.random.random(n_days) < 0.030).astype(np.int64)
        simulated_units = np.where(stock_outs == 1, np.random.randint(0, 15, size=n_days), simulated_units)

        # Inventory & lead times
        stock_on_hand = np.clip(np.random.normal(loc=300, scale=80, size=n_days).round(), 0, 750).astype(np.int64)
        lead_time_days = np.clip(np.random.normal(loc=6.5, scale=2.0, size=n_days).round(), 1, 15).astype(np.int64)
        suppliers = np.random.choice(supplier_pool, size=n_days)

        # Costs & Financials
        cost_ratios = np.random.uniform(0.45, 0.75, size=n_days)
        purchase_costs = np.round(list_price * cost_ratios, 2)

        gross_sales = np.round(simulated_units * list_price, 2)
        net_sales = np.round(gross_sales * (1.0 - discounts), 2)
        margin_pcts = np.round((list_price * (1.0 - discounts) - purchase_costs) / list_price, 3)

        sub_df = pd.DataFrame({
            'date': dates_arr[:n_days],
            'year': years_arr[:n_days],
            'month': months_arr[:n_days],
            'day': days_arr[:n_days],
            'weekofyear': weeks_arr[:n_days],
            'weekday': weekdays_arr[:n_days],
            'is_weekend': weekends_arr[:n_days],
            'is_holiday': holidays_arr[:n_days],
            'temperature': temps_subset,
            'rain_mm': rain_subset,
            'store_id': store_id,
            'country': country,
            'city': city,
            'channel': channel,
            'latitude': lat,
            'longitude': lon,
            'sku_id': sku_id,
            'sku_name': sku_name,
            'category': category,
            'subcategory': subcategory,
            'brand': brand,
            'units_sold': simulated_units,
            'list_price': list_price,
            'discount_pct': discounts,
            'promo_flag': promo_flags,
            'gross_sales': gross_sales,
            'net_sales': net_sales,
            'stock_on_hand': stock_on_hand,
            'stock_out_flag': stock_outs,
            'lead_time_days': lead_time_days,
            'supplier_id': suppliers,
            'purchase_cost': purchase_costs,
            'margin_pct': margin_pcts
        })

        chunk_buffer.append(sub_df)
        rows_in_buffer = sum(len(c) for c in chunk_buffer)

        if rows_in_buffer >= chunk_size or p_idx == len(store_sku_list) - 1:
            batch_df = pd.concat(chunk_buffer, ignore_index=True)
            batch_df.to_csv(temp_csv, mode='a', header=first_chunk, index=False)
            rows_written += len(batch_df)
            first_chunk = False
            chunk_buffer = []
            
            elapsed = time.time() - start_time
            rate = rows_written / elapsed if elapsed > 0 else 0
            print(f"Written {rows_written:,} / {target_total_rows:,} rows ({rows_written/target_total_rows*100:.1f}%) - {rate:,.0f} rows/sec")

    if os.path.exists(output_csv):
        os.remove(output_csv)
    os.rename(temp_csv, output_csv)

    total_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"\nSuccessfully generated {rows_written:,} rows in {total_time:.2f} seconds!")
    print(f"Output File: {output_csv} ({file_size_mb:.2f} MB / {file_size_mb/1024:.2f} GB)")

if __name__ == '__main__':
    source_file = '/home/michaelfernandes/Desktop/Projects/FMCG_Spark/fmcg_sales_3years_1M_rows.csv'
    output_file = '/home/michaelfernandes/Desktop/Projects/FMCG_Spark/fmcg_sales_5M_rows.csv'
    
    target_rows = 5000000
    if len(sys.argv) > 1:
        target_rows = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    generate_synthetic_dataset(source_file, output_file, target_total_rows=target_rows)
