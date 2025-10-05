"""
Test to demonstrate that column ordering is handled correctly by pandas
"""
import pandas as pd

# Required features in expected order
REQUIRED_FEATURES = [
    'koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_impact',
    'koi_model_snr', 'koi_max_mult_ev', 'koi_num_transits',
    'koi_steff', 'koi_srad', 'koi_kepmag', 'koi_insol', 'koi_teq'
]

# Scenario 1: Columns in different order
print("=" * 80)
print("TEST: Column Order Handling in Pandas")
print("=" * 80)

# Create sample data with columns in DIFFERENT order
data_wrong_order = pd.DataFrame({
    'koi_depth': [100, 200, 300],
    'koi_period': [10, 20, 30],
    'koi_prad': [1.5, 2.0, 2.5],
    'koi_duration': [3, 4, 5],
    'koi_impact': [0.1, 0.2, 0.3],
    'koi_model_snr': [20, 25, 30],
    'koi_max_mult_ev': [1.0, 1.5, 2.0],
    'koi_num_transits': [10, 15, 20],
    'koi_steff': [5000, 5500, 6000],
    'koi_srad': [1.0, 1.1, 1.2],
    'koi_kepmag': [14, 15, 16],
    'koi_insol': [1.0, 1.5, 2.0],
    'koi_teq': [300, 350, 400]
})

print("\n📋 Original DataFrame (columns in DIFFERENT order):")
print(f"   Columns: {list(data_wrong_order.columns)}")
print(f"   First row: {data_wrong_order.iloc[0].to_dict()}")

# Reorder columns using pandas indexing
data_correct_order = data_wrong_order[REQUIRED_FEATURES]

print("\n✅ Reordered DataFrame (using data[REQUIRED_FEATURES]):")
print(f"   Columns: {list(data_correct_order.columns)}")
print(f"   First row: {data_correct_order.iloc[0].to_dict()}")

# Verify order is correct
is_correct_order = list(data_correct_order.columns) == REQUIRED_FEATURES
print(f"\n{'✅' if is_correct_order else '❌'} Column order matches expected: {is_correct_order}")

# Show that values are preserved correctly
print("\n📊 Value Comparison:")
print(f"   Original koi_period[0]: {data_wrong_order['koi_period'].iloc[0]}")
print(f"   Reordered koi_period[0]: {data_correct_order['koi_period'].iloc[0]}")
print(f"   Values match: {data_wrong_order['koi_period'].iloc[0] == data_correct_order['koi_period'].iloc[0]}")

print("\n" + "=" * 80)
print("CONCLUSION: Pandas automatically handles column reordering")
print("When you use data[column_list], pandas reorders columns to match the list")
print("This ensures consistent column order for model training/inference")
print("=" * 80)
