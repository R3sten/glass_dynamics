import pandas as pd
import numpy as np

# Import the custom logger
from glass_dynamics.core.logger import logger

def generate_all_features(atomic_df: pd.DataFrame, chemical_props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all possible combinations of Weighted (W) and Absolute (A) features
    using 5 aggregators (Max, Min, Mean, Std, Sum) for all available chemical properties.
    
    Args:
        atomic_df (pd.DataFrame): DataFrame with elemental atomic fractions.
                                  Index represents the glasses, columns are elements.
        chemical_props_df (pd.DataFrame): DataFrame with raw chemical properties.
                                          Index must be the element symbols.
                                          
    Returns:
        pd.DataFrame: A DataFrame containing all generated features (approx. num_properties * 10).
    """
    logger.info(f"Generating all W and A features from {chemical_props_df.shape[1]} chemical properties...")
    
    all_glass_features = []

    for idx, row in atomic_df.iterrows():
        # Get elements present into the currently selected glass compound (> 0)
        present_elements = row[row > 0].index.tolist()
        C_vector = row[present_elements].values # Molar fractions
        
        glass_features = {}

        for prop_name in chemical_props_df.columns:
            # Check if property values exist for the present elements (skip NaNs if any)
            S_vector = chemical_props_df.loc[present_elements, prop_name].values
            
            # W (Weighted): C * S
            W = C_vector * S_vector
            # A (Absolute): S
            A = S_vector

            # Applying the 5 aggregator functions for W
            glass_features[f'W_Max_{prop_name}'] = np.max(W)
            glass_features[f'W_Min_{prop_name}'] = np.min(W)
            glass_features[f'W_Mean_{prop_name}'] = np.mean(W)
            glass_features[f'W_Std_{prop_name}'] = np.std(W) if len(W) > 1 else 0.0
            glass_features[f'W_Sum_{prop_name}'] = np.sum(W)

            # Applying the 5 aggregator functions for A
            glass_features[f'A_Max_{prop_name}'] = np.max(A)
            glass_features[f'A_Min_{prop_name}'] = np.min(A)
            glass_features[f'A_Mean_{prop_name}'] = np.mean(A)
            glass_features[f'A_Std_{prop_name}'] = np.std(A) if len(A) > 1 else 0.0
            glass_features[f'A_Sum_{prop_name}'] = np.sum(A)

        all_glass_features.append(glass_features)

    df_all_features = pd.DataFrame(all_glass_features, index=atomic_df.index)
    
    logger.info(f"Feature generation complete. Total features created: {df_all_features.shape[1]}")
    
    return df_all_features


def generate_cassar_35_features(atomic_df: pd.DataFrame, chemical_props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Directly generates the 35 specific features selected by Cassar in his paper,
    starting from the elemental atomic fractions.
    
    Args:
        atomic_df (pd.DataFrame): DataFrame with elemental atomic fractions.
                                  Index represents the glasses, columns are elements.
        chemical_props_df (pd.DataFrame): DataFrame with chemical properties.
                                          Index must be the element symbols.
                                          
    Returns:
        pd.DataFrame: A DataFrame containing exactly the 35 features used by Cassar.
    """
    logger.info("Generating the 35 specific Cassar features...")
    
    # Cassar's exact selected features
    absolute_features = [
        ('ElectronAffinity', 'std'), ('FusionEnthalpy', 'std'), ('GSenergy_pa', 'std'),
        ('GSmagmom', 'std'), ('NdUnfilled', 'std'), ('NfValence', 'std'),
        ('NpUnfilled', 'std'), ('atomic_radius_rahm', 'std'), ('c6_gb', 'std'),
        ('lattice_constant', 'std'), ('mendeleev_number', 'std'), ('num_oxistates', 'std'),
        ('nvalence', 'std'), ('vdw_radius_alvarez', 'std'), ('vdw_radius_uff', 'std'),
        ('zeff', 'std'),
    ]

    weighted_features = [
        ('FusionEnthalpy', 'min'), ('GSbandgap', 'max'), ('GSmagmom', 'mean'),
        ('GSvolume_pa', 'max'), ('MiracleRadius', 'std'), ('NValence', 'max'),
        ('NValence', 'min'), ('NdUnfilled', 'max'), ('NdValence', 'max'),
        ('NsUnfilled', 'max'), ('SpaceGroupNumber', 'max'), ('SpaceGroupNumber', 'min'),
        ('atomic_radius', 'max'), ('atomic_volume', 'max'), ('c6_gb', 'max'),
        ('c6_gb', 'min'), ('max_ionenergy', 'min'), ('num_oxistates', 'max'),
        ('nvalence', 'min'),
    ]

    # Initialize a list to hold the feature dictionaries for each glass
    all_glasses_features = []

    for idx, row in atomic_df.iterrows():
        # Keep only elements present in the glass (> 0)
        present_elements = row[row > 0].index.tolist()
        C_vector = row[present_elements].values 
        
        glass_features = {}
        
        # 1. Compute Absolute (A) features
        for prop, agg in absolute_features:
            S_vector = chemical_props_df.loc[present_elements, prop].values
            A = S_vector  # Absolute values of elements present
            
            if agg == 'std':
                val = np.std(A) if len(A) > 1 else 0.0
            else:
                # Fallback just in case, though the list only contains 'std'
                val = getattr(np, agg)(A) 
            
            # Note: keeping the exact nomenclature of your notebook (e.g. A_std_...)
            glass_features[f'A_{agg}_{prop}'] = val

        # 2. Compute Weighted (W) features
        for prop, agg in weighted_features:
            S_vector = chemical_props_df.loc[present_elements, prop].values
            W = C_vector * S_vector  # Molar fraction * Property value
            
            if agg == 'std':
                val = np.std(W) if len(W) > 1 else 0.0
            else:
                val = getattr(np, agg)(W)
                
            glass_features[f'W_{agg}_{prop}'] = val
            
        all_glasses_features.append(glass_features)

    df_35_features = pd.DataFrame(all_glasses_features, index=atomic_df.index)
    logger.info(f"Successfully generated {df_35_features.shape[1]} features for {len(df_35_features)} glasses.")
    
    return df_35_features