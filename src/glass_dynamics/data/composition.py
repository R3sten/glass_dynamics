import pandas as pd
import numpy as np
from chemparse import parse_formula
from glass_dynamics.core.logger import logger

def gen_atomic_df(composition_df: pd.DataFrame, chemical_properties: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a dataframe of compound (oxide) fractions into a dataframe 
    of elemental atomic fractions using matrix operations.
    
    Args:
        composition_df (pd.DataFrame): DataFrame where columns are compounds (e.g., 'SiO2', 'Al2O3')
                                       and values are their molar/weight fractions.
        chemical_properties (pd.DataFrame): DataFrame containing chemical properties. 
                                            Columns must represent all possible chemical elements
                                            (usually loaded with .T so elements are columns).
                                            
    Returns:
        pd.DataFrame: A new DataFrame with elemental atomic fractions (rows normalized to sum to 1).
    """
    logger.info("Converting compound fractions to elemental atomic fractions using chemparse...")
    
    compound_lst = composition_df.columns.tolist()
    all_elements = chemical_properties.columns.tolist()

    # Build the transformation matrix (element_guide)
    # Rows correspond to elements, Columns correspond to compounds
    element_guide = np.zeros((len(all_elements), len(compound_lst)))
    
    for j in range(len(compound_lst)):
        c = compound_lst[j]
        cdic = parse_formula(c)
        for el in cdic:
            if el in all_elements:
                i = all_elements.index(el)
                element_guide[i, j] += cdic[el]
            else:
                logger.warning(f"Element '{el}' from compound '{c}' not found in chemical_properties.")

    # Apply transformation to the entire dataframe via vectorized matrix operations
    atomic_df = np.zeros((len(composition_df), len(all_elements)))
    for i in range(len(compound_lst)):
        c = compound_lst[i]
        cdic = parse_formula(c)
        for el in cdic:
            if el in all_elements:
                j = all_elements.index(el)
                # Vectorized multiplication for the entire column
                atomic_df[:, j] += composition_df[c].values * element_guide[j, i]

    # Convert the resulting numpy array back to a Pandas DataFrame
    atomic_df = pd.DataFrame(
        atomic_df,
        columns=all_elements,
        index=composition_df.index,
    )
    
    # Normalize rows so that the sum of atomic fractions for each glass equals 1.0
    atomic_df = atomic_df.div(atomic_df.sum(axis=1), axis=0)

    logger.info(f"Successfully converted. Output shape: {atomic_df.shape}")
    
    return atomic_df