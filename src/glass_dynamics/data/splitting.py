import pandas as pd
from sklearn.model_selection import train_test_split
from glass_dynamics.core.logger import logger

def split_by_cassar_indices(X: pd.DataFrame, y: pd.DataFrame, test_ids_path: str, index_col: str = 'idx') -> tuple:
    """
    Splits the features and targets using explicitly provided indices (e.g., Cassar's test set).
    Assumes that the index of X and y corresponds to the indices provided in the test_ids file.
    
    Args:
        X (pd.DataFrame): The features dataframe.
        y (pd.DataFrame): The targets dataframe.
        test_ids_path (str): Path to the parquet or csv file containing the test indices.
        index_col (str): The column name in the test_ids file that contains the indices to match.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting dataset using predefined indices from {test_ids_path}...")
    
    try:
        # Load the predefined test indices
        if test_ids_path.endswith('.parquet'):
            test_ids_df = pd.read_parquet(test_ids_path)
        elif test_ids_path.endswith('.csv'):
            test_ids_df = pd.read_csv(test_ids_path)
        else:
            raise ValueError("Unsupported file format for test indices. Use .parquet or .csv")
            
        if index_col not in test_ids_df.columns:
            raise ValueError(f"Column '{index_col}' not found in the test_ids file.")
            
        test_idx_list = test_ids_df[index_col].values
        
        # Split data based on index matching
        X_test = X[X.index.isin(test_idx_list)]
        y_test = y[y.index.isin(test_idx_list)]
        
        # Drop the test indices to form the training set
        X_train = X.drop(test_idx_list, errors='ignore')
        y_train = y.drop(test_idx_list, errors='ignore')
        
        logger.info(f"Split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Failed to split data using Cassar indices: {e}")
        raise

def split_by_random_seed(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.10, random_state: int = 42) -> tuple:
    """
    Splits the features and targets randomly, ensuring reproducibility via a random seed.
    
    Args:
        X (pd.DataFrame): The features dataframe.
        y (pd.DataFrame): The targets dataframe.
        test_size (float): Proportion of the dataset to include in the test split (e.g., 0.10 for 10%).
        random_state (int): Seed used by the random number generator.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting dataset randomly (test_size={test_size*100:.0f}%, seed={random_state})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    logger.info(f"Split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test