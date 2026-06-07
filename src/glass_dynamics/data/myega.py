import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from glass_dynamics.core.logger import logger

def myega_equation(T: np.ndarray, tg: float, m: float, log_eta_inf: float, log_eta_tg: float = 12.0) -> np.ndarray:
    """
    Calculates the viscosity log10(eta) at given temperatures T using the MYEGA equation.
    
    Args:
        T (np.ndarray): Array of temperatures in Kelvin.
        tg (float): Glass transition temperature in Kelvin.
        m (float): Fragility index.
        log_eta_inf (float): High-temperature limit of viscosity (free parameter).
        log_eta_tg (float): Viscosity at Tg. Default is 12.0 (standard definition in Pa*s).
        
    Returns:
        np.ndarray: Predicted log10(viscosity).
    """
    # MYEGA equation formulation
    term1 = (m / (log_eta_tg - log_eta_inf)) - 1
    term2 = (tg / T) - 1
    
    log_eta = log_eta_inf + (log_eta_tg - log_eta_inf) * (tg / T) * np.exp(term1 * term2)
    return log_eta

def fit_myega(temperatures: np.ndarray, viscosities: np.ndarray, bounds: tuple = None):
    """
    Fits the MYEGA equation to experimental Temperature-Viscosity data, 
    estimating Tg, m, and log_eta_inf.
    
    Args:
        temperatures (np.ndarray): Experimental temperatures.
        viscosities (np.ndarray): Experimental log10(viscosity).
        bounds (tuple, optional): Lower and upper bounds for (Tg, m, log_eta_inf).
                                  Defaults to physically meaningful bounds.
        
    Returns:
        dict: A dictionary containing the fitted parameters ('tg', 'm', 'log_eta_inf') 
              and the fit quality metrics ('r2', 'rmse'). Returns None if the fit fails.
    """
    if bounds is None:
        # Default bounds: 
        # Lower limits: Tg=300 K, m=10, log_eta_inf=-10.0
        # Upper limits: Tg=2500 K, m=200, log_eta_inf=5.0
        bounds = ([300.0, 10.0, -10.0], [2500.0, 200.0, 5.0])
        
    # Initial guess: Tg = 800 K, m = 30, log_eta_inf = -3.0
    p0 = [800.0, 30.0, -3.0] 
    
    try:
        # Curve fitting with 3 free parameters
        popt, pcov = curve_fit(
            lambda T, tg, m, log_eta_inf: myega_equation(T, tg, m, log_eta_inf),
            xdata=temperatures,
            ydata=viscosities,
            p0=p0,
            bounds=bounds,
            maxfev=15000  # Increased max evaluations for 3 parameters
        )
        
        tg_fit, m_fit, log_eta_inf_fit = popt
        
        # Calculate fit metrics
        viscosities_pred = myega_equation(temperatures, tg_fit, m_fit, log_eta_inf_fit)
        r2 = r2_score(viscosities, viscosities_pred)
        rmse = np.sqrt(np.mean((viscosities - viscosities_pred)**2))
        
        return {
            "tg": tg_fit,
            "m": m_fit,
            "log_eta_inf": log_eta_inf_fit,
            "r2": r2,
            "rmse": rmse
        }
        
    except Exception as e:
        logger.debug(f"MYEGA fit failed for the given data points: {e}")
        return None