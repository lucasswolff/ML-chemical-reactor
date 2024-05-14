import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

model_yield = tf.keras.models.load_model("pfr_yield_neural_net_relu64_relu32_linear.h5")
model_multi = tf.keras.models.load_model("pfr_multi_neural_net_relu64_relu32_linear.h5")


def objective_function(variables):
    predicted_yield = model_yield.predict(np.array(variables).reshape(1, -1))[0]
    predicted_yield = np.clip(predicted_yield, 0, 0.75)
    return -predicted_yield

def predict_multi(x):
    predicted_multi = model_multi.predict(np.array(x).reshape(1, -1))[0]
    return predicted_multi

# T_MOLAR_FRACTION is smaller than H_MOLAR_FRACTION
def constraint1(x):
    return  x[2] - x[1]

# Sum of fractions is equal to one
def constraint2(x):
    return np.sum(x[1:6]) - 1.0

# deltaT smaller than 84
def constraint3(x):
    delta_T = predict_multi(x)[0] - x[6]
    return 84 - delta_T


class PFR_optimize:
    def __init__(self, current_input):
        self.current_input = current_input

    def optimize(self):
        constraints = ({'type': 'ineq', 'fun': lambda x: constraint1(x)}, 
                       {'type': 'ineq', 'fun': lambda x: constraint2(x)},
                       {'type': 'ineq', 'fun': lambda x: constraint3(x)})
        
        #### total molar flow bonds
        min_total_flow = 100
        max_total_flow = 150
        
        # bounds
        # TOTAL_MOLAR_FLOW, T_MOLAR_FRACTION, H_MOLAR_FRACTION, B_MOLAR_FRACTION, D_MOLAR_FRACTION, M_MOLAR_FRACTION
        # TEMPERATURE_0, PRESSURE
        bnds = ((min_total_flow, max_total_flow), (0, 0.5), (0, 1), (0, 1), (0, 1), (0, 1), (0, 550+273.15), (0, 40))
        
        initial_guess = self.current_input
        
        # optimization
        result = minimize(objective_function, initial_guess, method='COBYLA', 
                          constraints=constraints, bounds = bnds, tol=1e-3, options={'maxiter': 1000})
        
        # results
        optimal_variables = np.array(result.x)
        optimal_yield = result.fun*-1
        
        predicted_multi = predict_multi(optimal_variables)
        deltaT = predicted_multi[0] - optimal_variables[6]

        return optimal_variables, optimal_yield, predicted_multi, deltaT

