import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from create_database import Eletrolytic

voltage, enthalpy, efficiency = 18, -574.178, 0.95

eletrolytic_db = Eletrolytic(voltage, enthalpy, efficiency)

df, variables_max_profit_df = eletrolytic_db.generate_db()

# Define a function that returns the negative predicted profit (to convert maximization to minimization)
def objective_function(variables):
    predicted_profit = model.predict(np.array(variables).reshape(1, -1))
    return -predicted_profit[0]

model = tf.keras.models.load_model("neural_net_relu64_relu62_linear.h5")

# initial guess for the input variables: max profit captured in df
initial_guess = np.array(variables_max_profit_df)

# bounds
# energy, num_cells, molar_flow_nacl_0, molar_flow_total_0
bnds = ((0, None), (0, 800), (0, None), (0, 1000))

def constraint1(x):
    # num_cells is an integer
    return [x[1] - int(x[1]), int(x[1]) - x[1]]

def constraint2(x):
    # molar_flow_nacl_0 is smaller than 40% of molar_flow_total_0
    return  x[3] * 0.4 - x[2]

constraints = ({'type': 'ineq', 'fun': lambda x: constraint1(x)},
               {'type': 'ineq', 'fun': lambda x: constraint2(x)})

# optimization
result = minimize(objective_function, initial_guess, method='COBYLA', 
                  constraints=constraints, bounds = bnds, tol=1e-3, options={'maxiter': 1000})


# result
optimal_variables = result.x
optimal_profit = result.fun*-1

print("Optimal Variables:", optimal_variables)
print("Optimal Profit:", optimal_profit)
