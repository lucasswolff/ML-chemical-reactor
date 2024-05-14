import pandas as pd
import numpy as np
import time
        
def calculate_reaction(limit, coef_limit, molar_flow_0, current, num_cells, efficiency, coef, faraday, electrons_transf):
    delta = (current*num_cells*efficiency*coef)/(faraday*electrons_transf)
    
    if abs(delta) > abs(limit/coef_limit*coef):
        delta = limit/coef_limit*coef
        
    molar_flow = molar_flow_0 + delta
    return molar_flow


class Eletrolytic:
    def __init__(self, voltage, enthalpy, efficiency):
        self.voltage = voltage
        self.enthalpy = enthalpy
        self.efficiency = efficiency
        
    def generate_db(self):
        start_time = time.time()
        
        # Constants
        voltage = self.voltage # V
        enthalpy = self.enthalpy # KJ/Kg
        efficiency = self.efficiency
        faraday = 96485.3365
        
        #### 2 NaCl + 2 H2O -> 2e- -> 2 NaOH + Cl2 + H2
        coef_nacl = -2
        coef_h2o = -2
        coef_naoh = 2
        coef_cl2 = 1
        coef_h2 = 1
        electrons_transf = 2 
        
        # ranges of simulations for each variable
        currents = [x for x in range(1000, 20000, 500)]
        num_cells = [x for x in range(2, 300, 2)]
        molar_fraction_nacl_0 = [x/100 for x in range(1, 40, 1)]
        total_molar_flow_0 = [x for x in range(10, 160, 10)]
        
        current_list = []
        cells_list = []
        molar_flow_nacl_0_list = []
        molar_flow_h2o_0_list = []
        molar_flow_total_0_list = []
        molar_flow_nacl_list = []
        molar_flow_h2o_list = []
        molar_flow_h2_list = []
        molar_flow_naoh_list = []
        molar_flow_cl2_list = []
        
        # assuming no initial products in reactor
        molar_fraction_naoh = 0
        molar_fraction_cl2 = 0
        molar_fraction_h2 = 0
        
        
        for current in currents:
            for cell in num_cells:
                for total_flow in total_molar_flow_0:
                    for nacl_fraction in molar_fraction_nacl_0:
                        molar_flow_h2o_0 = total_flow*(1-nacl_fraction)
                        molar_flow_nacl_0 =  total_flow*nacl_fraction
                        
                        molar_flow_naoh_0 = total_flow*molar_fraction_naoh
                        molar_flow_cl2_0 = total_flow*molar_fraction_cl2
                        molar_flow_h2_0 = total_flow*molar_fraction_h2
                        
                        limit = min(molar_flow_h2o_0, molar_flow_nacl_0)
                        
                        if molar_flow_h2o_0 == limit:
                            coef_limit = abs(coef_h2o)
                        else:
                            coef_limit = abs(coef_nacl)
                            
                        molar_flow_h2o = calculate_reaction(limit, coef_limit, molar_flow_h2o_0, current, cell, efficiency, coef_h2o, faraday, electrons_transf)
                        molar_flow_nacl = calculate_reaction(limit, coef_limit, molar_flow_nacl_0, current, cell, efficiency, coef_nacl, faraday, electrons_transf)
                        molar_flow_h2 = calculate_reaction(limit, coef_limit, molar_flow_h2_0, current, cell, efficiency, coef_h2, faraday, electrons_transf)
                        molar_flow_naoh = calculate_reaction(limit, coef_limit, molar_flow_naoh_0, current, cell, efficiency, coef_naoh, faraday, electrons_transf)
                        molar_flow_cl2 = calculate_reaction(limit, coef_limit, molar_flow_cl2_0, current, cell, efficiency, coef_cl2, faraday, electrons_transf)
                        
                        # features
                        current_list.append(current)
                        cells_list.append(cell)
                        molar_flow_h2o_0_list.append(molar_flow_h2o_0)
                        molar_flow_nacl_0_list.append(molar_flow_nacl_0)
                        molar_flow_total_0_list.append(total_flow)
                        
                        # target
                        molar_flow_h2o_list.append(molar_flow_h2o)
                        molar_flow_nacl_list.append(molar_flow_nacl)
                        molar_flow_h2_list.append(molar_flow_h2)
                        molar_flow_naoh_list.append(molar_flow_naoh)
                        molar_flow_cl2_list.append(molar_flow_cl2)
                   
        
        ## CONVERT TO HOURLY MEASURES
        # To kw
        kw_array = np.array(current_list)*voltage/1000
        
        # to kmols/h
        molar_flow_h2o_0_kmolh_array = np.array(molar_flow_h2o_0_list)*3600/1000
        molar_flow_nacl_0_kmolh_array = np.array(molar_flow_nacl_0_list)*3600/1000
        molar_flow_total_0_kmolh_array = np.array(molar_flow_total_0_list)*3600/1000
        molar_flow_h2o_kmolh_array = np.array(molar_flow_h2o_list)*3600/1000
        molar_flow_nacl_kmolh_array = np.array(molar_flow_nacl_list)*3600/1000
        molar_flow_h2_kmolh_array = np.array(molar_flow_h2_list)*3600/1000
        molar_flow_naoh_kmolh_array = np.array(molar_flow_naoh_list)*3600/1000
        molar_flow_cl2_kmolh_array = np.array(molar_flow_cl2_list)*3600/1000
        
        # generate dataframe
        df = pd.DataFrame({
                            'ENERGY': kw_array,
                            'NUM_CELLS': cells_list,
                            'MOLAR_FLOW_H2O_0': molar_flow_h2o_0_kmolh_array,
                            'MOLAR_FLOW_NACL_0': molar_flow_nacl_0_kmolh_array,
                            'MOLAR_FLOW_TOTAL_0': molar_flow_total_0_kmolh_array,
                            'MOLAR_FLOW_H2O': molar_flow_h2o_kmolh_array,
                            'MOLAR_FLOW_NACL': molar_flow_nacl_kmolh_array,
                            'MOLAR_FLOW_H2': molar_flow_h2_kmolh_array,
                            'MOLAR_FLOW_NAOH': molar_flow_naoh_kmolh_array,
                            'MOLAR_FLOW_CL2': molar_flow_cl2_kmolh_array
                           })
        
        # simple assumptions, needs be enhanced with real cost
        df['COST'] = df['MOLAR_FLOW_H2O_0']*2+ df['MOLAR_FLOW_NACL_0']*10 + df['ENERGY']*0.1
        # simple assumptions, needs be enhanced with real sales data
        df['REVENUE'] = df['MOLAR_FLOW_H2']*30 + df['MOLAR_FLOW_NAOH']*20
        
        df['PROFIT'] = df['REVENUE'] - df['COST']
        

        
        # check if mass balance is off
        mm_h2o = 18
        mm_nacl = 58.5
        mm_h2 = 2
        mm_cl2 = 71
        mm_naoh = 40
        
        df['CHECK'] = (round(df['MOLAR_FLOW_H2O_0']*mm_h2o + df['MOLAR_FLOW_NACL_0']*mm_nacl, 5)
                       != 
                       round(df['MOLAR_FLOW_H2O']*mm_h2o + df['MOLAR_FLOW_NACL']*mm_nacl
                       + df['MOLAR_FLOW_H2']*mm_h2 + df['MOLAR_FLOW_NAOH']*mm_naoh
                       + df['MOLAR_FLOW_CL2']*mm_cl2, 5)
                       )
        
        # Get the variables with the maximum profit to be used as initial guess in optimization
        df_input_variables = df[['ENERGY', 'NUM_CELLS', 'MOLAR_FLOW_NACL_0', 
                                 'MOLAR_FLOW_TOTAL_0', 'PROFIT']]
        row_max_profit = df_input_variables.loc[df_input_variables['PROFIT'].idxmax()]
        variables_max_profit = row_max_profit.drop('PROFIT').values
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Script execution time: {elapsed_time:.4f} seconds")
        
        print("Number of row in df: " + str(len(df)))
        
        print("check failed: " + str(df['CHECK'].sum()) + " times")
        
        return df, variables_max_profit

