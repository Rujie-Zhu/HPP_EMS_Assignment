# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:05:20 2022

@author: ruzhu
"""

import SEMS as EMS
import utils
from matplotlib import pyplot as plt
import pandas as pd




parameter_dict = {
        
        # hpp parameters
        'hpp_grid_connection': 52,  # in MW 

        # hpp wind parameters
        'wind_capacity': 37.8, #in MW


        # hpp solar parameters
        'solar_capacity': 15,  # in MW
       

        # hpp battery parameters
        'battery_energy_capacity': 30,  # in MWh
        'battery_power_capacity': 10,  # in MW
        'battery_minimum_SoC': 0.05,
        'battery_maximum_SoC': 0.95,
        'battery_initial_SoC': 0.5,
        'battery_hour_discharge_efficiency': 0.985,  #
        'battery_hour_charge_efficiency': 0.975,
        'battery_self_discharge_efficiency': 0,
        # hpp battery degradation parameters
        'battery_initial_degradation': 0,  
        'battery_marginal_degradation_cost': 142000, # in /MWh
        'battery_capital_cost': 142000, # in /MWh
        'degradation_in_optimization': 0, # 1:yes 0:no
        
        # bid parameters
        'max_up_bid': 50,
        'max_dw_bid': 50,
        'min_up_bid': 5,
        'min_dw_bid': 5,
        
        # interval parameters: note that DI must <= SI
        'dispatch_interval': 1,
        'settlement_interval': 1,

        # Allowed deviation between RT and DA by law
        'deviation_MW': 52,  # MW

        # imbalance fee
        'imbalance_fee': 0.13,  
    }

simulation_dict = {
        'wind_as_component': 1,
        'solar_as_component': 1,  
        'battery_as_component': 1,
        'start_date': '1/1/21',
        'number_of_run_day': 3,   # 
        'out_dir':"./test/",

        'DA_wind': "DA",   #DA, Measurement
        'HA_wind': "HA" ,  #HA, Measurement
        'FMA_wind':"RT",#5min_ahead, Measurement
        'DA_solar': "Measurement",
        'HA_solar': "Measurement",
        'FMA_solar': "Measurement",
        'SP': "SM_forecast",  # SM_forecast;SM_cleared
        'RP': "reg_forecast_DNN", #reg_cleared;reg_forecast_pre
        'BP': 1, #1:forecast value 2: perfect value
        
        # Data
        'wind_dir': "../Data/Winddata2021_15min.csv",
        'solar_dir': "../Data/Solardata2021_15min.csv",
        'market_dir': "../Data/Market2021.csv",
               
        # for SEMS
        'number_of_wind_scenario': 3,
        'number_of_solar_scenario': 3, 
        'number_of_price_scenario': 3, 
    }

utils.run_SM(
        parameter_dict = parameter_dict,
        simulation_dict = simulation_dict,
        EMS = EMS,
        EMStype="SEMS"
       )   # run EMS with only spot market optimization
    
  


   
    