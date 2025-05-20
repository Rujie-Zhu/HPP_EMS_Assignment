# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:57:55 2023

@author: ruzhu



single imbalance settlement
"""

import pandas as pd
import numpy as np
#from numpy import matlib as mb
#import rainflow
#import math
import Deg_Calculation as DegCal
#import random
#import matplotlib.pyplot as plt
from docplex.mp.model import Model
#import gurobipy as gp
import os
#import openpyxl
from scipy.stats import norm

#from gurobipy import GRB
#import random
#import DEMS
import cplex
#import itertools
#import scipy
#from matplotlib import pyplot as plt
#from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#import time
from datetime import datetime




def ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict): 
    T0 = 96
    datetime_str = simulation_dict["start_date"]
    day_num_start = datetime.strptime(datetime_str, '%m/%d/%y').timetuple().tm_yday
    skips1 = range(1, ((day_num - 1 + day_num_start - 1) * T0)%(359*T0) + 1)
    skips2 = range(1, ((day_num - 1 + day_num_start - 1) * 24)%(359*24) + 1)

    Wind_data = pd.read_csv(simulation_dict['wind_dir'], skiprows = skips1, nrows=T0+exten_num)
    #Wind_scenario = pd.read_csv(simulation_dict['wind_scenario_dir'], skiprows = skips1, nrows=T0+exten_num)
    Solar_data = pd.read_csv(simulation_dict['solar_dir'], skiprows = skips1, nrows=T0+exten_num)
    Market_data = pd.read_csv(simulation_dict['market_dir'], skiprows = skips2, nrows=int(T/DI_num)+int(exten_num/DI_num))
    

    #SP_scenario = np.transpose(np.concatenate((Market_data['SM_forecast'].values.reshape(-1, 1), Market_data['SM_forecast_LEAR'].values.reshape(-1, 1)), axis=1))
    #reg_scenario = np.transpose(np.concatenate((Market_data['reg_forecast_DNN'].values.reshape(-1, 1), Market_data['reg_forecast_LEAR'].values.reshape(-1, 1)), axis=1))
    #probability_price = np.full((len(SP_scenario), 1), 1/len(SP_scenario))
    
    Wind_measurement = Wind_data['Measurement'] * PwMax
    Solar_measurement = Solar_data['Measurement'] * PsMax
    Wind_measurement = Wind_measurement[0:T0:int(4/DI_num)]
    Wind_measurement.index = range(int(T0/(4/DI_num)))
    Solar_measurement = Wind_measurement[0:T0:int(4/DI_num)]
    Solar_measurement.index = range(int(T0/(4/DI_num)))
    
    #Solar_measurement.index = range(int(T/DI_num))
    
    
    # DA_wind_forecast = Wind_data[simulation_dict["DA_wind"]] * PwMax
    # #DA_wind_forecast = Wind_data['Measurement'] * PwMax
    # HA_wind_forecast = Wind_data[simulation_dict["HA_wind"]] * PwMax
    # #HA_wind_forecast = Wind_data['Measurement'] * PwMax
    # RT_wind_forecast = Wind_data[simulation_dict["FMA_wind"]] * PwMax
    # #RT_wind_forecast = Wind_data['Measurement'] * PwMax
    
    # DA_wind_forecast = DA_wind_forecast[0:T0:int(4/DI_num)]
    # DA_wind_forecast.index = range(int(T0/(4/DI_num)))
    # HA_wind_forecast = HA_wind_forecast[0:T0:int(4/DI_num)]
    # HA_wind_forecast.index = range(int(T0/(4/DI_num)))
    # RT_wind_forecast = RT_wind_forecast[0:T0:int(4/DI_num)]
    # RT_wind_forecast.index = range(int(T0/(4/DI_num)))
    

    
    wind_scenario_num = simulation_dict["number_of_wind_scenario"]
    solar_scenario_num = simulation_dict["number_of_solar_scenario"]
    price_scenario_num = simulation_dict["number_of_price_scenario"]
    #HA_wind_forecast_scenario = Wind_data.iloc[:,5:] * PwMax
    
    # #indices = np.linspace(0, len(HA_wind_forecast_scenario.columns) - 1, wind_scenario_num, dtype=int)
    # indices = np.linspace(0, len(HA_wind_forecast_scenario.columns) - 1, wind_scenario_num, dtype=int)   
    # # Define the range of numbers from 0 to 19   
    # # Assume these numbers are evenly spaced in the context of a normal distribution
    # # We can use the mean and std deviation to characterize this distribution
    # mean = indices.mean()
    # std_dev = indices.std()   
    # # Calculate the probability density function (PDF) values for each number
    # pdf_values = norm.pdf(indices, mean, std_dev)   
    # # Normalize these PDF values so that their sum equals 1 (to represent probabilities)
    # probability_wind = pdf_values / pdf_values.sum()       
    #HA_wind_forecast_scenario = HA_wind_forecast_scenario[HA_wind_forecast_scenario.columns[indices]]
    
    indices = ['DA_' + str(i) for i in range(1, wind_scenario_num+1)]
    DA_wind_forecast_scenario = Wind_data[indices]
    DA_wind_forecast_scenario = DA_wind_forecast_scenario.to_numpy().transpose()
    DA_wind_forecast_scenario = DA_wind_forecast_scenario[:,0:T0:int(4/DI_num)] * PwMax

    #probability_wind = [1/wind_scenario_num ]*wind_scenario_num
    
    indices = ['DA_' + str(i) for i in range(1, solar_scenario_num+1)]
    DA_solar_forecast_scenario = Solar_data[indices]
    DA_solar_forecast_scenario = DA_solar_forecast_scenario.to_numpy().transpose()
    DA_solar_forecast_scenario = DA_solar_forecast_scenario[:,0:T0:int(4/DI_num)] * PsMax

    #probability_solar = [1/solar_scenario_num ]*solar_scenario_num

    indices = ['SM_forecast_' + str(i) for i in range(1, price_scenario_num+1)]
    SP_scenario = Market_data[indices]
    SP_scenario = SP_scenario.to_numpy().transpose()
    #SP_scenario = SP_scenario[:,0:T0:int(4/DI_num)] 


    indices = ['reg_forecast_' + str(i) for i in range(1, price_scenario_num+1)]
    RP_scenario = Market_data[indices]
    RP_scenario = RP_scenario.to_numpy().transpose()
    #RP_scenario = RP_scenario[:,0:T0:int(4/DI_num)] 
    
    probability = [1/price_scenario_num ]*price_scenario_num

    # probability_solar = [1/solar_scenario_num ]*solar_scenario_num
    
    # DA_solar_forecast = Solar_data[simulation_dict["DA_solar"]] * PsMax
    # HA_solar_forecast = Solar_data[simulation_dict["HA_solar"]] * PsMax
    # #HA_solar_forecast = Solar_data['Measurement'] * PsMax
    # RT_solar_forecast = Solar_data[simulation_dict["FMA_solar"]] * PsMax
    #RT_solar_forecast = Solar_data['Measurement'] * PsMax

    

    SM_price_cleared = Market_data['SM_cleared'] 
    # SM_price_forecast = Market_data[simulation_dict["SP"]] 
    #SM_price_forecast = Market_data['SM_cleared'] 

    # Reg_price_forecast = Market_data[simulation_dict["RP"]]
    Reg_price_cleared = Market_data['reg_cleared']
    #Reg_price_forecast = Market_data['reg_cleared']

    # BM_dw_price_forecast = pd.DataFrame(columns=['Down'])
    # BM_up_price_forecast = pd.DataFrame(columns=['Up'])
    # reg_up_sign_forecast = pd.DataFrame(columns=['up_sign'])
    # reg_dw_sign_forecast = pd.DataFrame(columns=['dw_sign'])
    
    # for i in range(int(T/DI_num)+int(exten_num/DI_num)):
    #     if Reg_price_forecast.iloc[i] > SM_price_cleared.iloc[i]:
    #         BM_up_price_forecast = pd.concat([BM_up_price_forecast, pd.DataFrame([Reg_price_forecast.iloc[i]], columns=['Up'])], ignore_index=True)
    #         BM_dw_price_forecast = pd.concat([BM_dw_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Down'])], ignore_index=True)
            
    #         reg_up_sign_forecast = pd.concat([reg_up_sign_forecast, pd.DataFrame([1], columns=['up_sign'])], ignore_index=True)
    #         reg_dw_sign_forecast = pd.concat([reg_dw_sign_forecast, pd.DataFrame([0], columns=['dw_sign'])], ignore_index=True)
    #     elif Reg_price_forecast.iloc[i] < SM_price_cleared.iloc[i]:
    #         BM_up_price_forecast = pd.concat([BM_up_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Up'])], ignore_index=True)
    #         BM_dw_price_forecast = pd.concat([BM_dw_price_forecast, pd.DataFrame([Reg_price_forecast.iloc[i]], columns=['Down'])], ignore_index=True)
            
    #         reg_up_sign_forecast = pd.concat([reg_up_sign_forecast, pd.DataFrame([0], columns=['up_sign'])], ignore_index=True)
    #         reg_dw_sign_forecast = pd.concat([reg_dw_sign_forecast, pd.DataFrame([1], columns=['dw_sign'])], ignore_index=True)
    #     else:
    #         BM_up_price_forecast = pd.concat([BM_up_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Up'])], ignore_index=True)
    #         BM_dw_price_forecast = pd.concat([BM_dw_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Down'])], ignore_index=True)   
            
    #         reg_up_sign_forecast = pd.concat([reg_up_sign_forecast, pd.DataFrame([0], columns=['up_sign'])], ignore_index=True)
    #         reg_dw_sign_forecast = pd.concat([reg_dw_sign_forecast, pd.DataFrame([0], columns=['dw_sign'])], ignore_index=True)
    
    # BM_dw_price_forecast = BM_dw_price_forecast.squeeze()
    # BM_up_price_forecast = BM_up_price_forecast.squeeze()
    # reg_up_sign_forecast = reg_up_sign_forecast.squeeze()
    # reg_dw_sign_forecast = reg_dw_sign_forecast.squeeze()        
    
    # if simulation_dict["BP"] == 2:
    #    BM_dw_price_forecast = Market_data['BM_Down_cleared']
    #    BM_up_price_forecast = Market_data['BM_Up_cleared']
    
    
    BM_dw_price_cleared = Market_data['BM_Down_cleared']
    BM_up_price_cleared = Market_data['BM_Up_cleared']
    
  

    reg_vol_up = Market_data['reg_vol_Up']
    reg_vol_dw = Market_data['reg_vol_Down']
    
    time_index = Wind_data['time'][0:T0:int(4/DI_num)]
    return SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_cleared, BM_up_price_cleared, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index, DA_wind_forecast_scenario, DA_solar_forecast_scenario, SP_scenario, RP_scenario, probability 

def f_xmin_to_ymin(x,reso_x, reso_y):  #x: dataframe reso: in hour
    y = pd.DataFrame()
    if reso_y > reso_x:
        a=0
        num = int(reso_y/reso_x)
        
        for ii in range(len(x)):        
            if ii%num == num-1:
                a = (a + x.iloc[ii][0]) /num   
                y = y.append(pd.DataFrame([a]))
                a = 0
            else:                       
                a = a + x.iloc[ii][0]     
        y.index = range(int(len(x)/num))
    else:
        y = pd.DataFrame(np.repeat(x.iloc[:,0],int(reso_x/reso_y)))
        y.index = range(int(24/reso_y))
    return y





        
        
def get_var_value_from_sol(x, sol):
    
    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient='index')
    
    return y
        
 
def SMOpt(deg_sign, dt, ds, dk, T, EBESS, PbMax, PwMax, PsMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    DA_wind_forecast_scenario, DA_solar_forecast_scenario, SP_scenario, RP_scenario, probability, SoC0, exten_num, scenario_num, C_dev, deviation):

    dt_num = int(1/dt) #DI
    

    dk_num = int(1/dk) #BI
    T_dk = int(24/dk)
    
    ds_num = int(1/ds) #SI
    T_ds = int(24/ds)
    dsdt_num = int(ds/dt) 

    # eta_cha_ha = eta_cha**(1/dt_num)
    # eta_dis_ha = eta_dis**(1/dt_num)
    # eta_leak_ha = 1 - (1-eta_leak)**(1/dt_num)

    eta_cha_ha = eta_cha
    eta_dis_ha = eta_dis
    eta_leak_ha = eta_leak
    #       
    setT = [i for i in range(T + exten_num)]  
    setS = [i for i in range(T_ds + int(exten_num/dsdt_num))]
    set_SoCT = [i for i in range(T + 1 + exten_num)] 
     
    setK = [i for i in range(T_dk + int(exten_num/dt_num))]
    
    
    RP_scenario_sub = np.repeat(RP_scenario, ds_num, axis=1)

    
    SMOpt_mdl = Model()

  # Define variables (must define lb and ub, otherwise may cause issues on cplex) 
    
    P_HPP_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, name='SM schedule subhourly')
    P_HPP_SM_k = SMOpt_mdl.continuous_var_dict(setK, lb=0, name='SM schedule hourly')
    P_w_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PwMax, name='SM wind subhourly')
    P_s_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PsMax, name='SM solar subhourly')
    P_dis_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM discharge subhourly') 
    P_cha_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM charge subhourly') 
    E_SM_t   = SMOpt_mdl.continuous_var_dict(set_SoCT, lb=-cplex.infinity, ub=cplex.infinity, name='SM SoC')
    z_t        = SMOpt_mdl.binary_var_dict(setT, name='Cha or Discha')
    


    
   
    
    
    
    
  # Define constraints
    for t in setT:
        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] == P_w_SM_t[t] + P_s_SM_t[t] + P_dis_SM_t[t] - P_cha_SM_t[t])
        SMOpt_mdl.add_constraint(P_dis_SM_t[t] <= (PbMax - PreUp ) * z_t[t] )
        SMOpt_mdl.add_constraint(P_cha_SM_t[t] <= (PbMax - PreDw) * (1-z_t[t]))       

        SMOpt_mdl.add_constraint(E_SM_t[t + 1] == E_SM_t[t] * (1-eta_leak_ha) - (P_dis_SM_t[t])/eta_dis_ha * dt + (P_cha_SM_t[t]) * eta_cha_ha * dt)
        SMOpt_mdl.add_constraint(E_SM_t[t+1] <= SoCmax*Emax )
        SMOpt_mdl.add_constraint(E_SM_t[t+1] >= SoCmin*Emax )
        
    

        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] <= P_grid_limit )
        


 
    for k in setK:
        for j in range(dt_num):
            SMOpt_mdl.add_constraint(P_HPP_SM_t[k * dt_num + j] == P_HPP_SM_k[k])
            SMOpt_mdl.add_constraint(P_HPP_SM_t[k * dt_num + j] == P_HPP_SM_k[k])        


    SMOpt_mdl.add_constraint(E_SM_t[0] == SoC0*Emax)  
    

    
    # second-stage variables and constriants
    
    setV = [i for i in range(scenario_num)]
    setTV = [(i,j) for i in setT for j in setV] 
    setSV = [(i,j) for i in setS for j in setV] 
         
    set_SoCTV = [(i,j) for i in set_SoCT for j in setV]
    
    
    P_tilde_SM_dis_t = SMOpt_mdl.continuous_var_dict(setTV, lb=0, ub=cplex.infinity, name='RT discHArge') 
    P_tilde_SM_cha_t = SMOpt_mdl.continuous_var_dict(setTV, lb=0, ub=cplex.infinity, name='RT charge') 
    P_tilde_w_SM_t = SMOpt_mdl.continuous_var_dict(setTV, lb=0, ub=PwMax, name='RT wind 15min')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    P_tilde_s_SM_t = SMOpt_mdl.continuous_var_dict(setTV, lb=0, ub=PsMax, name='RT solar 15min')
    E_tilde_SM_t   = SMOpt_mdl.continuous_var_dict(set_SoCTV, lb=SoCmin*Emax, ub=Emax, name='RT SoC')
       
    delta_tilde_P_HPP_t = SMOpt_mdl.continuous_var_dict(setTV, lb=-cplex.infinity, ub=cplex.infinity, name='RT imbalance')
    delta_tilde_P_HPP_s = SMOpt_mdl.continuous_var_dict(setSV, lb=-cplex.infinity, ub=cplex.infinity, name='RT imbalance 15min')
    
    tau_s = SMOpt_mdl.continuous_var_dict(setSV, lb=-cplex.infinity, ub=cplex.infinity, name='aux')

    z_tilde_t        = SMOpt_mdl.binary_var_dict(setTV, name='RT Cha or DiscSM')
    

    
    for v in setV:
        for t in setT:
            SMOpt_mdl.add_constraint(P_tilde_SM_dis_t[t,v] <= PbMax*z_tilde_t[t,v] )
            SMOpt_mdl.add_constraint(P_tilde_SM_cha_t[t,v] <= PbMax*(1-z_tilde_t[t,v]) )
            
            
            
            SMOpt_mdl.add_constraint(E_tilde_SM_t[t+1,v] == E_tilde_SM_t[t,v] * (1-eta_leak_ha) - (P_tilde_SM_dis_t[t,v])/eta_dis_ha * dt + (P_tilde_SM_cha_t[t,v]) * eta_cha_ha * dt )
            SMOpt_mdl.add_constraint(E_tilde_SM_t[t+1,v]<=Emax*SoCmax)
            SMOpt_mdl.add_constraint(E_tilde_SM_t[t+1,v]>=Emax*SoCmin)
                       
            SMOpt_mdl.add_constraint(delta_tilde_P_HPP_t[t,v] == P_tilde_w_SM_t[t,v] + P_tilde_s_SM_t[t,v] + P_tilde_SM_dis_t[t,v] - P_tilde_SM_cha_t[t,v]  - P_HPP_SM_t[t])

            

            SMOpt_mdl.add_constraint(P_tilde_w_SM_t[t,v] + P_tilde_s_SM_t[t,v] + P_tilde_SM_dis_t[t,v] - P_tilde_SM_cha_t[t,v]  <= P_grid_limit)
            SMOpt_mdl.add_constraint(P_tilde_w_SM_t[t,v] + P_tilde_s_SM_t[t,v] + P_tilde_SM_dis_t[t,v] - P_tilde_SM_cha_t[t,v]  >= 0)

            SMOpt_mdl.add_constraint(P_tilde_w_SM_t[t,v] <= DA_wind_forecast_scenario[v,t])
            SMOpt_mdl.add_constraint(P_tilde_s_SM_t[t,v] <= DA_solar_forecast_scenario[v,t])


        for s in setS:        
            SMOpt_mdl.add_constraint(tau_s[s,v] >= delta_tilde_P_HPP_s[s,v])
            SMOpt_mdl.add_constraint(tau_s[s,v] >= -delta_tilde_P_HPP_s[s,v])

            SMOpt_mdl.add_constraint(tau_s[s,v] <= deviation)
            
            for t in range(dsdt_num):
                SMOpt_mdl.add_constraint(delta_tilde_P_HPP_t[s*dsdt_num+t,v] == delta_tilde_P_HPP_s[s,v])
            
        SMOpt_mdl.add_constraint(E_tilde_SM_t[0, v] == SoC0*Emax)    
    
                

    if deg_sign == 1:
 
        SMOpt_mdl.maximize(SMOpt_mdl.sum(probability[i]*SP_scenario[i,k]*P_HPP_SM_k[k] for k in setK for i in setV) + SMOpt_mdl.sum(probability[i]*RP_scenario_sub[i,s] * delta_tilde_P_HPP_s[s,i] *ds for i in setV for s in setS)  - SMOpt_mdl.sum(probability[i]*ad*EBESS*mu*(P_tilde_SM_dis_t[t,i] + P_tilde_SM_cha_t[t,i] )*dt for i in setV for t in setT) - SMOpt_mdl.sum(probability[i] * C_dev * tau_s[s,i]*ds for s in setS for i in setV)
                                    )    
    else:
        SMOpt_mdl.maximize(SMOpt_mdl.sum(probability[i]*SP_scenario[i,k]*P_HPP_SM_k[k] for k in setK for i in setV) + SMOpt_mdl.sum(probability[i]*RP_scenario_sub[i,s] * delta_tilde_P_HPP_s[s,i] *ds for i in setV for s in setS)  - SMOpt_mdl.sum(probability[i] * C_dev * tau_s[s,i]*ds for s in setS for i in setV)
                                    )    
    
   # Solve MasterOpt Model
    SMOpt_mdl.print_information()
    sol = SMOpt_mdl.solve()
    aa = SMOpt_mdl.get_solve_details()
    print(aa.status)
    if sol:     
       P_dis_SM_t_opt = get_var_value_from_sol(P_dis_SM_t, sol)   
       P_cha_SM_t_opt = get_var_value_from_sol(P_cha_SM_t, sol)   
       P_w_SM_t_opt   = get_var_value_from_sol(P_w_SM_t, sol)   
       P_HPP_SM_t_opt = get_var_value_from_sol(P_HPP_SM_t, sol)  
       P_HPP_SM_k_opt = get_var_value_from_sol(P_HPP_SM_k, sol)  
       E_SM_t_opt = get_var_value_from_sol(E_SM_t, sol) 
       
       P_tilde_SM_dis_t_opt = get_var_value_from_sol(P_tilde_SM_dis_t, sol) 
       P_tilde_SM_cha_t_opt = get_var_value_from_sol(P_tilde_SM_cha_t, sol) 
       P_tilde_w_SM_t_opt = get_var_value_from_sol(P_tilde_w_SM_t, sol)
       delta_tilde_P_HPP_s_opt = get_var_value_from_sol(delta_tilde_P_HPP_s, sol)
       tau_s_opt = get_var_value_from_sol(tau_s, sol)
       obj = sol.get_objective_value() 
       
    
    return E_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, P_w_SM_t_opt       
           

