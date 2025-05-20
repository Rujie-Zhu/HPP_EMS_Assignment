import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#import time
from datetime import datetime
import os
import Deg_Calculation as DegCal
import math
from docplex.mp.model import Model

import cplex




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




def Revenue_calculation(parameter_dict, DI_num, T_SI, SI_num, SIDI_num, T, DI, SI, BI, P_HPP_SM_t_opt, P_HPP_RT_ts, P_HPP_RT_refs, SM_price_cleared, Reg_price_cleared, BM_dw_price_cleared, BM_up_price_cleared, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, s_UP_t, s_DW_t, residual_imbalance, exten_num):    
    # Spot market revenue
    SM_price_cleared_DI = SM_price_cleared.repeat(DI_num)
    SM_revenue = (P_HPP_SM_t_opt.squeeze()*SM_price_cleared_DI*DI).sum()
    

    # Regulation revenue
    Reg_price_cleared_DI = Reg_price_cleared.repeat(DI_num)
    #BM_dw_price_cleared_DI = BM_dw_price_cleared.repeat(DI_num)
   
    s_UP_t = pd.Series(s_UP_t)
    s_DW_t = pd.Series(s_DW_t)

    reg_revenue = (s_UP_t*P_HPP_UP_bid_ts.squeeze()*DI*Reg_price_cleared_DI).sum() - (s_DW_t*P_HPP_DW_bid_ts.squeeze()*BI*Reg_price_cleared_DI).sum() 
    
    # Imbalance revenue
    Reg_price_cleared_SI = Reg_price_cleared.repeat(SI_num)
    #BM_dw_price_cleared_SI = BM_dw_price_cleared.repeat(SI_num)
    P_HPP_RT_ts_15min = f_xmin_to_ymin(P_HPP_RT_ts, DI, 1/4)    
    P_HPP_RT_refs_15min = f_xmin_to_ymin(P_HPP_RT_refs, DI, 1/4)
    
    power_imbalance = pd.Series((P_HPP_RT_ts_15min.values -P_HPP_RT_refs_15min.values)[:,0])

    #pos_imbalance = power_imbalance.apply(lambda x: x if x > 0 else 0)
    #neg_imbalance = power_imbalance.apply(lambda x: x if x < 0 else 0)

    #im_revenue = np.sum(pos_imbalance*SI*BM_dw_price_cleared_SI) + np.sum(neg_imbalance*SI*BM_up_price_cleared_SI)
    im_revenue = (power_imbalance * Reg_price_cleared *SI).sum()
    # imbalance fee

    im_fee = (abs(power_imbalance*SI)*parameter_dict["imbalance_fee"]).sum()

    # Balancing market revenue    
    BM_revenue = reg_revenue + im_revenue - im_fee
   
    return SM_revenue, reg_revenue, im_revenue, BM_revenue, im_fee


def get_var_value_from_sol(x, sol):

    y = {}

    for key, var in x.items():
        if isinstance(key, tuple):
            y[key[0],key[1]] = sol.get_var_value(var)
        else:
            y[key] = sol.get_var_value(var) 

    y = pd.DataFrame.from_dict(y, orient='index')


   
    return y



        
def RTSim(dt, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, SoC0, P_HPP_t0, start, P_activated_UP_t, P_activated_DW_t, parameter_dict):  
    #RES_error = Wind_measurement[start] + Solar_measurement[start] - RT_wind_forecast[start] - RT_solar_forecast[start] 


    eta_cha_ha = eta_cha**(dt)
    eta_dis_ha = eta_dis**(dt)
    eta_leak_ha = 1 - (1-eta_leak)**(dt)

    # Optimization modelling by CPLEX    
    set_SoCT = [0, 1] 
    RTSim_mdl = Model()
  # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_W_RT_t   = RTSim_mdl.continuous_var(lb=0, ub=Wind_measurement[start], name='HA Wind schedule')
    P_S_RT_t   = RTSim_mdl.continuous_var(lb=0, ub=Solar_measurement[start], name='HA Solar schedule')
    P_HPP_RT_t = RTSim_mdl.continuous_var(lb=0, ub=P_grid_limit, name='HA schedule without balancing bidding')
    P_dis_RT_t = RTSim_mdl.continuous_var(lb=0, ub=PbMax, name='HA discharge') 
    P_cha_RT_t = RTSim_mdl.continuous_var(lb=0, ub=PbMax, name='HA charge') 
    P_b_RT_t   = RTSim_mdl.continuous_var(lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RT_t   = RTSim_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    z_t        = RTSim_mdl.binary_var(name='Cha or Discha')
    tau_t      = RTSim_mdl.continuous_var(lb=-cplex.infinity, ub=cplex.infinity, name='anxillary')
    tau2_t      = RTSim_mdl.continuous_var(lb=-cplex.infinity, ub=cplex.infinity, name='anxillary2')

 # parameter
    alpha1 = 10
    alpha2 = 1 
    alpha3 = 0.5  
  # Define constraints

    RTSim_mdl.add_constraint(P_HPP_RT_t == P_W_RT_t + P_S_RT_t + P_b_RT_t)
    RTSim_mdl.add_constraint(P_b_RT_t == P_dis_RT_t - P_cha_RT_t)
    RTSim_mdl.add_constraint(P_dis_RT_t <= (PbMax - PreUp) * z_t )
    RTSim_mdl.add_constraint(P_cha_RT_t <= (PbMax - PreDw) * (1-z_t))
    RTSim_mdl.add_constraint(SoC_RT_t[1] == SoC_RT_t[0] * (1-eta_leak_ha) - 1/Emax * P_dis_RT_t/eta_dis_ha * dt + 1/Emax * P_cha_RT_t * eta_cha_ha * dt)
    RTSim_mdl.add_constraint(SoC_RT_t[0]   <= SoCmax )
    RTSim_mdl.add_constraint(SoC_RT_t[0]   >= SoCmin )
    RTSim_mdl.add_constraint(P_HPP_RT_t <= P_grid_limit - PreUp)
    #RTSim_mdl.add_constraint(P_HPP_RT_t >= 0)
    RTSim_mdl.add_constraint(SoC_RT_t[0] == SoC0)
    RTSim_mdl.add_constraint(tau_t >= (P_HPP_RT_t - P_HPP_t0) - parameter_dict["deviation_MW"] )
    #RTSim_mdl.add_constraint(tau_t >= -(P_HPP_RT_t - P_HPP_t0) - parameter_dict["deviation_MW"] )
    RTSim_mdl.add_constraint(tau_t >= 0)
    RTSim_mdl.add_constraint(tau2_t >= (P_HPP_RT_t - P_HPP_t0) )
    RTSim_mdl.add_constraint(tau2_t >= -(P_HPP_RT_t - P_HPP_t0) )
    
    
    #if math.isclose(P_activated_UP_t, 0, abs_tol=1e-5) and math.isclose(P_activated_DW_t, 0, abs_tol=1e-5):
    obj = alpha1*tau_t + alpha2 * (Wind_measurement[start] + Solar_measurement[start] - P_W_RT_t - P_S_RT_t) + alpha3*tau2_t
   
   
    RTSim_mdl.minimize(obj)

  # Solve BMOpt Model
    RTSim_mdl.print_information()
    sol = RTSim_mdl.solve()
    aa = RTSim_mdl.get_solve_details()
    print(aa.status)
    if sol:
    #    SMOpt_mdl.print_solution()
        #imbalance_RT_to_ref = sol.get_objective_value() * dt
        P_HPP_RT_t_opt = sol.get_value(P_HPP_RT_t)
        P_W_RT_t_opt = sol.get_value(P_W_RT_t)
        P_S_RT_t_opt = sol.get_value(P_S_RT_t)
        P_dis_RT_t_opt = sol.get_value(P_dis_RT_t)
        P_cha_RT_t_opt = sol.get_value(P_cha_RT_t)
        SoC_RT_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_RT_t), orient='index')
        E_HPP_RT_t_opt = P_HPP_RT_t_opt * dt
        
        RES_RT_cur_t_opt = Wind_measurement[start] + Solar_measurement[start] - P_W_RT_t_opt - P_S_RT_t_opt
        #P_W_RT_cur_t_opt = Wind_measurement[start] - P_W_RT_t_opt
        #P_W_RT_cur_t_opt = pd.DataFrame(P_W_RT_cur_t_opt)
        #P_S_RT_cur_t_opt = Solar_measurement[start] - P_S_RT_t_opt
        #P_S_RT_cur_t_opt = pd.DataFrame(P_S_RT_cur_t_opt)


        z_t_opt = sol.get_value(z_t)

    else:
        print("RTOpt has no solution")
        #print(SMOpt_mdl.export_to_string())
    return E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt

def run_SM(parameter_dict, simulation_dict, EMS, EMStype):
    DI = parameter_dict["dispatch_interval"]
    DI_num = int(1/DI)    
    T = int(1/DI*24)
        
    
    SI = parameter_dict["settlement_interval"]
    SI_num = int(1/SI)
    T_SI = int(24/SI)
    SIDI_num = int(SI/DI)
    
  
    BI = 1
    BI_num = int(1/BI)
    T_BI = int(24/BI)
    
    Wind_component = simulation_dict["wind_as_component"]
    Solar_component = simulation_dict["solar_as_component"]
    BESS_component = simulation_dict["battery_as_component"]
    
    PwMax = parameter_dict["wind_capacity"] * Wind_component
    PsMax = parameter_dict["solar_capacity"] * Solar_component
    EBESS = parameter_dict["battery_energy_capacity"]     
    PbMax = parameter_dict["battery_power_capacity"] * BESS_component  
    SoCmin = parameter_dict["battery_minimum_SoC"] * BESS_component  
    SoCmax = parameter_dict["battery_maximum_SoC"] * BESS_component
    SoCini = parameter_dict["battery_initial_SoC"] * BESS_component
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"] * BESS_component
    PUPMax = parameter_dict["max_up_bid"] 
    PDWMax = parameter_dict["max_dw_bid"] 
    PUPMin = parameter_dict["min_up_bid"] 
    PDWMin = parameter_dict["min_dw_bid"] 
    
    day_num = 1
    Ini_nld = parameter_dict["battery_initial_degradation"]
    pre_nld = Ini_nld
    SoC0 = SoCini
    ld1 = 0
    nld1 = Ini_nld
    ad = 1e-7   # slope   
    capital_cost = parameter_dict["battery_capital_cost"] # €/MWh 
    replace_percent = 0.2     
    total_cycles = 3500
                             
    PreUp = PreDw = 0
    P_grid_limit = parameter_dict["hpp_grid_connection"]

    mu = parameter_dict["battery_marginal_degradation_cost"]
    
    deg_indicator = parameter_dict["degradation_in_optimization"]
    
    
    
    
    P_HPP_UP_t0 = 0
    P_HPP_DW_t0 = 0
    

    C_dev = parameter_dict["imbalance_fee"]

        
    
    #SoC_all = pd.DataFrame(columns = ['SoC_all'])
    
    exten_num = 0
    out_dir = simulation_dict['out_dir']
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    
    re  = pd.DataFrame(list(), columns=['SM_revenue','reg_revenue','im_revenue','im_fee', 'Deg_cost','Deg_cost_by_cycle'])
    sig = pd.DataFrame(list(), columns=['signal_up','signal_down'])
    cur = pd.DataFrame(list(), columns=['RES_cur'])
    de  = pd.DataFrame(list(), columns=['nld','ld','cycles'])
    ei  = pd.DataFrame(list(), columns=['energy_imbalance'])
    #reg = pd.DataFrame(list(), columns=['bid_up','bid_dw','w_up','w_dw','b_up','b_dw'])
    shc = pd.DataFrame(list(), columns=['SM','dis_SM','cha_SM','w_SM','RT','Ref','dis_RT','cha_RT'])
    slo = pd.DataFrame([ad], columns=['slope'])
    soc = pd.DataFrame(list(), columns=['SoC'])
    #bounds = pd.DataFrame(list(), columns=['UB','LB'])
    #worst_reg = pd.DataFrame(list(), columns=['up','down'])
    #worst_wind = pd.DataFrame(list(), columns=['wind'])
    #times = pd.DataFrame(list(), columns=['time-1','time12'])
    
    sig.to_csv(out_dir+'act_signal.csv',index=False)
    cur.to_csv(out_dir+'curtailment.csv',index=False)
    de.to_csv(out_dir+'Degradation.csv',index=False)
    ei.to_csv(out_dir+'energy_imbalance.csv',index=False)
    #reg.to_csv(out_dir+'reg_bids.csv',index=False)
    re.to_csv(out_dir+'revenue.csv',index=False)
    shc.to_csv(out_dir+'schedule.csv',index=False)
    slo.to_csv(out_dir+'slope.csv',index=False)
    soc.to_csv(out_dir+'SoC.csv',index=False)
    #bounds.to_csv(out_dir+'bounds.csv',index=False)
    #worst_reg.to_csv(out_dir+'worst_reg.csv',index=False)
    #worst_wind.to_csv(out_dir+'worst_wind.csv',index=False)
    #times.to_csv(out_dir+'time.csv',index=False)
    while day_num:
        Emax = EBESS*(1-pre_nld)
        

        SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_cleared, BM_up_price_cleared, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index, DA_wind_forecast_scenario, DA_solar_forecast_scenario, SP_scenario, RP_scenario, probability = EMS.ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)

        
        
        
        
        
        
    # Call EMS Model
        # Run SMOpt
        E_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, P_w_SM_t_opt = EMS.SMOpt(deg_indicator, DI, SI, BI, T, EBESS, PbMax, PwMax, PsMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                           DA_wind_forecast_scenario, DA_solar_forecast_scenario, SP_scenario, RP_scenario, probability, SoC0, exten_num, len(probability), C_dev, parameter_dict["deviation_MW"]) 
        
        #P_HPP_SM_t_opt.index = time_index[:T]
        P_HPP_SM_t_opt.index = range(T)
        #write_results(P_HPP_SM_t_opt, 'results_run.xlsx', (day_num-1)*T, 0, 'power schedule')

                      
        P_HPP_RT_ts = []
        P_HPP_RT_refs = []
        RES_RT_cur_ts = []
        residual_imbalance = []
        SoC_ts = []
        P_dis_RT_ts = []
        P_cha_RT_ts = []
        
        
        s_UP_t = np.zeros(T)
        s_DW_t = np.zeros(T)
        P_HPP_UP_bid_ts = pd.DataFrame(np.zeros(T))
        P_HPP_DW_bid_ts = pd.DataFrame(np.zeros(T))


        


        
        
        
            
        
               
        
              
        # BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
        # BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
        # BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
        # BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
     
        # BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
        # BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
        # BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
        # BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))

        SoC_ts.append({'SoC': SoC0}) 
       
        


        

       
        for i in range(0,24):
            exist_imbalance = 0
            for j in range(0, DI_num):    
                RT_interval = i * DI_num + j
                # run RTSim
                P_activated_UP_t = 0
                P_activated_DW_t = 0
                P_HPP_RT_ref = P_HPP_SM_t_opt.iloc[RT_interval,0] + P_activated_UP_t - P_activated_DW_t
    
                    
                E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                                   Wind_measurement, Solar_measurement, SoC0, P_HPP_RT_ref, RT_interval, P_activated_UP_t, P_activated_DW_t, parameter_dict) 
                SoC0 = SoC_RT_t_opt.iloc[1,0]
                 
                SoC_ts.append({'SoC': SoC0})
                P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt}) 
               
                       
    

                exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                residual_imbalance.append({'energy_imbalance': exist_imbalance}) 
                    



        

               
    

        residual_imbalance = pd.DataFrame(residual_imbalance)
        P_HPP_RT_ts = pd.DataFrame(P_HPP_RT_ts)
        P_HPP_RT_refs = pd.DataFrame(P_HPP_RT_refs)
        P_dis_RT_ts = pd.DataFrame(P_dis_RT_ts)
        P_cha_RT_ts = pd.DataFrame(P_cha_RT_ts)
        RES_RT_cur_ts = pd.DataFrame(RES_RT_cur_ts)
 
        
        SM_revenue, reg_revenue, im_revenue, BM_revenue, im_fee = Revenue_calculation(parameter_dict, DI_num, T_SI, SI_num, SIDI_num, T, DI, SI, BI, P_HPP_SM_k_opt, P_HPP_RT_ts, P_HPP_RT_refs, SM_price_cleared, Reg_price_cleared,BM_dw_price_cleared, BM_up_price_cleared, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, s_UP_t, s_DW_t, residual_imbalance, exten_num)     

          
        SoC_all = pd.read_csv(out_dir+'SoC.csv')
        SoC_ts = pd.DataFrame(SoC_ts)
        if SoC_all.empty:
           SoC_all = SoC_ts
        else:
           SoC_all = pd.concat([SoC_all, SoC_ts]) 
        
        SoC_all = SoC_all.values.tolist()
        
        SoC_for_rainflow = SoC_all
        SoC_for_rainflow = [SoC_for_rainflow[i][0] for i in range(int(day_num*T))]
    
    
        ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t, cycles = DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, day_num)
        
        Deg_cost = (nld - pre_nld)/replace_percent * EBESS * capital_cost
        
    
        if day_num==1:
           Deg_cost_by_cycle = cycles.iloc[0,0]/total_cycles * EBESS * capital_cost  
        else:                
           Deg = pd.read_csv(out_dir+'Degradation.csv') 
           cycle_of_day = Deg.iloc[-1,2] - Deg.iloc[-2,2] 
           Deg_cost_by_cycle = cycle_of_day/total_cycles * EBESS * capital_cost        
    
        P_HPP_RT_ts.index = range(T)
        P_HPP_RT_refs.index = range(T)
        P_dis_RT_ts.index = range(T)
        P_cha_RT_ts.index = range(T)        
        output_schedule = pd.concat([P_HPP_SM_t_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, P_w_SM_t_opt, P_HPP_RT_ts, P_HPP_RT_refs, P_dis_RT_ts, P_cha_RT_ts], axis=1)
        output_revenue = pd.DataFrame([SM_revenue, reg_revenue, im_revenue, im_fee, Deg_cost, Deg_cost_by_cycle]).T
        output_revenue.columns=['SM_revenue','reg_revenue','im_revenue','im_fee', 'Deg_cost','Deg_cost_by_cycle']
        #output_bids = pd.concat([P_HPP_UP_bid_ts, P_HPP_DW_bid_ts,P_w_UP_bid_ts, P_w_DW_bid_ts,P_b_UP_bid_ts, P_b_DW_bid_ts], axis=1)
        output_act_signal = pd.concat([pd.DataFrame(s_UP_t, columns=['signal_up']), pd.DataFrame(s_DW_t, columns=['signal_down'])], axis=1)
        #output_time = pd.concat([pd.DataFrame([run_time], columns=['time-1']), pd.DataFrame([run_time2], columns=['time0'])], axis=1)
        #output_time = pd.concat([pd.DataFrame([run_time], columns=['time-1']), pd.DataFrame([run_time2], columns=['time0'])], axis=1)
        #output_bounds = pd.concat([pd.DataFrame(UBs, columns=['UB']), pd.DataFrame(LBs, columns=['LB'])], axis=1)
        if day_num == 1:
            output_deg = pd.concat([pd.DataFrame([Ini_nld, nld], columns=['nld']), pd.DataFrame([0, ld], columns=['ld']), pd.DataFrame([0, cycles.iloc[0,0]], columns=['cycles'])], axis=1)
        else:
            output_deg = pd.concat([pd.DataFrame([nld], columns=['nld']), pd.DataFrame([ld], columns=['ld']), cycles], axis=1)
        
        #output_worst_reg = pd.concat([P_tilde_HPP_UP_ts, P_tilde_HPP_DW_ts], axis=1)
        #output_worst_wind = xi_tilde_ts
        
        #write_results(output_schedule, 'results_run.xlsx', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        #write_results(output_bids, 'results_run.xlsx', (day_num-1)*T_BI, [0,1], 'power bids')
        #write_results(pd.DataFrame(s_UP_t, columns=['signal_up']), 'results_run.xlsx', (day_num-1)*T, [0], 'act_signal')
        #write_results(pd.DataFrame(s_DW_t, columns=['signal_down']), 'results_run.xlsx', (day_num-1)*T, [1], 'act_signal')
        #write_results(SoC_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'SoC')
        #write_results(residual_imbalance, 'results_run.xlsx', (day_num-1)*T_SI, [0], 'energy imbalance')
        #write_results(RES_RT_cur_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'RES curtailment')
        #write_results(output_revenue, 'results_run.xlsx', day_num-1, [0,1,2,3], 'Revenue')
        #write_results(pd.DataFrame([Ini_nld, nld], columns=['nld']), 'results_run.xlsx', day_num-1, [0], 'Degradation')
        #write_results(pd.DataFrame([ld0, ld], columns=['ld']), 'results_run.xlsx', day_num-1, [1], 'Degradation')
        #write_results(output_schedule, out_dir+'schedule.csv', 'csv', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        output_schedule.to_csv(out_dir+'schedule.csv', mode='a', index=False, header=False)
        output_act_signal.to_csv(out_dir+'act_signal.csv', mode='a', index=False, header=False)
        output_deg.to_csv(out_dir+'Degradation.csv', mode='a', index=False, header=False)
        SoC_ts.to_csv(out_dir+'SoC.csv', mode='a', index=False, header=False)
        residual_imbalance.to_csv(out_dir+'energy_imbalance.csv', mode='a', index=False, header=False)
        RES_RT_cur_ts.to_csv(out_dir+'curtailment.csv', mode='a', index=False, header=False)
        output_revenue.to_csv(out_dir+'revenue.csv', mode='a', index=False, header=False)
        
        #write_results(output_bounds, out_dir+'bounds.csv', 'csv', (day_num-1)*T, [0,1], 'bounds')
        #write_results(output_worst_reg, out_dir+'worst_reg.csv', 'csv', (day_num-1)*T, [0,1], 'worst_reg')
        #write_results(output_worst_wind, out_dir+'worst_wind.csv', 'csv', (day_num-1)*T, [0], 'worst_wind')
        #write_results(output_time, out_dir+'time.csv', 'csv', (day_num-1)*T, [0,1], 'output_time')
        #write_results(output_time, out_dir+'time.csv', 'csv', (day_num-1)*T, [0,1], 'output_time')
        Pdis_all = pd.read_csv(out_dir+'schedule.csv', usecols=[3])
        Pcha_all = pd.read_csv(out_dir+'schedule.csv', usecols=[4])
        nld_all = pd.read_csv(out_dir+'Degradation.csv', usecols=[0])
        ad_all =pd.read_csv(out_dir+'slope.csv', usecols=[0])
        ad = DegCal.slope_update(Pdis_all, Pcha_all, nld_all, day_num, 7, T, DI, ad_all)
        
        pd.DataFrame([ad], columns=['slope']).to_csv(out_dir+'slope.csv', mode='a', index=False, header=False)
        #write_results(pd.DataFrame([ad], columns=['slope']), out_dir+'slope.csv', 'csv', day_num-1, [0], 'Degradation')
        if nld>0.2:
            break
        

        pre_nld = nld
        day_num = day_num + 1 
        if day_num > simulation_dict["number_of_run_day"]:
            print(P_grid_limit)
            break