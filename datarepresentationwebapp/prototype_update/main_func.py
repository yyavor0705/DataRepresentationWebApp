#!/usr/bin/env python

from prototype_update.helper_func import *
from prototype_update.carbonation import Carbonation_Model, load_df_R_ACC



# Input from file
df_pars = read_input_file('data.xlsx')  # from user upload file: df that supplies parameters to pars

# Raw parameters to init Model obj
class Wrapper: pass

pars = Wrapper()  # empty class to store raw parameters

pars.cover_mean = df_pars.cover_mean.values[0]  # mm .values[0] returns a number rather than an array; number works in model.run()
pars.cover_std = df_pars.cover_std.values[0]
pars.RH_real = df_pars.RH_real.values[0]
pars.t_c = df_pars.RH_real.values[0]
pars.x_c = df_pars.x_c.values[0]  # m
pars.ToW = df_pars.ToW.values[0]
pars.p_SR = df_pars.p_SR.values[0]

pars.option = Wrapper()  # empty sub class to store a sub group of raw parameters

pars.option.choose = False                  # from boolean check in the UI
pars.option.cement_type = 'CEM_I_42.5_R+SF' # from drop down select in the UI
                                                # CEM_I_42.5_R
                                                # CEM_I_42.5_R+FA
                                                # CEM_I_42.5_R+SF
                                                # CEM_III/B_42.5
pars.option.wc_eqv = 0.6
pars.option.df_R_ACC = load_df_R_ACC()  # load a df defined in carbonation.py
pars.option.plot = True


# run model
n_year = read_input_user('input year (default 50)')
M = Carbonation_Model(pars)
M.run(n_year)    # more attributes are generated and attached to M
M.postproc(plot=True) # this generates the plot


# calibration to field data
carb_depth_field = read_input_file('data_cal.xlsx')
M_cal, _, _ = M.calibrate(n_year, carb_depth_field.values, print_out=True)

M_cal.report()
