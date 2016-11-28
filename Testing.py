
# coding: utf-8

# # Testing the accuracy of the fit

# In[1]:

get_ipython().run_cell_magic('capture', '', '%run FittingExercise.ipynb')


# In[2]:

true_params = ParameterSet(6.9, # logA in mol/L/s
                49., # Ea kJ/mol
                -13., # ∆H kJ/mol
                -42.# ∆S J/mol/K
                )

print("starting_guess =", starting_guess)
print("optimized_parameters =", optimized_parameters)
print("standard_errors =",standard_errors)

print("How many 'standard errors' from the true value was the optimized value?:")
discrepancy = (array(optimized_parameters) - array(true_params)) / array(standard_errors)
discrepancy = ParameterSet(*discrepancy)
for key,value in discrepancy._asdict().items():
    print(key,value)


# Use Pandas to quickly format the output nicely in a table

# In[3]:

import pandas as pd
pd.DataFrame([starting_guess,true_params,optimized_parameters,standard_errors,discrepancy],
             columns=starting_guess._fields,
             index=['starting_guess','true_params','optimized_parameters','standard_errors','discrepancy'])


# Hopefully the discrepancies (actual error divided by reported standard error $\sigma$) are all within $\pm$ 2 (i.e. $|{\epsilon}|<2\sigma$)

# In[ ]:



