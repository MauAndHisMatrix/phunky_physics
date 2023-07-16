# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:22:42 2018

@author: iandu
"""
# Least squares fit for y = mx.
# Returns the best fit parameters, plots showing the data with best-fit line and the residuals, and a second copy of the main plot which includes the origin.

# Modified by MLloyd to run standalone from command line
# Edit lines 22-24 to include your data

# Put your own data values into arrays x, y and errs, or modify the script to read in data from your own text/csv file into these arrays - see notes on Blackboard about this.

#  Then save and run the script.


import numpy as np
import matplotlib.pyplot as plt
import scipy.special  as ss

x=np.array([1.0,2.0,3.0,4.0,5.0])
y=np.array([1.1,1.8,2.9,4.1,5.0])
errs=np.array([0.1,0.2,0.1,0.1,0.2])


def Prob_Chi2_larger( Ndof,  Chi2 ): 
    """   Calculates the probability of getting a chi2 larger than observed
          for the given number of degrees of freedom,   Ndof .
          Uses incomplete gamma function,  defined as 
                \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} dt        """
    p = ss.gammainc( Ndof/2.0, Chi2/2.0  )
    return( p )


def WLSFit_m( x, y, u  ):
    """Weighted Least Squares y = mx   """
    
    print('\n   Weighted Least Squares fit of x y u data points ')
    print('   to a straight line  y = m*x , that goes through the origin' )
    print('   with the slope,m, determined by Weighted Least Squares  ')   
    
    """
    m =   Sum( wi * yi * xi )  /  Sum( wi * xi * xi )
    """
 
    num = np.size(x)
    print('\nInput: Number of data points ', num)    

    if num<=1 :
        print('Only one data points; Nothing to fit, Return')
        return( 0.0, 0.0, 0.0, 0.0,  0.0 )
    
    print('\n  Output \n')
    
    wa = u**(-2)
    
    Swxy = np.sum( wa * x * y )
    Swxx = np.sum( wa * x * x )
    m = Swxy / Swxx
    u_m = np.sqrt( 1.0 / Swxx )
        
    print(' Slope:  m  =   {0:8.5} '.format(m))
    
    #print('Output: Uncertainty in slope, m, is   ',u_m)
    print(' Uncertainty in slope: u_m =  {0:8.5}'.format(u_m) )
    #print(' Percentage uncertainty of slope is ',   100.0*u_m/m)
    print(' Percentage uncertainty of slope:   {0:8.5} %'.format(100.0*u_m/m))
    print('')    
   
    resids = y - m*x
    G = resids*resids*wa
    chi2 = np.sum((resids*resids)*wa)
    Ndof = num-1
    print(' chi2 = {0:8.4}'.format(chi2) )
    print('         for ', Ndof, 'degrees of freedom')
    #print(' Reduced chi2 is ', chi2/Ndof)
    print(' Reduced chi2 is {0:8.3}'.format(chi2/Ndof))
    
    
    prob = Prob_Chi2_larger(Ndof, chi2)
    print(' Probability of getting a chi2 larger(smaller)')
    print('     than {0:8.4}  is {1:8.3} ({2:6.3} ) '.format( chi2 , 1.0-prob, prob )  )  
    
    if ( (1.0-prob)<0.2 ):
        print(' Warning: Chi2 is large; consider uncertainties underestimated',\
              '\n       or data not consistent with assumed relationship.')
    
    if ( (prob)<0.2 ):
        print(' Warning: Chi2 is small; consider uncertainties overestimated',\
              '\n       or data selected or correlated.')
    
    print('\n Note: Uncertainties are calculated from the u, the uncertainties of the y-values. ')
    print('  and are independent of the value of chi2. ')
   
    print('\n    Summary of data')    
    print('index  x-values    y-values    u-values     weights   residuals ')
    for i in range(0,num):
        print('{0:3}{1:12.3g}{2:12.3g}{3:12.3g}{4:12.3g}{5:12.3g}'.\
              format(i,x[i],y[i],u[i],wa[i],resids[i]))
    
    """  Plots:  y vs x with uncertainty bars and
             residuals vs x 
             x-range extended by 10%    """

    xm=np.array([0.0,0.0])  # Set up array for min and max values of x
    extra_x = 0.1*(np.amax(x) - np.amin(x))
    xm[0]=np.amin(x) - extra_x  # set min x-val of model to plot
    xm[1]=np.amax(x) + extra_x  # set max x-val of model to plot
    ym = xm*m
   
    plt.subplot(2,1,1)
    plt.xlim(xm[0],xm[1])
    plt.ylim(ym[0],ym[1])
    plt.errorbar(x,y,u,fmt='ro')  # plot the data points 
    plt.plot(xm,ym)  #plot the fitted line
  
    plt.subplot(2,1,2)
    plt.xlim(xm[0],xm[1])
    plt.errorbar(x,resids,u,fmt='ro') # plot the uncertainty bars
    plt.plot([xm[0],xm[1]],[0.0,0.0],'black'  )  # plot the line y = 0.0 
    plt.show()

    """ plot showing origin    """

    if xm[0]>0.0:
        xm[0]=0.0
    if xm[1]<0.0:
        xm[1]=0.0
    ym = xm * m 

    plt.xlim(xm[0],xm[1])
    plt.errorbar( x,y,u,fmt='ro')
    plt.plot( xm,ym  )
    plt.show()
    

    return( m, u_m,  chi2  )


    
WLSFit_m( x, y, errs  )
    
    
    
    
    
    
    
    
    