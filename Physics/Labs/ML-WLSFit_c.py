# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:22:42 2018

@author: iandu
"""
# Least squares fit for y = mx + c where m is fixed.
# Returns the best fit parameters, plots showing the data with best-fit line and the residuals, and a second copy of the main plot which includes x=0 to show the intercept.

# Modified by MLloyd to run standalone from command line
# Edit lines 23-26 to include your data

# Put your own data values into arrays x, y and errs, or modify the script to read in data from your own text/csv file into these arrays - see notes on Blackboard about this.
#  Set the value of the slope, m.

#  Then save and run the script.


import numpy as np
import matplotlib.pyplot as plt
import scipy.special  as ss

x=np.array([5.0,6.0,7.0,8.0,9.0])
y=np.array([3.1,3.8,4.9,6.1,7.0])
errs=np.array([0.1,0.2,0.1,0.1,0.2])
m = 1.0  # m is the (known) slope.

def Prob_Chi2_larger( Ndof,  Chi2 ): 
    """   Calculates the probability of getting a chi2 larger than observed
          for the given number of degrees of freedom,   Ndof .
          Uses incomplete gamma function,  defined as 
                \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} dt        """
    p = ss.gammainc( Ndof/2.0, Chi2/2.0  )
    return( p )


def WLSFit_c( x, y, u, m  ):
    """Weighted Least Squares y = mx + c , with fixed m   """
    
    print('\n   Weighted Least Squares fit of x y u data points ')
    print('   to a straight line  y = m*x + c ' )
    print('   with the slope, m, fixed, and  ')
    print('   with the intercept, c, determined by Weighted Least Squares. ')   
   
    
    """
    c = (Sum(w*y)-m*Sum(wx))/Sum(w) =  ywbar - m*xwbar 
    """
 
    num = np.size(x)
    print('\nInput: Number of data points ', num)    
    if num<=1 :
        print('Only one data points; Nothing to fit, Return')
        return( 0.0, 0.0, 0.0, 0.0,  0.0 )
    
    print('\n  Output \n')
    
    wa = u**(-2)
    wbar = np.mean(wa)
    xwbar = np.mean(x*wa)/wbar
    ywbar = np.mean(y*wa)/wbar
    #wxxbar = np.mean( wa*xa*xa ) / wbar
    #wxybar = np.mean( wa*xa*ya ) / wbar
         
    c = ywbar - m * xwbar
    #print('Output: Intercept, c, is  ', c )
    print(' Intercept:  c  =  {0:8.5}'.format(c))
    
   
    u_c =  np.sqrt(   1.0 / (wbar*num) )
    #print(' Uncertainty in intercept, c, is ', c)
    print(' Uncertainty of intercept:   u_c = {0:8.5}  '.format( u_c )  )
    print(' Percentage uncertainty of intercept:  {0:8.4g} %'.format(100.0*(u_c/c)) )
    print('')
   
    
    print(' Mean weighted x:  xwbar =  {0:8.5}'.format(xwbar))
    #print(' Root mean weighted square x: x_rms = {0:8.5} '.format(np.sqrt(Denom)) )
    print(' Mean weighted y:   ywbar = {0:8.5}'.format(ywbar))
    u_ywbar = np.sqrt(  1/(num*wbar ) )
    print(' Uncertainty in ywbar:     ', u_ywbar )
    print(' Uncertainty in ywbar:   {0:8.5}'.format(u_ywbar))
    print(' Percentage uncertainty of ywbar:   {0:8.3} %'.format(100.0*u_ywbar/ywbar))
    print('')
    
   
    resids = y - (m*x + c)
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
    print('index  x-values    y-values    u-values     weights   residuals'\
          '   contribution to chi2')
    for i in range(0,num):
        print('{0:3}{1:12.3g}{2:12.3g}{3:12.3g}{4:12.3g}{5:12.3g}{6:12.3}'.\
              format(i,x[i],y[i],u[i],wa[i],resids[i],G[i]))
    
    """  Plots:  y vs x with uncertainty bars and
             residuals vs x 
             x-range extended by 10%    """

    xm=np.array([0.0,0.0])  # Set up array for min and max values of x
    extra_x = 0.1*(np.amax(x) - np.amin(x))
    xm[0]=np.amin(x) - extra_x  # set min x-val of model to plot
    xm[1]=np.amax(x) + extra_x  # set max x-val of model to plot
    ym = xm*m+c

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

    """ plot including intercept     """ 

    if xm[0]>0.0:
        xm[0]=0.0
    if xm[1]<0.0:
        xm[1]=0.0
    ym = xm * m + c

    plt.xlim(xm[0],xm[1])
    plt.errorbar( x,y,u, fmt='ro'  )  #plot data with errorbars
    plt.plot(xm,ym)  #plot data points
    plt.plot(xm,[0.0,0.0],'black')  #plot x-axis
    plt.plot([0.0,0.0],ym,'black')  #plot y-axis
    plt.show()

    return(  c, u_c, chi2  )

WLSFit_c( x, y, errs, m  )
    
    
    
    
    
    
    
    
    
    
    