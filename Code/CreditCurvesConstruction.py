""" Project : quantitative strategies in building credit curves
Done by : Othmane ZARHALI

"""

# Packages
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import scipy.interpolate as inter
import scipy.stats
from scipy.special import gamma, factorial2
from scipy import linalg
#import pyqt_fit


# Parameters :
l=0.4
r=0.01
Rec=0.7
subMaturity = [5, 6,10, 20,30,40]





# CDS curve building ##################################################################################################
# Polynomial Interpolation
def exponential_cdf(l:float, x:float):
    return 1 - np.exp(-l*x)

def CDS_price(l:float, r:float, Rec:float):
    T = [0.25*(i+1) for i in range(100)]  # list of maturities
    spread = np.zeros(100, dtype=float)
    for i in range(100):
        Tinf = np.array([j for j in T if j<=T[i]])
        # print(Tinf)
        # print(np.exp(-r*Tinf)*0.25*(1-var.exponential_cdf(l,Tinf)))
        spread[i]=(1-Rec)*exponential_cdf(l, T[i])/np.sum(np.exp(-r*Tinf)*0.25*(1-exponential_cdf(l,Tinf)))
    CDS_Dataframe={"T":T, "spread":spread}
    res = pd.DataFrame(data=CDS_Dataframe)
    res=res.set_index("T")
    return res

# Test of the previous function
#print(CDS_price(5,0.5,0.05))

def sub_select(df:pd.DataFrame, listSubIndex:list):
    subIndex=np.array(listSubIndex)
    subIndexFiltered=[j for j in df.index.values if (j in subIndex) ]
    subIndexValue = [i/0.25 - 1 for i in subIndexFiltered]
    d={"T":sorted(subIndexFiltered), "spread":df.iloc[subIndexValue, 0]}
    res = pd.DataFrame(data=d)
    res = res.set_index("T")
    return res

def optimal_bandwidth(Xsupport:np.ndarray, Ysupport:np.ndarray):
    # type=="Silvermann":
    sigma_est = 1
    res = 1.06*sigma_est/len(Xsupport)**0.2
    return res

def Newton_method(x:float, Xsupport:np.ndarray, Ysupport:np.ndarray, h:float):
    return np.sum(scipy.stats.norm.pdf((Xsupport - x)/h)*Ysupport)/np.sum(scipy.stats.norm.pdf((Xsupport - x)/h))

def spline_method(x:float, Xsupport:np.ndarray, Ysupport:np.ndarray, order:float=3):
    return inter.spline(Xsupport, Ysupport, x,order=order, kind="smoothest")


def interpolate_linear(curve_CDS, index_input):
    x_input, y_input, x_all = index_input.copy(), np.squeeze(
        curve_CDS.loc[index_input, :].values), curve_CDS.index.values
    yinterp = np.interp(x_all, x_input, y_input)

    plt.plot(x_input, y_input, '*', label='input_points', color='blue')
    plt.plot(x_all, yinterp, '-', label='interpolated')
    # plt.plot(curve_CDS.index.values,np.squeeze(curve_CDS.values),label='true')
    plt.legend(loc='best')
    plt.title('Linear interpolation')
    plt.show()
    print("MSE linear interpolation: ", np.mean((curve_CDS.values - yinterp) ** 2))


def interpolate_spline(curve_CDS, index_input, order=3):
    x_input, y_input, x_all = index_input.copy(), np.squeeze(
        curve_CDS.loc[index_input, :].values), curve_CDS.index.values
    yinterp = spline_method(x_all, x_input, y_input, order=order)

    plt.plot(x_input, y_input, '*', label='input_points', color='blue')
    plt.plot(x_all, yinterp, '-', label='interpolated')
    plt.legend(loc='best')
    plt.title('Spline interpolation, order =%i '%order)
    plt.show()
    print("MSE spline interpolation (order=" + str(order) + "): ", np.mean((curve_CDS.values - yinterp) ** 2))

def interpolate_Newton(curve_CDS, index_input, type="Silvermann"):
    x_input, y_input, x_all = index_input.copy(), np.squeeze(curve_CDS.loc[index_input, :].values), curve_CDS.index.values
    h_opt = optimal_bandwidth(x_input, y_input)
    yinterp = [Newton_method(x, x_input, y_input, h=1) for x in x_all]
    plt.plot(x_input, y_input, '*',label='input_points',color='blue')
    plt.plot(x_all, yinterp, '-x',label='interpolated')
    plt.legend(loc='best')
    plt.title('Newton interpolation')
    plt.show()
    print("MSE Newton interpolation: ", np.mean((curve_CDS.values - yinterp) ** 2))


# Kernel interpolation
def gauss_integral(n):
    r"""
    Solve the integral
    \int_0^1 exp(-0.5 * x * x) x^n dx
    See
    https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions
    Examples
    --------
    >>> ans = gauss_integral(3)
    >>> np.allclose(ans, 2)
    True
    >>> ans = gauss_integral(4)
    >>> np.allclose(ans, 3.75994)
    True
    """
    factor = np.sqrt(np.pi * 2)
    if n % 2 == 0:
        return factor * factorial2(n - 1) / 2
    elif n % 2 == 1:
        return factor * norm.pdf(0) * factorial2(n - 1)
    else:
        raise ValueError("n must be odd or even.")

def epanechnikov(x, dims=1):
    normalization = 2 / (dims + 2)
    dist_sq = x ** 2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = (1 - dist_sq)[mask] / normalization
    return out

def gaussian(x, dims=1):
    normalization = dims * gauss_integral(dims - 1)
    dist_sq = x ** 2
    return np.exp(-dist_sq / 2) / normalization

def box(x, dims=1):
    normalization = 1
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = 1 / normalization
    return out

def exponential(x, dims=1):
    normalization = gamma(dims) * dims
    return np.exp(-x) / normalization

def biweight(x, dims=1):
    normalization = 8 / ((dims + 2) * (dims + 4))
    dist_sq = x ** 2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = np.maximum(0, (1 - dist_sq) ** 2)[mask] / normalization
    return out

def triweight(x, dims=1):
    normalization = 48 / ((dims + 2) * (dims + 4) * (dims + 6))
    dist_sq = x ** 2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = np.maximum(0, (1 - dist_sq) ** 3)[mask] / normalization
    return out

def Kernel_interpolation_function(x,X,Y,kernel=gaussian):
    h = optimal_bandwidth(X, Y)
    return np.sum(kernel((X-x)/h)*Y)/np.sum(kernel((X-x)/h))


def Kernel_interpolation(curve_CDS, index_input,kernel = gaussian):
    x_input, y_input, x_all = index_input.copy(), np.squeeze(curve_CDS.loc[index_input, :].values), curve_CDS.index.values
    yinterp = [Kernel_interpolation_function(x, x_input, y_input,kernel) for x in x_all]
    plt.plot(x_input, y_input, '*', label='input_points', color='blue')
    plt.plot(x_all, yinterp, '-x', label='interpolated')
    plt.legend(loc='best')
    plt.title('Kernel interpolation, Kernel = %s'%kernel)
    plt.show()
    print("MSE Kernel interpolation: ", np.mean((curve_CDS.values - yinterp) ** 2))

# Local polynomial
def Local_Polynomial_Interpolation_function(x,X,Y,order,kernel=gaussian):
    h = optimal_bandwidth(X, Y)
    kernel_list = [float(kernel(((X[i]-x)/h))) for i in range(len(X))]
    D = np.diag(kernel_list)
    X_matrix = np.ones((len(X),order+1))
    shiftedX = X - x
    for i in range(order+1):
        X_matrix[:,i]=np.asarray(shiftedX)**i
    beta = linalg.inv(np.transpose(X_matrix).dot(D.dot(X_matrix))).dot(np.transpose(X_matrix).dot(D.dot(Y)))
    power_x=np.asarray([x**i for i in range(order+1)])
    return np.sum(power_x*beta)

def Local_Polynomial_Interpolation(curve_CDS, index_input, kernel=gaussian,order=0):
    x_input, y_input, x_all = index_input.copy(), np.squeeze(curve_CDS.loc[index_input, :].values), curve_CDS.index.values
    # print("y input = ", y_input)
    # print("x_input = ", x_input)
    yinterp = [float(Local_Polynomial_Interpolation_function(x, x_input, y_input,order, kernel)) for x in x_all]
    plt.plot(x_input, y_input, '*', label='input_points', color='blue')
    plt.plot(x_all, yinterp, '-x', label='interpolated')
    plt.legend(loc='best')
    plt.title('Local Polynomial interpolation, Kernel = %s' % kernel)
    plt.show()
    print("MSE Local Polynomial: ", np.mean((curve_CDS.values - yinterp) ** 2))



            # TEST INTERPOLATIONS
curveCDS = CDS_price(l,r, Rec)
#print(curveCDS)

########## test on average
index_all = np.array(curveCDS.index.values)
index_input = np.sort(np.unique(np.random.random_integers(0,len(curveCDS)-1,20)))
# interpolate_linear(curveCDS,index_all[index_input])
# interpolate_spline(curveCDS,index_all[index_input], order=1)
# interpolate_spline(curveCDS,index_all[index_input], order=2)
# interpolate_spline(curveCDS,index_all[index_input], order=3)
# interpolate_Newton(curveCDS,index_all[index_input])
# Kernel_interpolation(curveCDS, index_all[index_input]) #Gaussian
# Kernel_interpolation(curveCDS, index_all[index_input],epanechnikov) #epanechnikov
# Kernel_interpolation(curveCDS, index_all[index_input],biweight) #biweight
# Kernel_interpolation(curveCDS, index_all[index_input],triweight) #triweight
# Local_Polynomial_Interpolation(curveCDS, index_all[index_input],gaussian,3)

# Boostrapping the survival curve ######################################################################################
def inverse_spread_function(v, x,y,r, Rec ):
    ZC = np.exp(-r*x)
    num = -y*0.25*(np.sum(ZC[1:-1]*(v[:-2] + v[1:-1])) + ZC[-1]*v[-2]) + (1-Rec)*(np.sum((ZC[1:-1]+ ZC[:-2])*(-v[1:-1]+ v[:-2])) + (ZC[-2] + ZC[-1])* v[-2])
    denom= 0.25*y*ZC[-1] + (1-Rec)*(ZC[-2]+ZC[-1])
    res = num/denom
    return res

def recursive_bootstrap(x, y,r, Rec):
    res = np.ones(len(x))
    res[0]=1
    for i in range(1,len(x)):
        # print(i)
        # print(res[i])
        # print(res[:(i+1)])
        # print(x[:(i+1)])
        # print(y[i])
        res[i] = inverse_spread_function(res[:(i+1)], x[:(i+1)],y[i], r, Rec)
    return res

# Forward default rate survival curve :
def Forward_bootstrapping_method(curve_CDS, index_input, l, r, Rec, type="linear"):
    x_input, y_input, x_all = index_input.copy(), np.squeeze(
        curve_CDS.loc[index_input, :].values), curve_CDS.index.values
    x_all = np.append([0], x_all)
    yinterp = np.interp(x_all, x_input, y_input)
    Q_true = [np.exp(-l * x) for x in x_all]
    Q_bootstrapp = recursive_bootstrap(x_all, yinterp, r, Rec)
    plt.plot(x_all, Q_bootstrapp, '-x', label='bootstrapped Q', color='red')
    plt.plot(x_all, Q_true, 'o', label='true Q')
    plt.legend(loc='best')
    plt.show()
    print("MSE bootstrapping method: ", np.mean((Q_true - Q_bootstrapp) ** 2))

#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
#Forward_bootstrapping_method(curveCDS, index_all[index_input], l, r, Rec)
#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %







