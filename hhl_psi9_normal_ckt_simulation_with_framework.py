# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 01:19:32 2024

@author: mridu
"""
#%%
import numpy as np
from numpy import array
import math
import cmath
#import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
from decimal import Decimal
#%%
#switch 0 is for propagation of noise with visible noise at the undesired outcome
#switch 1 is for propagation of noise with modulus of the subtraction between ibm and simulation data
switch = 1
range_ = 10
#%%

# Quantum Gates for HHL
I = np.array([[1, 0], [0, 1]])
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
Sqrt_X = (1/2)*array([[1+1j,1-1j],[1-1j,1+1j]])
Not = np.array([[0, 1], [1, 0]])
U2 = np.array([[0, -1], [-1, 0]])
U1 = 0.5 * np.array([[-1 + 1j, 1 + 1j], [1 + 1j, -1 + 1j]])
Bra_0 = np.array([[1], [0]])
Bra_1 = np.array([[0], [1]])
TBra_0 = np.transpose(Bra_0)
TBra_1 = np.transpose(Bra_1)
outer_00 = np.dot(Bra_0, TBra_0)
outer_11 = np.dot(Bra_1, TBra_1)
Swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
Ru = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1j]])
theta1 = 2 * np.arcsin(1/1)
RY01 = np.array([[np.cos(theta1/2), -np.sin(theta1/2)], [np.sin(theta1/2), np.cos(theta1/2)]])
theta2 = 2 * np.arcsin(1/2)
RY10 = np.array([[np.cos(theta2/2), -np.sin(theta2/2)], [np.sin(theta2/2), np.cos(theta2/2)]])
Ru_inv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])
inv_U2 = np.array([[0, -1], [-1, 0]])
inv_U1 = 0.5 * np.array([[-1 - 1j, 1 - 1j], [1 - 1j, -1 - 1j]])

#%%
#functions for plots and multiplication
def nTensorProduct(*args):
  result = np.array([[1.0]])
  for i in args:
    result = np.kron(result, i)
  return result
def nDotProduct(*arg):
    res = np.eye(16)
    for i in arg:
     res = np.dot(res, i)
    return res
def matrix_mult(M1,M2):
    r = [[0 for x in range(len(M2[0]))] for y in range(len(M1))]
    #loops for matrix multiplication

    for row in range(len(M1)):                         #loop the no. of row for the first Matrix
        for column in range(len(M2[0])):               #loop the no.of column for the second Matrix   
            for common in range(len(M2)):              #common row and column for result r
                r[row][column] += M1[row][common] * M2[common][column]

    #for val in r:
        #print(val)
    r = np.array(r)
    return(r)
def simulation_plot(Psi_n,title,n):
    sqps = (np.absolute(Psi_n))**2;      #absolute square or probability of Psi stage data
    #print('sqps = \n', sqps)
    sqps_Tran = sqps.transpose()         #transpose of sqps
    sqps_list = sqps_Tran.tolist()       #convert Transposed data to list
    height =sqps_list[0]                 #convert the data with one square bracket
    #print('height= ',height)
    #x_axis = []     
    #nn = 0
    #for nn in range(2** n):
    #    x_axis.append(nn)
    #   nn = nn+1
    x_axis = ['0000','0001' ,'0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
    plt.bar(x_axis, height);
    plt.xticks(rotation= 55)
    for i, v in enumerate(height):
        if(v>0):
            plt.text(i, v, str(round(v, 3)), ha='center', va='bottom', fontsize = 8)
    #axis label
    plt.xlabel('Basis number')
    plt.ylabel('Probability')
    plt.title(title)
    plt.show();

#%%
# loop for dot multiplication,bit flip and error propagation simulations for each node of Psi
def test_node(Prev_Psi,Prev_IBMQ_data,psi_num,n,*args,**kwargs):
    #print("$$$$ Previous IBMQ data $$$$",Prev_IBMQ_data)
    test = Prev_Psi
    Psi_y_pre = Prev_Psi
    IBMQ_Data_pre = Prev_IBMQ_data
    node_num = 1
    Psi_limit = 0 #to achieve IBM_psi data or list for node 1 Psi_limit equals 1..
    av_prop_err_nodebynode = [];
    av_bitflip_err_nodebynode = [];
    noise = [];
    av_noise_list = [];
    Prev_noise =[];
    total_noise = [0] * 16; #placing zeros initially
    #gate_error =  y = [0.007] * 20;
    y = [0] * 16; #placing zeros initially
    noise = [0] * 16; #placing zeros initially
    z = [0] * 16; #placing zeros initially
    av_acc_Prop_err = [];
    total_noise_list = []
    total_prop_error = [];
    total_bitflip_error = [];
    max_bitflip_error = [];
    max_prop_error = [];
    max_noise = [];
    total_gate_error = [];
    min_gate_error = []
    #print(args)
    #print('000000000000000000000000000')
    #print(args[2])
    for  i in args:
        
        test = matrix_mult(i,test)#new matrix 'i' multiply previous dot product or test#current vector Psi
        psi_n = str(psi_num)
        node_n = str(node_num)
        title = 'Simulation plot for Psi '+psi_n+' node '+node_n
        #print(node_num)
        #print('\n')
        #print(i)
        #print("$$$$ Previous IBMQ data 1 $$$$",Prev_IBMQ_data)
        #print("length of matrix",len(args))
        #-----------------------------------------------------------------------------------------------#
        s = simulation_plot(test, title, n) #expected outcome for the stage/Psi i
        #Bit flip Error function starts here..
        single_bitflip = Single_Bit_Flip_Error(test, psi_num, n, node_num)
        #average propagation error calculation for each node
        total_bitflip_error.append(sum(single_bitflip))
        max_bitflip_error.append(max(single_bitflip))
        av_bitflip_error = sum(single_bitflip)/len(single_bitflip)  
        av_bitflip_err_nodebynode.append(av_bitflip_error)
        #print('====================================================')
        #print('Average bit flip error\n',av_bitflip_err_nodebynode )
        #print('====================================================')
        #-----------------------------------------------------------------------------------------------#
        #Propagation Error Functions start here..
        #collection of IBM data through function 'myCodeExp': Each call collect the IBM data for current Psi/stage number
        def myCodeExp(Psi_limit,**kwarg):
            x=str(Psi_limit)
            for key, value in kwargs.items():
                ibm = value  # value for IBMQx is collected
                if x in key: #match 'x' in IBMQx
                    break    #x is matched with IBMQx so no more iteration after IBMQx   
            #print('dash dash dash dash' )    
            #print('IBMQ data no. : ',Psi_limit)
            #print('Print inside the loop..',ibm)
            #Psi_limit = Psi_limit + 1 #after each iteration this increases by one
            return(ibm)
        
        ibmq_data = myCodeExp(Psi_limit,**kwargs)
        #print('ibmq_data \n',ibmq_data) 
        #print('previous IBMQ data: \n',IBMQ_Data_pre)
        #print('previous Psi: \n',Psi_y_pre)
        #print('new Psi: \n',test)
        #print('dash dash dash dash' ) 
        num = len(args)-1
        #accumulated propagation error function
        #z = acc_Propagation_error(prev_noise=noise, prev_prop_error=z, curr_gate_matrix=i)
        #z_avg = sum(z)/len(z)
        #av_acc_Prop_err.append(z_avg)
        #calling the propagation funcion for previous node
        
        y = Propagation_Psi(Psi_y_pre, IBMQ_Data_pre, ibmq_data,i, psi_num, node_num, n)
        #print("======================CHECK FOR PROP ERR========================")
        #print('Previous Simulated Psi:',Psi_y_pre)
        #print("Previous IBMQ",IBMQ_Data_pre)
        #print("Current IBMQ",ibmq_data)
        #print("Current Gate Matrix",i)
        #print("======================CHECK FOR PROP ERR========================")
        y_a = Propagation_Psi_withoutNoise(y, test, psi_num, node_num)
        #average propagation error calculation for each node
        av_prop_error = sum(y)/len(y)  
        total_prop_error.append(sum(y))
        av_prop_err_nodebynode.append(av_prop_error)
        max_prop_error.append(max(y))
        
        #print('====================================================')
        #print('average Propagation error: \n',av_prop_err_nodebynode)
        #print('====================================================')#calling the propagation funcion 
        
        
        
        #-----------------------------------------------------------------------------------------------#
        #extracting noise only
        sqps = np.absolute(test)**2     #convertion of simulation data from amplitude to probability
        
        noise = noise_extract(sqps, ibmq_data, n=4)
        print('noise: ',noise)
        # Convert each element to Decimal during calculation
        sum_of_noise = sum(noise)
        total_noise_list.append(sum_of_noise)
        av_noise = sum_of_noise / len(noise)
        max_noise.append(max(noise))
        #print('Average noise:   ',av_noise)
        #av_noise = sum(noise)/len(noise)
        av_noise_list.append(av_noise)
        #-----------------------------------------------------------------------------------------------#
        
        #-----------------------------------------------------------------------------------------------#
        
        #update few parameters for the next iteration..
        Psi_y_pre = test
        IBMQ_Data_pre = ibmq_data #save this ibmq_data as previous IBMQ data for the next iteration
        node_num = node_num + 1 #node number is updated here..
        Psi_limit = Psi_limit + 1 #after each iteration this increases by one
        Prev_Prop_Err_1node = y;
        
    #print(av_bitflip_err_nodebynode) 
    #print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
    #print('accumulated propagation error: ',av_acc_Prop_err)
    #print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
    print('Average noise: ',av_noise_list)
    print('Total noise: \n',total_noise_list)
    #print(av_prop_err_nodebynode)
    #x_axis = [n for n in range(1, range_)]
    x_axis = [1,2,3,4,5,7,8,9]
    # Perform element-wise calculation for Total_Gate_Error
    total_gate_error = [abs(noise - propagation - bitflip) for noise, propagation, bitflip in zip(total_noise_list, total_prop_error, total_bitflip_error)]
    min_gate_error = [abs(noise - propagation - bitflip) for noise, propagation, bitflip in zip(max_noise, max_prop_error, max_bitflip_error)]

    
    # ------------------------------------------------------------------------------------#
    
    # ------------------------------------------------------------------------------------#
#%%    
    # Plot the graph
    plt.figure(figsize=(12, 8))
    plt.plot(x_axis,total_noise_list,marker = 'o', label='Total noise at each node',color= 'red')
    plt.plot(x_axis,total_prop_error,marker = 's',label = 'Total Propagation Error',color = 'purple')
    plt.plot(x_axis,total_bitflip_error,marker = 'p', label = 'Total Bitflip Error', color = 'blue')
    plt.plot(x_axis,total_gate_error, marker = 'h', label = 'Total Gate Error',color = 'magenta')
    # Add labels and legend
    plt.xlabel('Node Number')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')
    plt.ylim(10**(-6), 1.1)
    # Add grid and set x-axis ticks as integers
    plt.grid(True)
    plt.xticks(range(1, range_))
    
    # Show the plot
    plt.show()
    
    #print('Total_Noise: \n',total_noise_list)
    print('Total_Propagation_Error: \n',total_prop_error)
    #print('Total_Bitflip_Error',total_bitflip_error)
    #print('Total gate error: \n', total_gate_error)

#%%    
    # Plot the graph
    plt.figure(figsize=(12, 8))
    plt.plot(x_axis,max_noise,marker = 'd', label='Maximum noise at each node',color= 'green')
    plt.plot(x_axis,max_prop_error ,marker = 's',label = 'Maximum Propagation Error',color = 'indigo')
    plt.plot(x_axis,max_bitflip_error ,marker = 'p', label = 'Maximum Bitflip Error', color = 'orange')
    plt.plot(x_axis,min_gate_error, marker = 'h', label = 'Minimum Gate Error',color = 'magenta')
    
    # Add labels and legend
    plt.xlabel('Node Number')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')
    plt.ylim(10**(-6), 1)
    # Add grid and set x-axis ticks as integers
    plt.grid(True)
    plt.xticks(range(1, range_))
    
    # Show the plot
    plt.show()
    #print('Maximum_Noise: \n',max_noise)
    #print('Maximum_Propagation_Error: \n',max_prop_error)
    #print('Maximum_Bitflip_Error',max_bitflip_error)
            
    return(test)
#%%
#functions common for Single Bit Flip Error 
def plot_processed_data(h,n,psi_num,count):
    #print('bit flip without desired output= \n',h)
    # create list x_axis
    #x_axis = []     
    #nn = 0
    #for nn in range(2**n):
    #    x_axis.append(nn)
    #    nn = nn+1
    x_axis = ['0000','0001' ,'0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
    plt.bar(x_axis, h);
    plt.xticks(rotation= 55)
     #axis label
    plt.xlabel('Basis number')
    plt.ylabel('Probability')
    for i, v in enumerate(h):
        if(v>0):
            plt.text(i, v, str(round(v, 3)), ha='center', va='bottom', fontsize = 8)
    psi_num = str(psi_num)
    count = str(count)
    title = 'Single Bit Flip for Psi '+psi_num+' node'+count+' without desired output'
    plt.title(title)
    plt.show();
#%%
#functions common for Single Bit Flip Error 
def plot_processed_data1(h,n,psi_num,count,title_):
    #print('bit flip without desired output= \n',h)
    # create list x_axis
    #x_axis = []     
    #nn = 0
    #for nn in range(2**n):
    #    x_axis.append(nn)
    #    nn = nn+1
    x_axis = ['0000','0001' ,'0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
    
    plt.bar(x_axis, h);
    plt.xticks(rotation= 55)
     #axis label
    plt.xlabel('Basis number')
    plt.ylabel('Probability')
    psi_num = str(psi_num)
    # Add y-axis values on top of each bar
    # Add y-axis values on top of each bar
    for i, v in enumerate(h):
        if(v>0):
            plt.text(i, v, str(round(v, 3)), ha='center', va='bottom', fontsize = 8)
        

    count = str(count)
    title = title_
    #title = 'Propagation Error for Psi '+psi_num+' node'+count+title_withorwithout
    plt.title(title)
    plt.show();
#%%
def to_eliminate_desired_output(sqps,error_data,n):   
    #print('at first, sqps at input= \n',sqps)#sqps is a vector which is converted to list
    #print('error_data at input= \n',error_data)#error_data is a list taken from
    trans = [[sqps[j][i] for j in range(len(sqps))] for i in range(len(sqps[0]))] 
    #print("\n")
    #print('trans= \n',trans)
    sqps_h = trans[0]
    sqps_h = np.round(sqps_h,3)#rounding to 3 decimal place
    
    #print('list form of sqps = \n',sqps_h)
    #print('error data: \n', error_data)
    '''data_1 and btflp_error list is created'''
    data_1 = []
    error = []
    for i in range(2**n):
     data_1.append(0)
     error.append(0)
    #print('data_1 = \n',data_1)
    for j in range(2**n):
        if (sqps_h[j] == 0):
            data_1[j] = 1
            #print('j =',j)
            #print('sqps = \n',sqps_h[j])
        else:
            data_1[j] = 0
    #print('Data_1 modified = \n',data_1) 
    for k in range(2**n):
        error[k] = data_1[k] * error_data[k]
    #print(' error data is: \n',error)
    return(error)
#%%
def noise_extract(sqps,error_data,n):   
    #print('at first, sqps at input= \n',sqps)#sqps is a vector which is converted to list
    #print('error_data at input= \n',error_data)#error_data is a list taken from
    trans = [[sqps[j][i] for j in range(len(sqps))] for i in range(len(sqps[0]))] 
    #print("\n")
    #print('trans= \n',trans)
    sqps_h = trans[0]
    sqps_h = np.round(sqps_h,4)#rounding to 4 decimal place
    
    #print('list form of sqps = \n',sqps_h)
    '''data_1 and btflp_error list is created'''
    #data_1 = []
    error = []
    error_abs = []
    #print('data_1 = \n',data_1)
    #print('Data_1 modified = \n',data_1) 
    error = [sqps_val - error_data_val for sqps_val, error_data_val in zip(sqps_h, error_data)]
    print(error)
    error_abs = np.abs(error)
    
    return(error_abs)
#%%
# Single Bit flip code
def Single_Bit_Flip_Error(Psi,psi_num,n,count): #count is the node number updated from the loop of actual code
                                                #n= bit number, Psi= Psi 2-9,psi_num=numerical value for Psi
    def single_bitflip_data(Psi,n):
        n>0
        b= bitflip_matrix(n)                    #generate single bitflip matrix for n bit
        #print('Bitflip Matrix: \n',b)
        sqps = (np.absolute(Psi))**2            # Probability = absolute square of Psi data
        #print('sqps= \n',sqps)
        b1 = matrix_mult(b, sqps)               # single bit flip data = b * sqps
        
        P1 = (0.053)/4
        b2 = np.dot(P1,b1)
        return(b2)
    def bitflip_matrix(n):
        n>0
        dimension = ((2)**n)
        matrx = [[0 for x in range(dimension)]for y in range(dimension)]
        #loop
        for c in range(dimension):
            for r in range(dimension):
                col = convert_to_binary(c,n)
                row  = convert_to_binary(r,n)
                count = compare_binary_numbers(row,col,n)
                if count == 1:
                    matrx[c][r] = 1
                else:
                    matrx[c][r] = 0
        return(matrx)
    def convert_to_binary(a,n):
        binary_nbit = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(n)] ) )
        return(binary_nbit(a))
    def compare_binary_numbers(r,c,n):
        r_list = convert_to_list(r)
        c_list = convert_to_list(c)
        count = 0
        for i in range(n):
            if (r_list[i] != c_list[i]):
                count = count + 1
            else:
                count = count
        return(count)
    def convert_to_list(a):                     #converts string or integer to list
        list_ = [int(x) for x in str(a)]
        return(list_)
    def matrix_mult(M1,M2):
        r = [[0 for x in range(len(M2[0]))] for y in range(len(M1))]
        #loops for matrix multiplication

        for row in range(len(M1)):                         #loop the no. of row for the first Matrix
            for column in range(len(M2[0])):               #loop the no.of column for the second Matrix   
                for common in range(len(M2)):              #common row and column for result r
                    r[row][column] += M1[row][common] * M2[common][column]

        #for val in r:
            #print(val)
        return(r)
    def single_bitflip_plot(btflp,n,psi_num,count):
        trans = [[btflp[j][i] for j in range(len(btflp))] for i in range(len(btflp[0]))]
        #print("\n")
        #print('trans= \n',trans)
        h = trans[0]
        #P1 = (1-0.838)/4 #P1 is the error of Psi 1 from IBM Quito Hardware
        #for i in range((2)**n):
        #   h[i] = h[i] * P1
        #print('h = \n',h)
        #x_axis = []     
        #nn = 0
        #for nn in range(2** n):
        #    x_axis.append(nn)
        #    nn = nn+1
        x_axis = ['0000','0001' ,'0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
        plt.xticks(rotation= 55)
        plt.bar(x_axis, h);
         #axis label
        plt.xlabel('Basis number')
        plt.ylabel('Single bit flip Prob.')
        psi_num = str(psi_num)
        count = str(count)
        title = 'Single Bit Flip for Psi'+psi_num+' node'+count+'-overall'
        plt.title(title)
        plt.show();
        return(h) # returns data to be modified for other graphs
    
    
    # execution of the main function
    x = single_bitflip_data(Psi, n)
    plot_for_x = single_bitflip_plot(x, n, psi_num, count)
    sqps_x = np.absolute(Psi)**2
    x_a = to_eliminate_desired_output(sqps_x,plot_for_x, n)
    plot_for_x_a = plot_processed_data(x_a, n,psi_num,count)
    return(plot_for_x)
#%%
#single bit flip code- 2 functions
def Propagation_Psi(Prev_Psi, Prev_IBMQ_data ,IBMQ_data,Matrix,psi_num,node_num,n):
    Matrix_sqr= np.absolute(Matrix)**2
    sqps_prev = np.square(np.absolute(Prev_Psi))
    if(switch == 0):
        noise_Psi1 = to_eliminate_desired_output(sqps_prev,Prev_IBMQ_data,n) #data we obtain here is a list
                                                                          #we need to convert it to array enclosed with double
                                                                          # square brackets in order to execute matrix_mult or
   
                                                                     #matrix multiplication
    elif(switch == 1 ):
        noise_Psi1 = noise_extract(sqps_prev,Prev_IBMQ_data,n)                                                          #matrix multiplication

    noise_Psi_n11 = [noise_Psi1]                                # square bracket added for the process of converting list to double
                                                                      # double square bracket array

    noise_Psi21 = array(noise_Psi_n11)                                #convert to array 
    
    error_prop_n = matrix_mult(noise_Psi21, Matrix_sqr)             #data we obtain here is an array/list. We need to convert
                                                            #it to list[0] format in order to plot graph
    #print('Propagation Matrix',Matrix_sqr)
    #print('Propagation vector',noise_Psi21)
    #print('Propagated vector',)
    
    e_pn = error_prop_n[0]                                            # e_pn is a list of error data for Psi_n
    psi_numb = str(psi_num)
    node_numb = str(node_num)
    plot_title = 'Propagation Error for Psi '+psi_numb+' node '+node_numb+' overall'
    Error_Prop_plot = plot_processed_data1(e_pn,n, psi_num, node_num,plot_title) ###############  PROBLEM  ################################
    return(e_pn)

#%%
def Propagation_Psi_all_error(Prev_Psi, Prev_IBMQ_data ,IBMQ_data,Matrix,psi_num,node_num,n):
    Matrix_sqr= np.absolute(Matrix)**2
    sqps_prev = np.square(np.absolute(Prev_Psi))
    
    noise_Psi1 = to_eliminate_desired_output(sqps_prev,Prev_IBMQ_data,n) #data we obtain here is a list
                                                                      #we need to convert it to array enclosed with double
                                                                      # square brackets in order to execute matrix_mult or
                                                                      #matrix multiplication

    noise_Psi_n11 = [noise_Psi1]                                # square bracket added for the process of converting list to double
                                                                      # double square bracket array

    noise_Psi21 = array(noise_Psi_n11)                                #convert to array 
    
    error_prop_n = matrix_mult(noise_Psi21, Matrix_sqr)             #data we obtain here is an array/list. We need to convert
                                                            #it to list[0] format in order to plot graph
    #print('Propagation Matrix',Matrix_sqr)
    #print('Propagation vector',noise_Psi21)
    #print('Propagated vector',)
    #print('propagation sqps: ',sqps)
    e_pn = error_prop_n[0]                                            # e_pn is a list of error data for Psi_n
    psi_numb = str(psi_num)
    node_numb = str(node_num)
    plot_title = 'Propagation Error for Psi '+psi_numb+' node '+node_numb+' overall'
    Error_Prop_plot = plot_processed_data1(e_pn,n, psi_num, node_num,plot_title) ###############  PROBLEM  ################################
    return(e_pn)
    
#this should have been named propagation error without desired output, named it wrong..    
def Propagation_Psi_withoutNoise(e_pn,Psi,psi_num,node_num):
    sqps = np.absolute(Psi)**2
    
    prop_error_only = to_eliminate_desired_output(sqps, e_pn,4)  #data we obtain is an arra/list ready for plot
       
    psi_numb = str(psi_num)
    node_numb = str(node_num+1)
    plot_title1 = 'Propagation Error for Psi '+ psi_numb +' node '+ node_numb +' without desired output' 
    Psi_prop_err_only_plot = plot_processed_data1(prop_error_only,4,psi_num, node_num,plot_title1) #plot graph from list
    epn = prop_error_only
    #print('-----------------------------------------------------------')
    #print('Propagation Error, Psi:',psi_num,'node number: ,', node_num)
    #print(epn)
    #print('-----------------------------------------------------------')
    
        
    return(epn)

        

#%%
import pandas as pd

# data read till Psi 3 from CSV file and save them as list
file_path = 'data.csv'
row1 = [];row2 = [];row3 = [];row4 = [];row5 = [];
row6 = [];row7 = [];row8 = [];row9 = [];

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Iterate through the rows and create lists with names 'row1' to 'row8'
for i, row in df.iterrows():
    row_list_name = 'row' + str(i + 1)
    globals()[row_list_name] = row.tolist()
#%%    
#%%
# Each of the nodes or staes of the HHL quantum circuit is defined by the tensor products, from T1 to T8.
# HHL circuit can be found in the data collection Qiskit script.
#Psi 1
I = np.array([[1, 0], [0, 1]])
T1_n0 = nTensorProduct(Not,I,I,I);
T1 = T1_n0

#Psi 2

H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
T2_n0 = nTensorProduct(I,H,H,I);
T2 = T2_n0

#Psi 3
T3_n0 = nTensorProduct(I,I,outer_00,I) + nTensorProduct(U1,I,outer_11,I)
#print(T3_n0.shape)
T3_n1 = nTensorProduct(I,outer_00,I,I) + nTensorProduct(U2,outer_11,I,I)
T3 = nDotProduct(T3_n0,T3_n1)

#Psi 4
T4_n0 = nTensorProduct(I,H,I,I)
T4_n1 = nTensorProduct(I,Ru,I)
T4_n2 = nTensorProduct(I,I,H,I)
T4_n3 = nTensorProduct(I,Swap,I)

T4 = nDotProduct(T4_n3,T4_n2,T4_n1,T4_n0)

#Psi_5
T5_n0 = nTensorProduct(I,I,outer_00,I) + nTensorProduct(I,I,outer_11,RY01)
T5_n1 = nTensorProduct(I,outer_00,I,I) + nTensorProduct(I,outer_11,I,RY10)

T5 = nDotProduct(T5_n1,T5_n0)

#Psi 7
Swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]);
Ru_inv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])
T7_n0 = nTensorProduct(I,Swap,I)
T7_n1 = nTensorProduct(I,I,H,I)
T7_n2 = nTensorProduct(I,Ru_inv,I)
T7_n3 = nTensorProduct(I,H,I,I)

T6 = nDotProduct(T7_n3,T7_n2,T7_n1,T7_n0)
#T6 = nDotProduct(T7_n0,T7_n1,T7_n2,T7_n3)
print('T6',T6)
#Psi 8
T8_n0 = nTensorProduct(I,outer_00,I,I) + nTensorProduct(inv_U2,outer_11,I,I)
T8_n1 = nTensorProduct(I,I,outer_00,I) + nTensorProduct(inv_U1,I,outer_11,I)

T7 = nDotProduct(T8_n1, T8_n0)
#T7 = nDotProduct(T8_n0, T8_n1)

#Psi 9
T9_n0 = nTensorProduct(I,H,H,I)
T8 = T9_n0

#Psi List

HHL_list = [T1,T2,T3,T4,T5,T6,T7,T8]

#HHL Psi9 IBM Brisbane Data Test
#print('row1 \n',row1)
print('row8 \n',row8)

#IBMQ Data in Kwarg list
HHL_IBM_Psi9 = {'node0': row1,
               'node1' : row2,
               'node2' : row3,
               'node3' : row4,
               'node4' : row5,
               'node5' : row6,
               'node6' : row7,
               'node7' : row8
               }
#framework call
# loop for different simulations at each node
prv_ibmq_data = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  #initial ibmq data
Psi_0 = array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])#initial Psi
HHL_brisbane = test_node(Psi_0, prv_ibmq_data,'',4,*HHL_list,**HHL_IBM_Psi9)
#%%

                                     ####END######


#%%









