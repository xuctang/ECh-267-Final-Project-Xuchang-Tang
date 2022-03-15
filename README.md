# ECh-267-Final-Project-Xuchang-Tang
 Github repository including the Python code for ECh 267 final project from Xuchang Tang

 The two files belong to the two systems considered in this project. 
 
 To run the .py file, simply scroll down to the if __name__ == '__main__': section. You can adjust the parameters such as the cost matrices, ratio_arr for the array of Q ratio values for consideration, number of simulation, and the number of trials for each Q ratio by adjust the "trial" variable.
 
 To run the .ipynb file, simply run each blocks in sequence until you pass the main function and reached the parameter sections. Similarly, you can adjust the parameters and run for results.
 
 For both files, you can change the steady state input and thus the steady state value in the parameter section. The initial states are fixed for each problem. To change the initial states, you would have to visit the main function (longest one) and change the first row state_history accordingly (not recommended to change).
 
 All generated figures are stored in path "./Figures" with clearly marked titles. 
 
 Please note that these two files might take a long time to run. For reference, it took me 45 - 60 min to run a ratio_arr length of 8 with 10 trials each. Please leave time for the code running.
 
 The code is the preliminary result due to limited time. It has not been optimized yet.
