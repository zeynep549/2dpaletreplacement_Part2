import pyomo
import time
from pyomo.environ import *
import pandas as pd
import os
os.environ['GRB_LICENSE_FILE'] = '/Users/ecederya/Desktop/gurobi.lic'
#necesarry for my own computer

df = pd.read_excel('END395_ProjectPartIDataset.xlsx')
df2 = pd.read_excel('END395_ProjectPartIIDataset.xlsx',sheet_name="Pallets6")
dfPart2 = pd.read_excel('/Users/ecederya/Desktop/end395/END395_ProjectPartIIDataset.xlsx')

cumulative_values1 = dfPart2[1].tolist()
cumulative_values2 = dfPart2[2].tolist()
cumulative_values3 = dfPart2[3].tolist()
cumulative_values4 = dfPart2[4].tolist()
coefficient = df['Coefficient'].tolist()
startLocks = df['Lock1(m)'].tolist()
endLocks = df['Lock2(m)'].tolist()
harm_values = df['H-arm'].tolist()
weight_values = df2['Weight'].tolist()
max_weights = df['Max Weight'].tolist()
codes = df2['Code'].tolist()
positionNames = df['Position'].tolist()

M = ConcreteModel()
start_time = time.time()


M.intervals = ([2,1,3,4])
M.I = RangeSet(2,34)
M.J = RangeSet(2,105)
M.K = RangeSet(1,3)
M.A = RangeSet(2,39)
M.B = RangeSet(40,83)
M.C = RangeSet(84,105)
M.Q = [2,22,84,95,40,51,62,73,3,23,85,96,41,52,63,74,4,24,86,97,
                      5,42,53,64,75,25,87,98,6,26,88,99,43,54,65,76,7,27,89,100,
                      44,55,66,77,8,28]
M.S = [21,39,20,38,19,37,18,36,105,94,17,35,104,93,83,72,61,50,16,103,92,34,15,82,
                      71,60,49,102,91,33,14,81,70,59,48,101,90,13,32,80,69,58,47,12,31,
                      11,30,79,68,57,46]

#Parameters
M.w = Param(M.I,mutable=True)
M.cm1 = Param(M.J,mutable=True)
M.cm2 = Param(M.J,mutable=True)
M.cm3 = Param(M.J,mutable=True)
M.cm4 = Param(M.J,mutable=True)
M.coef = Param(M.J,mutable=True)
M.start1 = Param(M.A,mutable=True)
M.end1 = Param(M.A,mutable=True)
M.start2 = Param(M.B,mutable=True)
M.end2 = Param(M.B,mutable=True)
M.start3 = Param(M.C,mutable=True)
M.end3 = Param(M.C,mutable=True)
M.harm = Param(M.J,mutable=True)
M.maxweight = Param(M.J,mutable=True)
M.fuelConsumption = Param([1,2,3,4],mutable=True)

#Decision Variables 
M.x = Var(M.I,M.J,M.K,within=Binary)
M.intervalSelection = Var(M.intervals,within=Binary)


DOW = 110257
DOI = 61.6

for i, value in enumerate(weight_values):
    M.w[i+2].value = value 
    
for i, value in enumerate(harm_values):
    M.harm[i+2].value = value 
    

for i, value in enumerate(cumulative_values1):
    #print(value)
    M.cm1[i+2].value = value 

for i, value in enumerate(cumulative_values2):
    #print(value)
    M.cm2[i+2].value = value 

for i, value in enumerate(cumulative_values3):
    #print(value)
    M.cm3[i+2].value = value 

for i, value in enumerate(cumulative_values4):
    #print(value)
    M.cm4[i+2].value = value 
    
for i, value in enumerate(coefficient):
    #print(value)
    M.coef[i+2].value = value 
    
    
for i, value in enumerate(startLocks):
    if i > 37:
        break
    M.start1[i+2].value = value 
    
for i, value in enumerate(endLocks):
    if i > 37:
        break
    M.end1[i+2].value = value 
    
for i, value in enumerate(startLocks):
    if i >= 38 and i <= 81:
        M.start2[i+2].value = value 
    
for i, value in enumerate(endLocks):
    if i >= 38 and i <= 81:
        M.end2[i+2].value = value 
        
for i, value in enumerate(startLocks):
    if i >= 82 and i <= 103:
        M.start3[i+2].value = value 
    
for i, value in enumerate(endLocks):
    if i >= 82 and i <= 103:
        M.end3[i+2].value = value 


M.fuelConsumption[2] = 1
M.fuelConsumption[1] = 2
M.fuelConsumption[3] = 3
M.fuelConsumption[4] = 4   

#Objective Function
M.obj = Objective(expr = sum(M.intervalSelection[i]*M.fuelConsumption[i] for i in M.intervals), sense = minimize)


#Constraints
M.constraints = ConstraintList()

#ensuring k values are correct(loading type)
for j in M.J:
    if 2 <= j <= 39:
        M.constraints.add(sum(M.x[i,j,2]for i in M.I) == 0)
        M.constraints.add(sum(M.x[i,j,3]for i in M.I) == 0)
    elif 40 <= j <= 83:
        M.constraints.add(sum(M.x[i,j,1]for i in M.I) == 0)
        M.constraints.add(sum(M.x[i,j,3]for i in M.I) == 0)
    else:
        M.constraints.add(sum(M.x[i,j,1]for i in M.I) == 0)
        M.constraints.add(sum(M.x[i,j,2]for i in M.I) == 0)

    
        
for j, value in enumerate(max_weights, start=2):
    M.constraints.add(sum(sum(M.x[i, j, k] * M.w[i] for k in M.K) for i in M.I) <= value)

#blue envelope constraints
M.constraints.add((sum(sum(sum(M.x[i,j,k]*M.w[i] for k in M.K)for j in M.J)for i in M.I) + DOW)
 >= 120000)
M.constraints.add((sum(sum(sum(M.x[i,j,k]*M.w[i] for k in M.K)for j in M.J)for i in M.I) + DOW)
 <= 180000)

M.constraints.add((-1*(sum((M.harm[j]-363495)*(sum(sum(M.x[i,j,k]*M.w[i] for k in M.K)for i in M.I))/2500 for j in M.J)+DOI))+235-(sum(sum(sum(M.x[i,j,k]*M.w[i] for k in M.K)for i in M.I)for j in M.J)*1000 + DOW)
<=0)
M.constraints.add(2*(sum((M.harm[j]-363495)*(sum(sum(M.x[i,j,k]*M.w[i] for k in M.K)for i in M.I))/2500 for j in M.J)+DOI) -240 - (sum(sum(sum(M.x[i,j,k]*M.w[i] for k in M.K)for i in M.I)for j in M.J)*1000 + DOW)
 <= 0)

#pallet compatibility constraints
for j in M.J:
    for i in M.I:
        if (2 <= j <= 21 and 2<=i<=32):
            M.constraints.add(sum(M.x[i,j,k] for k in M.K) == 0)
        if (62 <= j <= 83 and 2<=i<=32):
            M.constraints.add(sum(M.x[i,j,k] for k in M.K) == 0)
        if (84 <= j <= 94 and 2<=i<=32):
            M.constraints.add(sum(M.x[i,j,k] for k in M.K) == 0)
        if (22 <= j <= 39 and 33<=i<=34):
            M.constraints.add(sum(M.x[i,j,k] for k in M.K) == 0)
        if (40 <= j <= 61 and 33<=i<=34):
            M.constraints.add(sum(M.x[i,j,k] for k in M.K) == 0)
        if (95 <= j <= 105 and 33<=i<=34):
            M.constraints.add(sum(M.x[i,j,k] for k in M.K) == 0)
        
            
M.constraints.add(sum(M.intervalSelection[i] for i in M.intervals)==1)

for i in M.I:
    M.constraints.add(sum(sum(M.x[i,j,k]for k in M.K)for j in M.J) == 1)

for j in M.J:
    M.constraints.add(sum(sum(M.x[i,j,k]for k in M.K)for i in M.I) <= 1)
    

#cumulative weight constraints based on interval selection 
for j, a in zip(M.Q, range(2, 48)):
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[1] for i in M.I for k in M.K for q in M.Q if M.Q.index(q) <= M.Q.index(j)) <= M.cm1[a])
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[2] for i in M.I for k in M.K for q in M.Q if M.Q.index(q) <= M.Q.index(j)) <= M.cm2[a])
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[3] for i in M.I for k in M.K for q in M.Q if M.Q.index(q) <= M.Q.index(j)) <= M.cm3[a])
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[4] for i in M.I for k in M.K for q in M.Q if M.Q.index(q) <= M.Q.index(j)) <= M.cm4[a])


for j, a in zip(M.S, range(105,54,-1)):
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[1] for i in M.I for k in M.K for q in M.S if M.S.index(q) <= M.S.index(j)) <= M.cm1[a])
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[2] for i in M.I for k in M.K for q in M.S if M.S.index(q) <= M.S.index(j)) <= M.cm2[a])
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[3] for i in M.I for k in M.K for q in M.S if M.S.index(q) <= M.S.index(j)) <= M.cm3[a])
    M.constraints.add(sum(M.x[i,q,k]*M.w[i]*M.coef[q]*M.intervalSelection[4] for i in M.I for k in M.K for q in M.S if M.S.index(q) <= M.S.index(j)) <= M.cm4[a])



#overlapping constraints among single row and side-by-side
for j1 in M.A:
    for j2 in M.B:
        if(M.start1[j1].value >= M.start2[j2].value and M.start1[j1].value <= M.end2[j2].value):
            M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                              sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)
        if(M.end1[j1].value >= M.start2[j2].value and M.end1[j1].value <= M.end2[j2].value):
            M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                              sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)

#overlapping constraints for "single row" loading type
for j1 in M.A:
    for j2 in M.A:
        if j1 != j2:
            if(M.start1[j1].value >= M.start1[j2].value and M.start1[j1].value <= M.end1[j2].value):
                M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                                  sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)
            if(M.end1[j1].value >= M.start1[j2].value and M.end1[j1].value <= M.end1[j2].value):
                M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                                  sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)

#overlapping constraints for the lower deck of the plane
for j1 in M.C:
    for j2 in M.C:
        if j1 != j2:
            if(M.start3[j1].value >= M.start3[j2].value and M.start3[j1].value <= M.end3[j2].value):
                M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                                  sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)
            if(M.end3[j1].value >= M.start3[j2].value and M.end3[j1].value <= M.end3[j2].value):
                M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                                  sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)

#overlapping constraints for "side-by-side" loading type, allowing L and R placing 
for j1 in M.B:
    for j2 in M.B:
        if j1 != j2 and ((40<=j1<=50 and 62<=j2<=72) or (51<=j1<=61 and 73<=j2<=83)):
            if(M.start2[j1].value >= M.start2[j2].value and M.start2[j1].value <= M.end2[j2].value):
                M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                                  sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)
            if(M.end2[j1].value >= M.start2[j2].value and M.end2[j1].value <= M.end2[j2].value):
                M.constraints.add(sum(sum(M.x[i,j1,k] for k in M.K)for i in M.I) + 
                                  sum(sum(M.x[i,j2,k] for k in M.K)for i in M.I) <= 1)

front_positions = range(2,83)  
aft_positions = range(84,105) 

# Add constraints for each pallet based on RestrictedLoadingType, RestrictedPosition, and RestrictedLocation
for index, row in df2.iterrows():
    pallet_index = codes.index(row['Code']) + 2  # Adjust the index according to your codes list

    # Apply loading type restrictions
    if row['RestrictedLoadingType'] == 'SBS':
        M.constraints.add(sum(M.x[pallet_index, j, 2] for j in M.J) == 1)
    elif row['RestrictedLoadingType'] == 'SR':
        M.constraints.add(sum(M.x[pallet_index, j, 1] for j in M.J) == 1)

    # Apply position restrictions
    if row['RestrictedPosition'] != 0:  
       position_index = positionNames.index(row['RestrictedPosition'])
       M.constraints.add(sum(M.x[pallet_index, position_index+2, k] for k in M.K) == 1)

    # Apply location restrictions
    if row['RestrictedLocation'] == 'FD':
        M.constraints.add(sum(M.x[pallet_index, j, k] for j in front_positions for k in M.K) == 1)
    elif row['RestrictedLocation'] == 'LD':
        M.constraints.add(sum(M.x[pallet_index, j, k] for j in aft_positions for k in M.K) == 1)
    
    if row['NeighboringPalletKey'] != 0:
        neighbour_index = row['NeighboringPalletKey']+1
        for j1 in M.J:
            for j2 in M.J:
                if(abs(j1-j2)!=1 and j1 in M.J and j2 in M.J):
                    M.constraints.add( sum(M.x[neighbour_index, j2, k] for k in M.K) +
                                     sum(M.x[pallet_index, j1, k] for k in M.K) <= 1)
                  
solver = SolverFactory('gurobi')

solution = solver.solve(M)
cpu_time = time.time() - start_time


for i, value in enumerate(codes):
    for j,value2 in enumerate(positionNames):
        for k in M.K:
            if (M.x[i+2,j+2,k].value == 1):
                print("Pallet "  + str(value) + " is assigned to position " + str(value2) )

#Print the CPU time
print("CPU Time:", cpu_time, "seconds")
solution.write()
print()
for i in [2,1,3,4]:
    if (M.intervalSelection[i].value == 1):
        print("CG interval " + str(i) + " is selected.")