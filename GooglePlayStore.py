import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('E:/Data Science/Data Science Course tihe/Data Sets/googleplaystore3-1.csv',na_values='#')
#'''
df.describe()  # give information about numeric columns
df.describe(include="all")#Similar to summary in R
v=df.describe(include='all')
v  # see variable explorer to show all data 
df.dtypes# types of columns of data set similar to str function in R
df.head(3)#take the head of the dataframe (first three lines)
df.tail(12)#Gives the last 3 rows
df.shape#this gives you the dimension of the dataframe(like dim in R)-tupels
df.columns#This gives you the column names
df.size #shows the size of dataframe

df.isna()#Gives a matrix of true and false:true means the point is not availabe
np.sum(df.isna()) #show number of NaN per columns
df['Reviews']#This gives you the Reviews column
df.iloc[:,4]#Another way to get the Reviews column if you know the column number
df.loc[:,'Reviews']#Another way to get the column, result is serie
df1=df.loc[:,['Rating','Reviews']]#Make a new dataframe with Rating and Reviews variables 
df1=df.iloc[:,[3,4]]#Same as above
df['Reviews'].isna()#Find the missing values in Reviews (a vector of True and False values)
sum(df['Reviews'].isna())#the # of missing values
sum(df['Reviews'].notna())
np.sum(df['Reviews'].notna())
df['Type'].unique()#find the levels of variable Type  array(['Free', nan, 'Paid'], dtype=object)
df['Category'].unique()
len(df['Category'].unique())  # 33

# index means rows
df.index  #shows the name/number of each row
df.index[df['Reviews'].isna()]#Find the number of rows with missing values
df.index[df['Type']=='Free']

df['Reviews'].iloc[95] #nan shows the row index number 95 at 'Reviews'
df.iloc[95,:]  #show data in row 95 for all columns
df['Reviews'][95]  # nan

df.index[df['Rating'].isna()][0]    # 23  #result is a list 
df.index[df['Rating'].isna()][9]    # 185
np.where(df['Rating'].isna())       # same as df.index
np.where(df['Reviews'].isna())[0]   #an array includes all rows of NaN
len(df.index[df['Rating'].isna()])  #1473


df.index[df['Rating']>4.9]#Find the rows with Ratings larger than 4.9
len(df.index[df['Rating']>4.9])    # 274
sum(df['Rating']>4.9) #Find the number of these Rates   274

Reviews=df.loc[df['Rating']>4.9,'Reviews']#Find the Reviews for apps has Rating over 4.9
Review2=df['Reviews'].loc[df['Rating']>4.9] #same as above command
Reviews_lowrate=df.loc[df['Rating']<3.5,'Reviews']#Find the Reveiews for Ratings less than 3.5

np.mean(Reviews)             # 8.68
np.mean(Reviews_lowrate)     # 4803
np.std(Reviews)              # 16.140  enherafe me'yar Standard deviation
np.std(Reviews_lowrate)      # 26513

#Find the number of available observations
sum(df.loc[df['Rating']>4.9 ,'Type']=="Free")
hotreviews = df.loc[df['Reviews']>10000, 'Reviews'] # rows with high reviews
np.mean( df.loc[df['Reviews']>10000,'Reviews']) #  1135947.370759675
np.mean(hotreviews)  # 1135947.370759675

v=df.loc[df['Rating']>4.9,'Type']
v.value_counts()   
# Free    236
# Paid     27
# name: Type, dtype: int64

list(df.loc[df['Rating']>4.9,'Type']).count('Free')    # 236
list(df.loc[df['Rating']>4.9,'Type']).count('Paid')     # 27
list(df.loc[df['Rating']>4.9,'Type']).count(np.nan)  # 11 couldnot count missing values

sum(df.loc[df['Rating']>4.9,'Type'].isna())  # 11

df.loc[df['Rating']>4.9 , 'Type'].unique() # array(['Free', nan, 'Paid'], dtype=object)

type(df)

df1=df.iloc[:,[3,4]]
df1.agg(np.mean,axis=0)#.agg(fun,axis) applies a function to rows or columns of a dataframe.Compute the mean of each variable (each column) 
df1.agg(np.sum,axis=0)#Computes the summation of Rating and Reviews for each row. axis=1 means than apply the function on each row (move columnwise)
df1.agg(['sum','min','max'])#Applies two functions on both variables 
'''       Rating       Reviews
    sum  39260.0  4.762137e+09
    min      1.0  0.000000e+00
    max      5.0  7.815831e+07
'''

df1.agg(['sum','min','max']).index        #Index(['sum', 'min', 'max'], dtype='object')
df1.agg(['sum','min','max']).loc['sum',:]          # shows sum for both column
df1.agg(['sum','min','max']).loc['sum','Rating']   # 39260.000000

#dictionary's dataTypes
df.agg({'Reviews':['sum','min'],'Rating':['mean','max']})#Applies various functions on different columns

df['Last.Updated']=df['Last.Updated'].agg(pd.Timestamp)#Change the Last.Updated column to dates 
df['Last.Updated'].head(20)

df['Last.Updated'][0]-df['Last.Updated'][1]  # Timedelta('-8 days +00:00:00') Compute the differencesin days and hours
df.loc[0,'Last.Updated'].date()              #Get the date  datetime.date(2018, 1, 7)
df.loc[0,'Last.Updated'].day                 #Get the day
df.loc[0,'Last.Updated'].day_name()
df.loc[0,'Last.Updated'].dayofweek
df.loc[0,'Last.Updated'].month_name()
up=df.loc[0,'Last.Updated'].now()-df.loc[0,'Last.Updated']#Get the difference between today and and the first value in Last.Updated
up.days     #Get the difference in days
up.asm8
up.components
up.value

pd.Timestamp('1.9.2020').day             # 9
pd.Timestamp('1.9.2020').month           # 1
pd.Timestamp('1.9.2020').month_name()    # January

pip install persiantools
from persiantools.jdatetime import Jalalidate
v=Jalalidate(pd.Timestamp('1/9/2020'))
v.today()
pd.Timedelta()

'Write a function in python'
def interval(x):#This function computes the days from the last time the app was updated
    #x is the date taken from Last.Updated variable, like df.loc[0,'Last.Updated']
    x=pd.Timestamp(x)
    interval=(x.now()-x).days
    return(interval)
df['Updates']=df['Last.Updated'].agg(interval)#Add a column to the dateframe called "Updates"

np.mean(df['Updates'].loc[df['Rating']>4.9])
np.std(df['Updates'].loc[df['Rating']>4.9])


'Group the dataframe'
df.loc[df['Type'].isna(),'Type']='Not Available'#Change the missing values to a category
np.unique(df['Type']) #array(['Free', 'Not Available', 'Paid'], dtype=object)
df['Type'].unique()  # array(['Free', 'Not Available', 'Paid'], dtype=object)
sum(df['Type']=='Not Available')  # 400

df2=df.groupby('Type')#Group the data based on the variable Type
df2.count()
df2.agg({"Rating":['mean','std']})#Compute the mean and standard deviation of Rating for each level of variable Type
df3=df2.agg({"Rating":['mean','std'],"Reviews":['mean','std'],"Updates":['mean','std']})
df2.get_group('Free')   # note that only groupby has this function and dataframe has not it


'Multi index dataframe'
df3.iloc[:,1]
df3.columns
df3.loc[:,['Reviews','Rating']] # recall two column
df3.loc[:,('Reviews', 'std')]#reCall one column and its sub column
df3.loc['Free',('Reviews', 'std')]#Call an element using its row and column names
df3.loc[:,[('Rating','mean'),('Reviews','std')]]
df3.loc['Paid',['Reviews','Updates']]

df.dropna(axis=0,inplace=True)#Delete the missing values (rows)
df4=df.unstack()#Vectorize the dataframe


df5=df.groupby('Category').agg({"Reviews":['mean'],'Rating':['mean']})#Group by categpry and compute the mean of Reviews and Rating 
df6=df5.sort_values(('Reviews','mean'),ascending=False)#sort the dataframe based on the Reviews
df6=df6.iloc[:10,:]


'plot a histogram for Rating' 
plt.hist(df['Rating'], 50, facecolor='g',edgecolor="r")

'Plot a pie chart for mean of reviews'
exp=np.zeros(10)
exp[0]=0.2
plt.pie(df6['Reviews'],labels=df6.index,explode=exp)

'Plot a violinplot for Rating'
plt.violinplot(df['Rating'],showmedians=True)


'Plot a barchart for number of apps in a category'
df7=df.groupby('Category')#Group by app category
plt.bar(df7.count().index,df7.count().App)#A barplot for number of apps in categories
plt.xticks(np.arange(0,33),df7.count().index,rotation=90)#Rotate the text to make it vertical


'make a list of Rating for each category as a component in the list'
l=[]#Define the initial value for a list
for x in np.unique(df['Category']):#Repeat for each category
 l.append(df['Rating'].loc[df['Category']==x])#Add a component to the list l. The component is the vector of Ratings for category x 
 plt.boxplot(l,notch=True)#Boxplots for Rating associated with each category
 plt.xticks(np.arange(1,34),np.unique(df['Category']),rotation=90)#Add and rotate the text
 plt.axhline(np.mean(df['Rating']),color='red')
 

 plt.scatter(df['Reviews'],df['Rating'],marker="o",c="red")#Make a Scatter plot of Reviews and Rating
 plt.xlabel('Reviews')
 plt.ylabel('Rating')
 

h=[]
for x in np.unique(df['Type']):#Repeat for each category
 h.append(df['Rating'].loc[df['Type']==x].dropna())#Add a component to the list l. The component is the vector of Ratings for category x 
 plt.boxplot(h,notch=True)#Boxplots for Rating associated with each category
 plt.xticks(np.arange(1,4),np.unique(df['Type']),rotation=90)#Add and rotate the text


I=[]
for x in np.unique(df['Type']):#Repeat for each category
 I.append(df['Reviews'].loc[df['Type']==x].dropna())#Add a component to the list l. The component is the vector of Ratings for category x 
 plt.boxplot(I,notch=True)#Boxplots for Rreviews associated with each category
 plt.xticks(np.arange(1,4),np.unique(df['Type']))#Add and rotate the text
 plt.xlabel('Type')
 plt.ylabel('Reviews')
#'''
