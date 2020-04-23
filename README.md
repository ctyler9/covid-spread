# covid-spread
Final project for CX4242 Spring 2020, looks to forecast the spread of COVID-19. Uses the data from John Hopkin's to populate a networkX graph which a gillespie algorithm is run to simulate the spread of the virus in a D3 webpage. 

This specific example looks to forecast the spread of the virus in New England: 
https://eholland7.github.io/2020-04-22-COVID-19-Cases-Projection/


## Model
Three components:
  ```
  DataManipulation.py
  gispPrediction.py
  util.py
  ```
  
 ### DataManipulation - 
 High Level:
 Takes in current date data from John Hopkin's and creates a networkX graph of each state called populated at the county level. 
 
 Use: 
 To use, call the UnitedStatesMap class (it is automatically initialized) and then call the following methods:
    ```
    - class.make_state(state) -> returns a networkX object of the state (str) passed in
    - class.SIR() -> returns the initial susceptible, infected, and recovered numbers per state
    ```
    
    
### gispPrediction - 
High Level:
Runs the gillespie algorithm on the networkX object passed into the class, where each node represents a person (options of per 100, 1000) and each edge gives it a chance to spread. 

Use:
To use, initialize the gispPrediction class with: 
   ```
   tmax -> how many days to run the prediction for
   t0 -> starting time
   infection_rate -> infection rate of virus
   recovery_rate -> recovery rate of virus
   s0 -> population that is initially susceptible
   i0 -> population that is initially infected
   r0 -> population that is initially recovered
   graph -> take networkX object of state in question
  ```
  
Then, run these methods (which are automatically called in the main function):
  ```
  gillespie() -> runs algorithm
  created_date(state) -> takes in input from algorithm and converts it in to a CSV with rows of counties and columns of dates and values of number infected; passes in state name (str)
  ```

### util - 
Calls all of the dependencies needed to run as well as some overflow functions. 


### General Use: 
To run all the functions together, just simply run the gispPrediction.py... it has everything set up with the main() function. You can customize the output for the main() function by passing in lists of specific states, infection rates, as well as an intege value of tMax and t0. 



## Visualization

  
