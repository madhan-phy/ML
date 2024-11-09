#Create Class: Class vs Instance 

se1 = ["Claire",23,"Junior Engineer",5000]
se2 = ["Chris",26,"Senior Engineer",8000]

class SoftwareEngineer: 
    def __init__(self,name,age,title,salary): #instance attributees
        self.name = name
        self.age = age
        self.title = title
        self.salary = salary
    
    alias = "White Collar Employee"  #class attributs
    
    

se1_instance = SoftwareEngineer(*se1)  #create instance using existing data
print(se1_instance.salary)

se3 = SoftwareEngineer("Thomas",56,"Principal Engineer",56000)
print (se3.age)