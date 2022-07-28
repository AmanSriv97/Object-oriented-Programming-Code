#!/usr/bin/env python
# coding: utf-8

# In[7]:


import time
from random import randint


# In[8]:


class Virus:
    def __init__(self):
        self.wave = 0
        self.__lives = [100, 100, 200, 250]
        self.__names = ["ALPHA", "BETA", "GAMMA", "SIGMA"]
        self.attRed = 0
        self.upgrade()
        
    def upgrade(self):
        self.wave += 1
        self.life = self.__lives[self.wave-1]
        self.name = self.__names[self.wave-1]


# In[9]:


class Vaccine:
    def __init__(self, vaccNum):
        self.__durabilities = [100, 150, 200, 250]
        self.__vaccineNames = ["covishield", "Covaxin", "Pfizer", "Sputnik"]
        self.name = self.__vaccineNames[vaccNum-1]
        self.vaccNum = vaccNum
        self.wave = 0
        self.boost = 10
        self.upgrade()
    
    def upgrade(self):
        self.wave += 1
        self.durability = self.__durabilities[self.vaccNum-1]
        self.boost *= self.wave


# In[10]:


class Action:
    def __init__(self):
        self.inj = [10, 5, 6, 4]
        self.eff = [10, 5, 4, 8]
        self.div = [4, 4, 3, 2]
    
    def inject(self, virus, vaccine):
        print("Your vaccine is boosted and it reduces the life of the virus by {}".format(self.inj[vaccine.vaccNum-1]))
        virus.life = max(0, virus.life-self.inj[vaccine.vaccNum-1])
        print("Vaccine’s DURABILITY : {} | Virus’s LIFE : {}".format(vaccine.durability, virus.life))
    
    def effect(self, virus, vaccine):
        print("Virus’s action reduced by {}".format(self.eff[vaccine.vaccNum-1]))
        virus.attRed = self.eff[vaccine.vaccNum-1]
        print("Vaccine’s DURABILITY : {} | Virus’s LIFE : {}".format(vaccine.durability, virus.life))
    
    def attack(self, virus, vaccine):
        print("VIRUS’s ACTION !")
        att = max(0, randint(0, virus.life//self.div[virus.wave-1]) - virus.attRed)
        print("Virus reduces the vaccine’s DURABILITY by {}".format(att))
        virus.attRed = 0
        vaccine.durability = max(0, vaccine.durability-att)
        print("Vaccine’s DURABILITY : {} | Virus’s LIFE : {}".format(vaccine.durability, virus.life))


# In[11]:


class Simulator(Action):
    def __init__(self):
        super().__init__()
        self.__db = {}
        self.loggedIn = None
        self.exit = False
        while not self.loggedIn and not self.exit:
            self.__home()
        while self.loggedIn and not self.exit:
            print("Testing for wave {}".format(self.__db[self.loggedIn][3]))
            self.fight()
            if self.loggedIn and self.__db[self.loggedIn][3] > 4:
                print('''Virus Defeated !
                            Vaccine proves to be effective during the fourth wave!!!
                            Congratulations, your vaccine has overpowered all the variants of the virus and hence
                            has proved to be effective in all the waves. Great Job!
                            Thanks for your participation. Now let’s get Vaccinated !!!''')
                break
    
    def __home(self):
        inp = int(input("Welcome To the Portal:\nPlease select your option:\n1. New User\n2. Existing Patient\n3. Exit\n"))
        if inp == 1:
            self.__register()
        elif inp == 2:
            self.__login()
        else:
            print("Thanks for your participation. Now Let’s get Vaccinated !!!")
            self.exit = True
    
    def __register(self):
        userName = input("Enter Username: ")
        aadhaar = input("Enter Aadhaar Number: ")
        if len(aadhaar) == 16 and aadhaar.isdecimal():
            vaccine = int(input("Choose your vaccine:\n1. Covishield\n2. Covaxin\n3. Pfizer\n4. Sputnik\n"))
            if vaccine < 1 or vaccine > 4:
                print("Please Enter correct vaccine number!!\nUser not registered!!")
            else:
                self.__db[userName] = [aadhaar, Vaccine(vaccine), Virus(), 1]
                print("Patient has been registered.\nUsername - {}\nAadhaar - {}\nVaccine Opted - {}".format(userName, aadhaar, self.__db[userName][1].name))
        else:
            print("Please Enter correct Aadhaar number!!")
    
    def __login(self):
        userName = input("Enter Registered Username: ")
        print("Verifying...")
        if userName in self.__db:
            print("Patient Found!!\nWelcome {},".format(userName))
            self.loggedIn = userName
        else:
            print("Patient not found !!! If you haven’t registered yet then please register first.")
    
    def fight(self):
        vacc = self.__db[self.loggedIn][1]
        virus = self.__db[self.loggedIn][2]
        while vacc.durability > 0 and virus.life > 0:
            print("Vaccine’s BOOST : {} | Vaccine’s DURABILITY : {} | Virus’s LIFE : {} | Virus Variant : {}".format(vacc.boost, vacc.durability, virus.life, virus.name))
            inp = int(input("Please select an action:\n1. INJECT\n2. EFFECT\n3. Exit\n"))
            if inp == 1:
                super().inject(virus, vacc)
            elif inp == 2:
                super().effect(virus, vacc)
            elif inp == 3:
                print("Exited at wave {}".format(self.__db[self.loggedIn][3]))
                self.loggedIn = None
                print("Thanks for your participation!! Let's get vaccinated!!")
                break
            else:
                print("Please provide correct input!!")
                continue
            if virus.life > 0:
                super().attack(virus, vacc)
        
        if self.loggedIn:
            if vacc.durability <= 0:
                print("Oops! Your vaccine fails to affect the {} Variant.\nHowever, The vaccine helps you fight against several attacks of the virus and proves to be useful. This shows how important the vaccine is in the fight against COVID-19.\nThanks for your participation. Now Let’s get Vaccinated !!!".format(virus.name))
                self.exit = True
            elif virus.life <= 0:
                print("Virus Defeated !\nVaccine proves to be effective during wave {}!!!\nMoving on to the next wave.".format(self.__db[self.loggedIn][3]))
                self.__db[self.loggedIn][3] += 1
                vacc.upgrade()
                virus.upgrade()


# In[13]:


sim = Simulator()


# In[ ]:




