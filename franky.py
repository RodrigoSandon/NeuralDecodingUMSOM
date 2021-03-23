#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 10:09:18 2021

@author: rodrigosandon
"""

#print("Hello World")

k = 5.00 #double

x = 5 #integer

y = 6*x + 16

#print("python" + " is cool")

# if 5 is greater than 2, print 5 is greater than 2

#Conditionals
x = 5

#Car game
budget = 5000
priceOfCar = 0






class priceOfCarGame:
    
    
    def __init__(self, default = 0):
        self.priceOfCar = default
        self.budget = 5000

    def typeOfCar(kindOfCar):
        global priceOfCar
        global budget
        
        if (kindOfCar == "Tesla"):
            return 6000
        if (kindOfCar == "Toyota"):
            return 2500
        
    def carEngine(fuel):
        if fuel == "diesel":
            return print("This is most likely a truck!")
        elif fuel == "oil":
            return print("This is a car!")
        else:
            return print ("This is an electric car!")
        
    def canIBuyIt(priceOfCar, budget):
        
        if priceOfCar > budget:
            print("Sorry you can't buy it")
        elif (priceOfCar == budget or priceOfCar < budget):
            print("Sold!")
                

#priceOfCarGame.carEngine("diesel")
#priceOfCarGame.typeOfCar("Tesla")

priceOfCarGame.canIBuyIt(priceOfCarGame.typeOfCar("Tesla"), budget)





list1 = [4,5,6,2,34,6,7]

#for loop
for x in list1: #how uch u wanna increment by is taken care of by python
    print(x)

#










