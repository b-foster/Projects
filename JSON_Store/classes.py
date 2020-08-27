'''
Hello, and welcome to my week 10 programming assignment

Introduction to Object Oriented Programming

DSC 510

Programming Assignment Week 10

Author: Brett Foster

July 28, 2020
'''

# Importing all necessary modules
from jsonstock import *
import time
import json
from itertools import cycle
import re
import locale

# Ultimately creates a "CashRegister" for the user to to use as they shop
class CashRegister:
    
    # Creates an empty list of items that the user adds to their cart
    items = []
    
    # Total of the items in the cart before tax is added
    pretax_total = 0
    
    # Tax rate (Picked California tax rate)
    tax_rate = 0.0775
    
    # Rate for member discount that can be added later
    discount_rate = 0.90
    
    # Initiates the creattion of a new customer
    def __init__(self, customer):
        self.customer = customer
        
    # Function for adding items to the CashRegister, item and total
    def add_item(self, brand, category, price):
        
        CashRegister.items.append(Items(new_user, brand, category, price))
        CashRegister.pretax_total += float(price)
    
    # Responsible for calculating the discount
    def club_discount(self):
        CashRegister.pretax_total = CashRegister.pretax_total * CashRegister.discount_rate
        
    # Funtion for finding tax and printing total information
    def get_total(self):
        
        tax = self.pretax_total * self.tax_rate
        print("Tax: {}".format(locale.currency(tax)))
        print("Total comes to: {}".format(locale.currency(self.pretax_total + tax)))
        print("\n")
        print("Thank you for shopping with us {}. Come back soon!".format(user_name))
    
    # Keeps track of the items in the cart
    def get_count(self):
        
        return self.items

# Class for the items available in the store
class Items(CashRegister):
    
    # Initiates the creation of each item
    def __init__(self, name, brand, category, price):
        self.name = CashRegister
        self.brand = brand
        self.category = category
        self.price = price
    
    # Function for adding a new item to the CashRegister
    def new_item(self):

        print('Adding to cart: {} - {}.'.format(self.brand, self.category))
        super().add_item(self.brand, self.category, self.price)
    
# Introduction
print("Hello. Welcome to the Foster Wholesale Tennis Cash Register Application.")
print("We are currently sponsored by Wilson, Babolat, Nike, and Adidas, only selling their products.")

time.sleep(1)

# Imports the inventory information
data = json.loads(inventory)

# Options for displaying the inventory
def inv_options():
    global a_or_r
    
    # User choice on display
    a_or_r = input("Are you looking for tennis apparel or racquets, or enter store to see all the inventory?" + "\n")
    
    while True:
        
        # Condition to break from the inventory store
        if a_or_r in ['', ' ']:
            break
        
        # Condition to break from the inventory store
        elif a_or_r[0].lower() in ['e', 'n', 'd']:
            break
        
        # Condition that shows the user the apparel inventory
        elif a_or_r[0].lower() == 'a':
            print("A list of our options are provided below." + "\n")
            app_inventory()
            break
        
        # Condition that shows the user the racquet inventory
        elif a_or_r[0].lower() == 'r':
            print("A list of our options are provided below." + "\n")
            rac_inventory()
            break
        
        # Condition that shows the user the entire store inventory
        elif a_or_r[0].lower() == 's':
            print("A list of our options are provided below." + "\n")
            store_inventory()
            break
        
        # Condition that returns an error and restarts the inventory loop
        else:
            print('Invalid entry, please try again. Options: Apparel, racquets, store, or exit to quit.')
            inv_options()
            break
            
# store_inventory nicely displays the entire inventory
def store_inventory():
    dash = '-' * 34
    print('INVENTORY')
    for item in data['stock']:
        print(dash) 
        print('{:<2}. {:<7} - {:<10} ==>  ${:<40}'.format(item['id'], item['brand'], item['category'], item['price']))
    shop()

# app_inventory nicely displays the apparel inventory        
def app_inventory():
    dash = '-' * 34
    print('APPAREL')
    for item in data['stock']:
        if item['id'] <= 14:
            print(dash) 
            print('{:<2}. {:<7} - {:<10} ==>  ${:<40}'.format(item['id'], item['brand'], item['category'], item['price']))
    shop()

# rac_inventory nicely displays the racquet inventory  
def rac_inventory():
    dash = '-' * 34
    print('RACQUETS')
    for item in data['stock']:
        if item['id'] > 14:
            print(dash) 
            print('{:<2}. {:<7} - {:<10} ==>  ${:<40}'.format(item['id'], item['brand'], item['category'], item['price']))
    shop()    

# User option function on whether or not to purchase more items
def more():
    purchase_more = input("Would you like to purchase any more items?" + "\n")
    
    while True:
        if purchase_more in ['', ' ']:
            break
        
        # Restarts inventory options if user chooses yes
        elif purchase_more[0].lower() == 'y':
            inv_options()
            break
        
        elif purchase_more[0].lower() == 'n':
            break
            
        elif purchase_more not in ['y', 'n']:
            break
        
        else:
            print("Invalid entry, please try again")

# purchase function allows the user to enter product numbers only, and creates new class Items for each new one
def purchase():
    
    while True:
        global a_or_r
        global user_num
        global new_user
        
        user_num = input("Enter a product number you would like to purchase here, or done to exit." + "\n")
        
        if user_num in ['', ' ']:
            break
        
        elif user_num[0].lower() in ['d', 'n']:
            break
        
        elif user_num in ['', ' ']:
            break
        
        # allows the class Items to be added to the unique "CashRegister"
        elif int(user_num) in range(1, 23):
            user_num = int(user_num) - 1
            Items(new_user, data['stock'][user_num].get('brand'), data['stock'][user_num].get('category'), data['stock'][user_num].get('price')).new_item()
            print("Complete!")
            continue
        
        else:
            print("Invalid entry, please try again.")

# shop functionstarts the purchase function initiation
def shop():
    
    user_question = input("Is there anything you would like to purchase (y,n)?" + '\n')
     
    if user_question in ['', ' ']:
        print("Invalid entry, please try again.")
        pass
    
    elif user_question[0].lower() == 'y':
        print("Great!")
        purchase()
        
    elif user_question[0].lower() == 'n':
        other = input("Would you like to shop any other categories?" + "\n")
        
        while True:
            
            if other[0].lower() == 'y':
                inv_options()
                purchase()
                break
            
            elif other[0].lower() == 'n':
                print("Alright. You now know where to find us for all your tennis equipment needs.")
                break
            
            else:
                print("Invalid entry, please try again.")
        
    else:
        print("Invalid entry, please try again.")
        shop()
        
# The check function allows the user to enter their email to "sign up" and apply a discount        
def check():
    global email
    global new_user
    
    email = input("Enter your email here:" + "\n")
    
    if email in ['', ' ']:
        print("Invalid email")
        return False
    
    # checks if the user entered email is vaild
    else:
        regex = re.search(r'[\w.-]+@[\w.-]+.\w+', email)
        
        # If the email is valid, will apply the current discount rate - 10%
        if regex:
            print("Valid email. Applying discount.")
            new_user.club_discount()
            print("New Pre-tax total: {}".format(locale.currency(CashRegister.pretax_total, grouping=True)))

        else:
            print("Invalid email.")
            print("You are welcome to sign up next time")
            
# discount function applies discount if a member or allows user to sign up
def discounts():
    
    club = input("Are you a memeber of our Club Tennis Group?" + "\n")
    while True:
        
        if club in ['', ' ']:
            print("Invalid entry, please try again.")
        
        # If the user is a member, applies a discount
        elif club[0].lower() == 'y':
            print("Perfect! We'll apply your discount now." + "\n")
            new_user.club_discount()
            print("New Pre-tax total: {}".format(locale.currency(CashRegister.pretax_total, grouping=True)))
            break
        
        # If the user is not a member, runs the check() function if they would like to join
        elif club[0].lower() == 'n':
            print("We can sign you up with just your email and you will receive 10% off your order today.")
            
            try:
                join = input("Would you like to sign up?" + "\n")
                if join[0].lower() not in ['y','n']:
                    raise ValueError
            except ValueError:
                print("Invalid entry, please try again.")
              
                    
            while True:
                
                if join[0].lower() not in ['y','n']:
                    print("Invalid entry, please try again.")
                    continue
                elif join[0].lower() == 'y':
                    check()
                    break
                elif join[0].lower() == 'n':
                    print("Next time then!")
                    break
            break
    
        else:
            print("Invalid entry, please try again.")
            
# Final user function that displays the users cart, total and tax, as well as discounts if applicable
def checkout():
    if int(len(new_user.get_count())) > 0:
        print('\n' + 'Your cart currently has ' + str(len(new_user.get_count())) + ' items.')
        
        # Displays the users items in their cart 
        for key, item in enumerate(new_user.items):
            print('{}.{} - {}: ${}'.format(key + 1, item.brand, item.category, item.price))
        
        print("\n")
        
        print("Pretax total: ${:.2f}".format(CashRegister.pretax_total))
        
        discounts()
        
        new_user.get_total()
        
    else:
        print('Thank you your interest.')

# The main function of the program, housing all other necessary functions    
def main():
    global new_user
    global user_name
    
    locale.setlocale(locale.LC_ALL, '')
    
    # Begins by creating a new CashRegister class for the user
    user_name = input("Who do we have the pleasure of helping shop today?" + "\n")
    new_user = CashRegister('{}'.format(user_name))
    
    # Intoduction of the program to the user
    print("Great " + user_name + "! What are you looking for today?")
    
    print("We have apparel from Adidas and Nike.")
    print("And different tennis racquets from Wilson and Babolat.")
    
    inv_options()

    more()    

    checkout()
    
    print("Best of luck on the tennis courts!")

if __name__ == '__main__':
    main()
