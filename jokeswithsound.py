'''
Hello, and welcome to my week 9 programming assignment

Introduction to Web Services (Jokes with Sound)

DSC 510

Programming Assignment Week 9

Author: Brett Foster

July 18, 2020
'''

import time
import requests
import json
import pyttsx3

engine = pyttsx3.init()

url_random = "http://api.chucknorris.io/jokes/random"

url_categories = "https://api.chucknorris.io/jokes/categories"

response = requests.request("GET", url_random)
categories = requests.request("GET", url_categories)
cat_json = categories.json()

print("Hello and welcome to the Foster's Home for Chuck Norris Jokes!")
print("The Online Leader in all Chuck Norris Jokes!!")

time.sleep(1)

print("Some of the highlights you can experience are random jokes, or categorized jokes.")
print("The laughter is up to you!")

def choice():
    print("Awesome! Would you like to hear a joke from a category or at random?")
    user_choice = input("Please answer with category or random." + "\n")
    clean_choice = user_choice[0].lower().replace(" ", "")
    
    while True:
        if clean_choice == 'c':
            category_joke()
            break
        
        elif clean_choice == 'r':
            random_joke()
            break
        
        elif user_choice not in ['c','r']:
            print("Invalid entry, please try again.")
            break
        
        else:
            print("Something went wrong.")
            break
    
    run_again()

def random_joke():

    print(response.json()["value"])
    engine.say(response.json()["value"])
    engine.runAndWait()
    
def category_joke():
    
    print("\n")
    print("Great! Please choose from the categories are listened below." + "\n")
    
    for x in range(len(cat_json)):
        print(cat_json[x].title())
    
    while True:
        user_category = input("Enter your choice of category here." + "\n")
        clean_category = user_category.lower().replace(" ", "")
        
        if clean_category not in cat_json:
            print("Not a category option, please try again.")
        else:
            url_cat = "https://api.chucknorris.io/jokes/random?category={input}".format(input=clean_category)
            break
        
    url_cat = "https://api.chucknorris.io/jokes/random?category={input}".format(input=clean_category)
    
    new_cat = requests.request("GET", url_cat)
    
    print(new_cat.json()['value'])
    engine.say(new_cat.json()['value'])
    engine.runAndWait()

def run_again():
    
    while True:
        user_continue = input("Would you like to hear another joke?" + "\n")
        clean_continue = user_continue[0].lower()
        if clean_continue == 'y':
            choice()
            break
        elif clean_continue == 'n':
            print("Sounds good! See you next time!")
            break
        else:
            print("Oops, something went wrong.")
            
    return False

def main():
    
    print("Would you like to hear a Chuck Norris joke?")
    
    while True:
        user_answer = input("Please answer with yes or no." + "\n")
        clean_answer = user_answer[0].lower()
        
        if clean_answer == 'y':
            choice()
            break
        
        elif clean_answer == 'n':
            print("Alfighty then! But you now know where to fine the best Chuck Norris jokes!")
            break
        else:
            print("Invalid entry, please try again.")
        
        
if __name__ == '__main__':
    main()
    
