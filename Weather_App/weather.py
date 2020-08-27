
'''
Hello, and welcome to my Weather Handling Application

Final Project

DSC 510

Week 10, Course Project

Author: Brett Foster

July 30, 2020
'''

# Imports necessary modules that are being used
import requests
import json
from datetime import datetime
import random
import csv
import time

# api-key: 77d31578acad00e89124a9883284b028

country_dict = {}
country_list = []

# Opens data files to analyze json data
with open("cityinfo.json", "r", encoding='utf-8') as f:
    data = json.load(f)
    
with open("states.json", "r", encoding='utf-8') as s:
    states = json.load(s)
    
# creates a dictionary for countries and country codes to be used with the API data
for i in data:
    if i["country"] in country_dict:
        country_dict[i["country"]] += 1
    else:
        country_dict[i["country"]] = 1

for key in country_dict:
    country_list.append(key)

# Introduction to the user
def intro():
    print("Hello, welcome to the Wonderful Weather App!")
    print("We are currently offering weather details using zipcodes or city/state combinations within the United States.")
    print("As well as weather information from around the globe, using a city and country combination.")
    print("Example: London, United Kingdom")
    print("Additionally, for some surprise, we offer weather information from a random location around the world")

# Initiates user choice and how they would like to receive weather data
def user_choice():
    print("Our options are as follows:" + "\n")
    print("1. United States locations.")
    print("2. Locations from Around the World.") 
    print("3. Random Location.")

# Controls the user_choice and where the user will ultimately be led
def choice_option():
    
    while True:
        try:
            user_option = input("Which option would you like to choose?" + "\n")
        
        except:
            if user_option not in ['1', '2', '3']:
                print("Invalid entry, please try again.")
       
        # Sents the user to the United States weather information function
        if user_option == '1':
            us_info()
            break
        
        # Sents the user to the weather information function for around the world
        elif user_option == '2':
            world_info()
            break
        
        # Provides the user with a random locations weather
        elif user_option == '3':
            random_info()
            break
            
        else:
            print("Invalid option, please try again.")
            break
  
# If initiated by a user, provides the user with additional information about the locations weather        
def more_info(response):
    try:
        sunrise = datetime.fromtimestamp(response['sys']['sunrise']).strftime('%H:%M:%S')
        sunset = datetime.fromtimestamp(response['sys']['sunset']).strftime('%H:%M:%S')
        low = response['main']['temp_min']
        high = response['main']['temp_max']
    except KeyError:
        print("Error handling weather request.")
    
    # User choice whether they want more information
    while True:
        info_choice = input("Would you like more weather information about this location (y/n)?" + "\n")
         
        if info_choice[0].lower() not in ['y', 'n']:
            print("Invalid entry, please try again.")
          
        # If the user chooses so, nicely displays the weather information
        elif info_choice[0].lower() == 'y':
            try:
                print("The temperature low for today is {}\u00b0F, while the high will be {}\u00b0F.".format(low, high))
                print("Sunrise: {} AM PST".format(sunrise))
                print("Sunset: {} PM PST".format(sunset))
                break
            except:
                print("")
        
        elif info_choice[0].lower() == 'n':
            break
        
        else:
            print("Oops, something went wrong.")
            break
        
# Fucntion for randomly picking a random location from the cityinfo.json file
def random_info():
    new = data[random.randint(1,140000)]

    # Opens the country_codes.txt file to accurately display the correct country info
    with open("country_codes.txt", "r") as csv_file:
        csv_data = csv.reader(csv_file, delimiter=",")
        for row in csv_data:
            if row[1] in new['country']:
                s_country = row[0]
    
    # Grabbing the correct API data from open weather
    random_api = "http://api.openweathermap.org/data/2.5/weather?q={},{}&units=imperial&APPID=77d31578acad00e89124a9883284b028".format("+".join(new['name'].split()), new['country'])
    
    random_test = requests.get(random_api)
    
    random_res = requests.request("GET", random_api).json()   
    
    # Prints out the weather information if the API response is valid
    if random_test.status_code == 200:
        print("\n" + "Your random location is {}, {}.".format(new['name'], s_country))
        print("The current temperature is {}\u00b0F.".format(random_res['main']['temp']))
        print("The temperature feels like {}\u00b0F, with on a humidity of {}.".format(random_res['main']['feels_like'], random_res['main']['humidity']))
        print("Weather description: {}.".format(random_res['weather'][0]['description']))
        more_info(random_res)
    else:
        print("Unfortunately, our system does not provide weather information for this location yet.")

# Function that allows the user to pick city/country information for around the world weather information
def world_info():
    print("Our system is still being updated, so please be patient.")
    
    # City and country input
    city = input("Please enter the name of the city you are would like weather information about." + "\n")
    country = input("Please enter the state this city is located in:" + "\n")
    
    # If the user enters a country name, the program with fetch the counties 2 letter code from the country_codes.txt file
    if len(country) > 2:
        with open("country_codes.txt", "r") as csv_file:
            csv_data = csv.reader(csv_file, delimiter=",")
            for row in csv_data:
                if row[0].lower() == country:
                    country = row[1]
    
    # API location based on user input
    world_api = "http://api.openweathermap.org/data/2.5/weather?q={},{},us&units=imperial&APPID=77d31578acad00e89124a9883284b028".format("+".join(city.split()), "+".join(country.split()))
    
    world_test = requests.get(world_api)
    
    # Tests the request, make sure it has some validity
    try:
        world_res = requests.request("GET", world_api).json()
    except Exception as e:
        print(e)  
    
    # If the request is valid, the city/country code, the program will display the weather information
    if world_test.status_code == 200:
        print("The current temperature in {}, {}, is {}\u00b0F.".format(world_res['name'], country.upper(), world_res['main']['temp']))
        print("The temperature feels like {}\u00b0F, with on a humidity of {}%.".format(world_res['main']['feels_like'], world_res['main']['humidity']))
        print("Weather description: {}.".format(world_res['weather'][0]['description']))
        more_info(world_res)
    else:
        print("Invalid city/country combination, please try again.")

# United states weather information function
def us_info():
    global info_res
    
    # Zip code or city/state weather lookup
    print("How would you like to look up a US location?")
    print("1. City, State" + "\n" + "2. Zip Code")
    
    while True:
        try:
            csz = input("Please choose an option:" + "\n")
        
        except:
            if csz not in ['1', '2']:
                print("Invalid entry, please try again. (1 or 2)")
        
        # User choice for city/state weather lookup
        if csz == '1':
            city = input("Please enter the name of the city you are would like weather information about." + "\n")
            state = input("Please enter the state this city is located in:" + "\n")
            if len(state) > 2:
                for key, value in states.items():
                    if value.lower() == state:
                        state = key
               
            # city/state API data retrieval
            cs_api = "http://api.openweathermap.org/data/2.5/weather?q={},{},us&units=imperial&APPID=77d31578acad00e89124a9883284b028".format("+".join(city.split()), "+".join(state.split()))
            
            cs_test = requests.get(cs_api)
            
            try:
                cs_res = requests.request("GET", cs_api).json()
            except requests.RequestException as e:
                print("OOPS!! Error occured. " + str(e))
           
            # Prints city/state weather information if valid
            if cs_test.status_code == 200:
                print("The current temperature in {},{}, is {}\u00b0F".format(city.title(), state.title(), cs_res['main']['temp']))
                print("The temperature feels like {}\u00b0F, with on a humidity of {}.".format(cs_res['main']['feels_like'], cs_res['main']['humidity']))
                print("Weather description: {}".format(cs_res['weather'][0]['description']))
                more_info(cs_res)
            else:
                print("Invalid city/state combination, please try again.")
            
            break
        
        # Weather lookup based on zip code user choice
        elif csz == '2':
            while True:
                try:
                    zippy = input("Please enter the zip code you are would like weather information about." + "\n")
                    if len(zippy.strip()) != 5:
                        raise ValueError
                    break
                
                except ValueError:
                    print("Not a valid zip code.")
               
            # Zip code data API based on user entered zip code
            zip_api = "http://api.openweathermap.org/data/2.5/weather?q={},us&units=imperial&APPID=77d31578acad00e89124a9883284b028".format(zippy)
            
            zip_test = requests.get(zip_api)
            
            try:
                zip_res = requests.request("GET", zip_api).json()
            except json.JSONDecodeError:
                print("Invalid zip code entered.")
            
            # If the zip code API is valid, the program will nicely print weather information
            if zip_test.status_code == 200:            
                print("The current temperature in {},{}, zip code: {}, is {}\u00b0F.".format(zip_res['name'],zip_res['sys']['country'], zippy, zip_res['main']['temp']))
                print("The temperature feels like {}\u00b0F, with on a humidity of {}.".format(zip_res['main']['feels_like'], zip_res['main']['humidity']))
                print("Weather description: {}".format(zip_res['weather'][0]['description']))
                more_info(zip_res)  
            break

# User input on continuing the weather program
def run_again():
    
    while True:
        try:
            more_weather = input("Would you like to get information about a different location?" + "\n")
        
            # Runs the program again if the user chooses so
            if more_weather[0].lower() == 'y':
                user_choice()
                
                choice_option()
                
                run_again()
                
                break
            
            # Ends the program
            elif more_weather[0].lower() in ['n', 'd']:
                print("Thank you for visisting. Enjoy the weather out there!")
                return False
                break
            
            else:
                print("Invalid entry, please try again.")
       
        except TypeError:
            print("Invalid entry, please try again.")
            
# Main program of the python file, houses all other initiated programs
def main():
    global zippy
    global city
    global state
    
    intro()
    
    user_choice()
    
    choice_option()
    
    run_again()


if __name__ == '__main__':
    main()
