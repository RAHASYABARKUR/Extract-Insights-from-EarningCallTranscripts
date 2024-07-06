#GEID 1011139629
#Nimit Shah
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import time
import os
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


# In[2]:


def get_transcript(url, title_flag = True) : #get title or transcript
    print(url)
    ap = requests.get(url)

    my_soup = bs(ap.content, "html.parser")

    if title_flag :
        x = my_soup.find("div", {"id":"a-cont"}) #transcript
        return x.get_text()
    else :
        x = my_soup.find("div", {"id":"a-hd"}) #title
        return x.h1.get_text()
        


# In[3]:


def get_name(url) : #get name of company based on search url
    print(url)
    ap = requests.get(url)
    soup = bs(ap.content, "html.parser")
    name_container = soup.find("div", {"class" : "symbol_title"})
    return name_container.h1.get_text()
    


# In[4]:


def open_new_tab(input_driver, i) : 
    arg1 = "window.open('about:blank', "
    arg2 = "'tab" + str(i) +"\'"
    arg3 = ");"
    
    arg = arg1 + arg2 + arg3

    #input_driver.execute_script("window.open('about:blank', 'tab2');")
    input_driver.execute_script(arg) #opens a new tab
    input_driver.switch_to.window(arg2.split("'")[1]) #switches to new tab


# In[8]:


#Following code searches for earnings call transcripts for various companies on seekingalpha.com, extracts all the URLs and scrapes the data from them
def scrape(email, pwd, geckodriver_path, tokens) :#id, pass for seekingalpha website, path to chrome geckodriver.exe file, seekingalpha tokens for companies
    os.mkdir('seeking-alpha-final1')
    os.chdir('seeking-alpha-final1')

    base_url_1 = "https://seekingalpha.com/symbol/"
    base_url_2 = "/earnings/transcripts"

    login_url = "https://seekingalpha.com/account/login"

    #email = "annlinchacko@gmail.com"
    #pwd = "annlinchacko"

    #tokens = ["fb", "amzn", "goog", "cs", "c", "ms", "jpm", "v", "db", "aapl", "bac", "gs", "wfc"] 
    #company tags wrt seeking alpha 

    #random.shuffle(tokens)

    capa = DesiredCapabilities.FIREFOX
    capa["pageLoadStrategy"] = "none"

    #driver = webdriver.Chrome(executable_path = r'C:\Users\NIMIT\AppData\Roaming\Python\Python38\chromedriver.exe')
    driver = webdriver.Chrome(executable_path = geckodriver_path)

    driver.maximize_window()

    driver.get(login_url)


    input_email = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#login_user_email"))) #get login field as soon as it appears
    input_email.send_keys(email) 

    input_pwd = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#login_user_password"))) #get pwd field as soon as it appears
    input_pwd.send_keys(pwd)

    input_pwd.send_keys(Keys.ENTER)
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#main_container"))) #wait for the login to proceed

    open_new_tab(driver, 2)

    for idx, token in enumerate(tokens) : 

        input_url = base_url_1 + token.upper() + base_url_2

        cname = get_name(input_url) #make folder of this name

        os.mkdir(cname)
        os.chdir(cname)

        driver.get(input_url)

        time.sleep(10)
        driver.refresh()
        time.sleep(5)

        SCROLL_PAUSE_TIME = 10 #time between scrolls 

        last_height = driver.execute_script("return document.body.scrollHeight")

        #keep scrolling to the bottom until all invisible links appear
        while True:
            print('inside while')
            driver.execute_script("window.scrollTo(0, (document.body.scrollHeight))//2;")

            time.sleep(SCROLL_PAUSE_TIME) #wait for the page to load 

            new_height = driver.execute_script("return document.body.scrollHeight") #calculate new scroll height and check with previous height
            if new_height == last_height:
                break
            last_height = new_height

        #extract urls
        for a in driver.find_elements_by_xpath('.//a') : #find all html elements with a tag 
            link = str(a.get_attribute('href')) #get href value from a tag
            if 'earnings-call-transcript' in link : #check if the link is for an earnings call
                #print(link)
                transcript = get_transcript(link)
                filename = get_transcript(link, title_flag = False) + '.txt'
                f = open(filename, "w+")
                f.write(transcript)
                f.close()

        os.chdir('..')
        open_new_tab(driver, idx+3)

    os.chdir('..')


# In[ ]:


email = "annlinchacko@gmail.com"
password = "annlinchacko"
tokens = ["fb", "amzn", "goog", "cs", "c", "ms", "jpm", "v", "db", "aapl", "bac", "gs", "wfc"]
gecko_path = r'C:\Users\NIMIT\AppData\Roaming\Python\Python38\chromedriver.exe'

scrape(email, password, gecko_path, tokens)
