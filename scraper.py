import os
import sys
import requests
import time
import json
import shutil
from collections import defaultdict
import pickle
import select
import signal

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

DF_INPUT_PATH="./fruits"
IMAGE_OUTPUT_PATH = './fruit_images'
DF_OUTPUT_PATH = "./fruit_image_index"


MIN_REQUEST = 10
MAX_REQUEST = 40
CATEGORY_REQUEST = 1000

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
    
signal.signal(signal.SIGALRM, timeout_handler)

def download_images(search_text, image_output_path, df_in, df_out, num_requested=20):
    print('Search Query', search_text)
    number_of_scrolls = num_requested / 400 + 1
    image_to_text = {}

    url = f"https://www.google.com/search?q={search_text}&source=&source=lnms&tbm=isch"
    try:
        driver = webdriver.Chrome("./chromedriver")
        driver.get(url)
    except:
        print('Internet Connection Failing')
        print('Restart when Internet is back on')
        sys.exit()

    headers = dict()
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    #extensions = {"jpg", "jpeg", "png", "gif"}
    extensions = {"jpg"}
    image_count = 0
    downloaded_image_count = 0
    no_internet = False

    for _ in range(int(number_of_scrolls)):
        for _ in range(10):
            driver.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(0.2)
            
        time.sleep(0.5)

        try:

            driver.find_element_by_xpath("//input[@value='Show more results']").click()
            
        except Exception as e:
            print("WARNING*: LESS IMAGES FOUND")
            break


    images = driver.find_elements_by_xpath('//div[contains(@class, "rg_meta")]')
    
    print("Total images:", len(images))

    for image in images:

        image_count += 1
        image_url = json.loads(image.get_attribute('innerHTML'))["ou"]
        image_type = json.loads(image.get_attribute('innerHTML'))["ity"]
        #print(f"Downloading image {image_count}")

        try:
            # download jpg images only
            if image_type in extensions:
                image_type = "jpg"
                #print('er1')
                try:
                    signal.alarm(120)
                    r = requests.get(image_url, stream=True, headers=headers)
                except TimeoutException:
                    continue
                #print('er2')
                if r.status_code == 200:
                    tmp_image_name = (search_text + '_' + str(downloaded_image_count)+"."+image_type).strip().replace('/', '_').replace(' ', '_')
                    tmp_image_path = os.path.join(image_output_path, tmp_image_name)
                    with open(tmp_image_path, "wb") as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                        
                        #print('er3')

        except Exception as e:
            print(f"WARNING**: IMAGE DOWNLOAD FAILED: {e}")

        #try:
        if image_type in extensions:
            if r.status_code == 200:
                tmp = list(df_in.loc[df_in['search_query']==search_text].iloc[0])

                df_out.loc[df_out.shape[0]] = tmp
                df_out.at[df_out.shape[0]-1, 'image_path'] = tmp_image_path
                        
                downloaded_image_count += 1
        
        #except Exception as e:
        #    print('ERROR***: IMAGE METADATA NOT ADDED TO DATAFRAME')
        #    print('ERROR***: NOTIFY NEERAJ IMMEDIATELY IF THIS HAPPENS')

        if downloaded_image_count >= num_requested:
            break
    
    print(f"Total Downloaded: {downloaded_image_count}/{image_count}")
    driver.quit()
    return df_out, downloaded_image_count


def read(df_input_path, df_output_path, restart):

    if not os.path.exists(df_input_path + '.xlsx'):
        print('DATAFRAME DOES NOT EXISTS:', df_input_path + '.xlsx')
        print('Make sure the pandas dataframe pickle file is downloaded in your directory before continuing.')
        sys.exit()

    df_in = pd.read_excel(df_input_path + '.xlsx')


    column_names = []
    for column in df_in:
        print(column)
        column_names.append(column)

    if restart:
        df_out = pd.DataFrame(columns=column_names)
    else:
        df_out = pd.read_csv(df_output_path + '.csv')
        
    return df_in, df_out

def write(df_out, df_output_path):

    df_out.to_csv(df_output_path + '.csv', index=False)

def main(df_input_path=DF_INPUT_PATH, image_output_path=IMAGE_OUTPUT_PATH, df_output_path=DF_OUTPUT_PATH, res = False):
    
    nodownloads = 0
    prev = False

    if res and os.path.exists(image_output_path) and os.path.exists('save.pkl') and os.path.exists(df_output_path):
        a = input('Are you sure you want to restart?')
        if a == 'True': restart = True
        else: restart = False
    elif res: restart = True
    else: restart = False
    

    if restart:
        if os.path.exists(image_output_path):
            shutil.rmtree(image_output_path)
        print('No cache found in storage')
        done = set()
    else:
        if os.path.exists('save.pkl'):
            with open('save.pkl', 'rb') as s:
                done = pickle.load(s)
                print('LOADING CACHED LIST...')
        else:
            print('No cache found in storage')
            done = set()

    df_in, df_out = read(df_input_path, df_output_path, restart)


    if not restart:
        if not os.path.exists(image_output_path):
            os.makedirs(image_output_path)
    else:
        if os.path.exists(image_output_path):
            shutil.rmtree(image_output_path)
        os.makedirs(image_output_path)

    queries = df_in['search_query'].tolist()

    print('Total Number of Queries', len(queries))


    for ind in range(len(queries)):
        
        if queries[ind] in done:
            continue

        print('Progress:', ind, '/', len(queries))
        num_requests = 100
        df_out, downloaded_image_count = download_images(queries[ind], image_output_path, df_in, df_out, num_requested=num_requests)

        if type(df_out) is str:
            print(df_out)
            break
        if downloaded_image_count > 5:
            done.add(queries[ind])
        if downloaded_image_count == 0:
            prev = True
            nodownloads += 1
        else:
            nodownloads = 0

        if nodownloads == 2:
            print("ERROR: INTERNET CONNECTION")
            print('RERUN WHEN YOU GET INTERNET')
            sys.exit()

        if ind % 10 == 0:
            
            if os.path.exists('save.pkl'):
                os.remove('save.pkl')
            with open('save.pkl', 'wb') as f:
                print('INTERMEDIATE SAVE OF CACHED PRODUCTS, DATAFRAME')
                pickle.dump(done, f)
            write(df_out, df_output_path)

    if os.path.exists('save.pkl'):
        os.remove('save.pkl')
    with open('save.pkl', 'wb') as f:
        print('Saving Cached Product. Saving Dataframe')
        pickle.dump(done, f)

    write(df_out, df_output_path)
    print("DOWNLOAD COMPLETED :))))))))))))))))")

def download_driver():

    if not os.path.exists('./chromedriver'):
        print('CHROMEDRIVE FILE DOES NOT EXIST')
        print('PLEASE COPY CHROMEDRIVE FILE, THEN RUN')
        sys.exit()

    valid_types = ['r', 'c']
    while True:
        restart_input = input('RESTART (r) or Continue (c)?: ')
        if not restart_input in valid_types:
            print('NOT A VALID INPUT. TRY AGAIN')
        else:
            break
    valid_types = ['y', 'n']
    if restart_input == 'r':
        while True:
            check = input("ARE YOU SURE YOU WISH TO RESTART? (y, n): ")
            if not check in valid_types:
                print('NOT A VALID INPUT. TRY AGAIN')
            else:
                break
        if check == 'y':
            restart = True
        else:
            restart = False
    else: restart = False

    main(res=restart)


if __name__=="__main__":
    start = time.time()
    download_driver()
    print(time.time()-start)

