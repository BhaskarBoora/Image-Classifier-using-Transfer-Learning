from bs4 import BeautifulSoup
import re
import urllib.request
import os

def get_soup(url,header):
  return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url,headers=header)),'lxml')


def get_image(query):

    os.chdir(os.path.dirname(__file__))
    DIR=os.getcwd()

    image_type = "Action"
    query= query.split()
    query='+'.join(query)
    
    url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch&imgsz=xlarge|xxlarge|huge/5"
    
    header = {'User-Agent': 'Mozilla/5.0'} 
    soup = get_soup(url,header)

    images = [a['src'] for a in soup.find_all("img", {"src": re.compile("gstatic.com")})]
    #cntr=161
    for img in images:
        raw_img = urllib.request.urlopen(img).read()
        DIR="E:\\ML_Project\\Garbage\\Toxic\\"
        cntr = len([i for i in os.listdir(DIR) if image_type in i]) + 1
        f = open(DIR + image_type + "_"+ str(cntr)+".jpg", 'wb')
        f.write(raw_img)
        f.close()
        #cntr=cntr+1


if __name__=='__main__':
    search=['kitchen','organic','vegetable','flower','leaves','fruit']
    a=['paper','glass','metals','plastics','recyclable']
    b=['toxic','medicine','paints','chemicals','bulbs','spray cans','fertilizer','pesticides','shoe polish']
    c=['electronic','phones','batteries','TVs','laptops']
    d=['cars']
    for i in d:
        get_image(i+' waste')
    