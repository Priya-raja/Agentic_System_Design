import os
import tempfile
import time
# from langchain_community.document_loaders import (TextLoader)

from dotenv import load_dotenv

load_dotenv()


with open('tempfile.txt','w') as file:
        file.write("Two roads diverged in a yellow wood,\n")
        file.write("Why don't scientists trust atoms? Because they make up everything!,\n")
        file_path = file.name

with open('tempfile.txt','r') as tempfile:     
 
        content = tempfile.readlines()  
        # documents = loader.load() 
      

        for doc in content:
            
            print(doc.strip())
            time.sleep(1)
            

 
           


