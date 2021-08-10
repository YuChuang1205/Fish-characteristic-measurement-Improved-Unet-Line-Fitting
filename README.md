
paper:  **Intelligent Measurement of Morphological Characteristics of Fish Using Improved U-Net**.
[链接](https://www.researchgate.net/publication/352390820_Intelligent_Measurement_of_Morphological_Characteristics_of_Fish_Using_Improved_U-Net)

#### If you want to get the public SIRST data set, please go to [[link]()]  
### If you want to reproduce the paper, you need to understand the following information.
**1.Create a "data" folder in the main directory, and its sub-directory structure is as follows:**  

>data
>>mydata
>>>test  
>>>> xx.jpg  
>>>>  ...  
>>>> xx.jpg  

>>>test_results  


>>>train  
>>>>aug  

>>>>image  
>>>>>xx.jpg  
>>>>> ...  
>>>>>xx.jpg 

>>>>label 
>>>>>xx.png  
>>>>> ...  
>>>>>xx.png 


**2.file information**  
  "main.py" is the file for training and testing the model;
  "model.py" is the Improve U-net structure;
  "data.py" is some processing functions
  
  
 **3.After creating the contents of the "data" folder, you can directly run "main.py".** 




