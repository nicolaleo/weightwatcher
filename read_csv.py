


import pandas as pd
import matplotlib.pyplot as plt
#csv_filename='yolov5m_massivo_finetuned.csv'

csv_filenames=['yolov5m_massivo_finetuned.csv','yolov7_massivo_scratch_best.csv','yolov7_massivo_best.csv','yolov7-tiny_massivo_scratch.csv','yolov7-e6e.csv']

for csv_filename in csv_filenames:
    df = pd.read_csv(csv_filename)

    #print(df.to_string()) 

    #df.describe()
    print(df.head())


    print(df.alpha)




    plt.hist(df.alpha,bins=50)
    plt.savefig("alpha_distrib_"+csv_filename.split(".")[0]+".png")
    plt.show() 

