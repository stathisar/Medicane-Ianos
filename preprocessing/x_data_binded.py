import pandas as pd
X1 = pd.read_csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-#ianos-20240823_142323.csv")
X2 = pd.read_csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-#ianos-20240825_053644.csv")
X3 = pd.read_csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-ζημιές-20240823_225412.csv")
X4 = pd.read_csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-καιρός-20240823_151540.csv")

X_total = pd.concat([X1,X2,X3,X4], ignore_index=True)
X_total_complete =  X_total.drop_duplicates()

#export as csv
X_total_complete.to_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/X/x_data_binded.csv')
