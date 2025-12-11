############################
#Frequency per source
############################
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/home/stathis/Desktop/Research/Ianos_published/plots/string_freq_per_source_september_7058.csv")
variables = ["Instagram", "YouTube", "X", "Flickr"]
plt.figure(figsize=(10,6))
plt.plot(df['Instagram'], label='Instagram', linewidth=1, color = 'blue')
plt.plot(df['YouTube'], label='YouTube', linewidth=1, color = 'black')
plt.plot(df['X'], label='X', linewidth=1, color = 'blue')
plt.plot(df['Flickr'], label='Flickr', linewidth=3, color = 'red')
plt.xlabel("Daily interval")
plt.ylabel("Frequencies per source")
plt.legend(loc = 'lower right')#(bbox_to_anchor = (0.83,0.1))
plt.grid(True, alpha=0.3)
plt.show()

