q1 <- read.csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-#ianos-20240823_142323.csv")
q2 <- read.csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-#ianos-20240825_053644.csv")
q3 <- read.csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-ζημιές-20240823_225412.csv")
q4 <- read.csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/X/TwiBot-καιρός-20240823_151540.csv")

q.total <- rbind(q1,q2,q3,q4)
qtotal <- unique(q.total)

write.csv(qtotal, "/media/stathis/StathisUSB/final_classification_march_9/text_data/X/X_total.csv")

#create also a sample of 150 rows
data.sample <- sample(rownames(qtotal), 150)
data.train <- qtotal[data.sample, ]
write.csv(data.train, "/home/stathis/Desktop/Research/FLOOD/Sources/X/train.csv")
