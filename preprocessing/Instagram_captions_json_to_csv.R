require(jsonlite)
data = jsonlite::fromJSON("/media/stathis/StathisUSB/Research/Ianos - Deucalion/IANOS DATA/ianos/ianos.json")
class(data)
setwd("/media/stathis/StathisUSB/")

#data$GraphImages$location$name[[1]]
df = as.data.frame(matrix(nrow = 4656, ncol = 5))
names(df) = c("timestamp", "text", "id", "location.name","shortcode")
#we need: id, text, timestamp,location
for (i in 1:nrow(df)){
  timestamp <- as.numeric(data$GraphImages$taken_at_timestamp[[i]])
  if (length(data$GraphImages$edge_media_to_caption$edges[[i]]) == 0){
    text <- NA
  }else{
    text <- data$GraphImages$edge_media_to_caption$edges[[i]]
  }
  id <- data$GraphImages$id[[i]]
  location.name <- data$GraphImages$location$name[[i]]
  shortcode <- data$GraphImages$shortcode[[i]]
  df[i, ] <- cbind(timestamp, text, id, location.name, shortcode)
  rm(timestamp, text, id, location.name, shortcode)
}

df <- as.data.frame(df)
df$text <- as.character(df$text)
class(df$id)
class(df$location.name)
class(df$shortcode)
write.csv(df, "/media/stathis/StathisUSB/final_classification_march_9/text_data/Instagram_captions/Instagram_captions.csv")
