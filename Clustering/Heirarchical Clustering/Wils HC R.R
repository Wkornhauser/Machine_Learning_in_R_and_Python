# Heirarchical Clustering

# Importing the mall dataset
dataset = read.csv("Mall_Customers.csv")
X = dataset [4:5]

# Building the dendrogram to find optimal number of clusters
dendrogram = hclust(dist(X, method = "euclidean"), method = "ward.D")
plot(dendrogram,
     main = paste("Dendrogram"),
     xlab = 'Customers',
     ylab = 'Euclidean Distances')

# Find hierarchial clustering to the mall dataset
hc = hclust(dist(X, method = "euclidean"), method = "ward.D")
y_hc = cutree(hc,
              k = 5)

# Visualising the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters of Clients"),
         xlab = "Annual Income",
         ylab = "Spending Score")