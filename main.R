library(networkD3)
library(igraph)
library(plyr)

dt <- as.matrix(read.csv("subjects - Sheet1.csv", check.names = FALSE, row.names = 1))

# To avoid problems on next steps we set NA as 0
dt[is.na(dt)] <- 0

# Map subject labels to names
gp <- dt[,1]
gp <- mapvalues(gp, 
                from = c(1, 2, 3, 4, 5, 6, 7), 
                to = c("Mathematics", "Statistics", "Optimization", 
                       "Machine Learning", "Information Theory", 
                       "Computer Science", "Artificial Intelligence"))

# Drop subject label column
dt <- dt[,-1]

# Create graph from adjacency matrix
g <- graph.adjacency(dt, mode = "undirected")

# Convert to object suitable for networkD3
dt_g <- igraph_to_networkD3(g, group = gp)

# Generate graph
forceNetwork(Links = dt_g$links, Nodes = dt_g$nodes,
             Source = "source", Target = "target", NodeID = "name",
             Group = "group", opacity = 1, zoom = TRUE, legend = T, 
             opacityNoHover = 0.5)

saveNetwork(net, "index.html", selfcontained = T)
