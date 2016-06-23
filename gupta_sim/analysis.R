library('Hmisc')

out <- read.csv("~/Documents/stage/gupta_sim/myfile.csv", sep="")


colors = ifelse(out$replay == 'True', 'red', 'blue')
plot(out$step[1100:1500], out$cell[1100:1500], col=colors[1100:1500])

dist_goal = pmin(sqrt((out$x-200)^2 + (out$y - 900)^2), sqrt((out$x-1800)^2 + (out$y - 900)^2))
sign = ifelse((out$y >= 200 & out$x > 400 & out$x < 1600) | out$y > 800, -1, 1)

dist_goal = sign * dist_goal
out = cbind (out, dist_goal)

ordered = order(dist_goal)
out = cbind(out, ordered)

to_look = c()
replay = FALSE
for (i in 1:length(out$replay)) {
  if (replay == FALSE && out$replay[i] == "True") {
      to_look = c(to_look, i)
      replay = TRUE
  } else if (replay == TRUE && out$replay[i] == "False") {
    replay = FALSE
  }
}

lmar = 100
rmar = 250

for (look in to_look) {
  em = embed(out$dist_goal[(look-lmar):(look+rmar)],3) #10 représente la taille de ta fenêtre
  means = apply(em,1,mean)
  mins = apply(em, 1, min)
  maxs = apply(em, 1, max)
  plot(means, col=colors[seq((look-lmar),(look+rmar))], ylim=c(-1000, 1000), main="mean")
  errbar(x=1:length(means), y=means, yplus=maxs, yminus=mins, add=T, type="n")
  lines(means)
  abline(0, 0, col="green")
  plot(out$step[(look-lmar):(look+rmar)], out$dist_goal[(look-lmar):(look+rmar)],
       type="l",
       main=paste("Replay dans", look-lmar, '-', look+rmar),
       ylab="Distance au but",
       xlab="Pas de temps")
  points(out$step[(look-lmar):(look+rmar)], out$dist_goal[(look-lmar):(look+rmar)], col=colors[(look-lmar):(look+rmar)])
  abline(0, 0, col="green")
}
