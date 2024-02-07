import matplotlib.pyplot as plot
with open('winrates.txt', 'r') as f:
    winrates = [float(line.strip()) for line in f]

# Now winrates is a list of your win rates
print(winrates)
generations = range(0, 100)
print("Final win/draw rate : " + str(winrates[99])+"%" )
plot.plot(generations,winrates)
plot.show()
with open('/content/drive/MyDrive/Model/winrates.txt', 'w') as f:
    for rate in winrates:
        f.write(str(rate) + '\n')