p4 = (177.5, 9.51)
p7 = (177.27, 150.03)
p11 = (230.21, 9.35)
p14 = (230.09, 150.71)

#for p5 and p6
diff_x = p14[0] - p11[0]
diff_y = p14[1] - p11[1]
for i in range(12, 14):
    x = p11[0] + ((i - 11) / 3) * diff_x
    y = p11[1] + ((i - 11) / 3) * diff_y
    print("P{}: X = {}, Y = {}".format(i, x ,y))