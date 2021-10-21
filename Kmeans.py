import math
from typing import Counter

#Tính khoảng cách
def euclid(x,y):
    d = 0
    for i in range(len(x)):
        d += (y[i] - x[i])**2
    return math.sqrt(d)

#List các điểm [x1,x2,x3...]
# x1,x2,x3,x4,x5,x6,x7,x8 = [2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]
x1,x2,x3,x4,x5,x6,x7,x8 =[2, 8], [2, 5], [1, 2], [5,8], [7,3], [6,4], [8,4], [4,7]
list = [x1,x2,x3,x4,x5,x6,x7,x8]

#Chia thành K cụm
k = int(input("Nhập số cụm: "))

#Chọn tâm và gán nhãn tâm
labels = [-1 for i in range(len(list))]
temp = 0
center = []
labels_old = labels.copy()

for i in range(k):
    a = int(input('Chọn vị trí tâm: '))
    center.append(list[a])

while True:
    #Gán nhãn các điểm trong list
    for i in range(len(list)):
        kc = []
        for j in range(k):
            kc.append(euclid(list[i],center[j]))
        labels[i] = kc.index(min(kc))
    print(center)
    print(labels)
    #Tìm lại tâm mới 
    for i in range(k):
        count = 0
        sum_x, sum_y = 0, 0
        for j in range(len(labels)):
            if(i == labels[j]):
                sum_x += list[j][0]
                sum_y += list[j][1]
                count += 1
        center[i] = [sum_x/count,sum_y/count]
    print(labels_old,'\n')
    #Nếu nhán mới và cũ trùng nhau thì dừng thuật toán
    if(labels_old == labels):
        break
    labels_old = labels.copy()
