import train3
import numpy as np

batch_Size = 64
epoch = 510

path = ['i=0', 'i=1','i=2','i=3','i=4']

best_acc=[]

for i in range(5):
   
    print(path)

    acc = train3.train(batch_Size, epoch, path[i], path[i])
    
    best_acc.append(acc)

print(best_acc)

mean_acc = sum(np.array(best_acc))/len(best_acc)

print(mean_acc)

