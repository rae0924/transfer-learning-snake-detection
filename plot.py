import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

file_dir = os.path.dirname(__file__)
path = os.path.join(file_dir, 'save/train_history.pkl')
history = pickle.load(open(path,'rb'))
train_losses = history[0]
train_accuracies = [d*100 for d in history[1]]
val_losses = history[2]
val_accuracies = [d*100 for d in history[3]]
print(train_accuracies[-1], val_accuracies[-1] )
epochs = range(len(history[0]))

fig1, ax1 = plt.subplots()
ax1.set_title('epochs vs. bce_loss')
ax1.scatter(epochs, train_losses, label='train_loss', c='b')
ax1.plot(epochs, val_losses, label='val_loss', c='r')
ax1.set_xlabel('epochs')
ax1.set_ylabel('bce_loss')
ax1.grid()
ax1.set_axisbelow(True)
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.set_title('epochs vs. accuracy')
ax2.scatter(epochs, train_accuracies, label='train_accuracy', c='b')
ax2.plot(epochs, val_accuracies, label='val_accuracy', c='r')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.grid()
ax2.set_axisbelow(True)
ax2.legend(loc='center right')

fig1.savefig(os.path.join(file_dir, 'save/loss_graph.png'))
fig2.savefig(os.path.join(file_dir, 'save/accuracy_graph.png'))


