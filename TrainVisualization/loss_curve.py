import numpy as np
from matplotlib import pyplot as plt


np_type = 'Markov'


total_loss = np.load('../numpy/%s/total_loss.npy' % np_type)
val_loss = np.load('../numpy/%s/val_loss.npy' % np_type)

plt.suptitle({'Markov': 'MTF', 'Gramian': 'GAF', 'Recurrence': 'RP'}[np_type]+'-Siamese-VGG16')
plt.subplot(2, 1, 1)
plt.ylim(0, 1)
plt.plot(total_loss, label='train loss', color='dodgerblue')
plt.legend()
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.ylim(0, 1)
plt.plot(val_loss, label='val loss', color='coral')
plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')

plt.savefig('../pic/loss_curve/%s.png' %
            {'Markov': 'MTF', 'Gramian': 'GAF', 'Recurrence': 'RP'}[np_type], transparent=True)
plt.clf()
