
import matplotlib.pyplot as plt

def show_info(info, basenet,name):
    train_loss, train_acc, val_loss, val_acc = info
    plt.figure(figsize=(12, 5))
    # plt.plot(train_loss, label='train_loss')
    plt.plot(train_acc, label='train_acc')
    # plt.plot(val_loss, label='val_loss')
    plt.plot(val_acc, label='val_acc')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.title('Accuracy curves for %s' % basenet)
    plt.savefig(name+"accuracy" +'.png')
    # plt.show()

def show_info_loss(info, basenet,name):
    train_loss, train_acc, val_loss, val_acc = info
    plt.figure(figsize=(12, 5))
    plt.plot(train_loss, label='train_loss')
    # plt.plot(train_acc, label='train_acc')
    plt.plot(val_loss, label='val_loss')
    # plt.plot(val_acc, label='val_acc')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.title('Loss curves for %s' % basenet)
    plt.savefig(name+"loss"+'.png')
    # plt.show()
