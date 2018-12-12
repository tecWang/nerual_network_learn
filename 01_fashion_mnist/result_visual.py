import keras
import matplotlib.pyplot as plt


def model_paint(model, filepath="./"):
    """
        when rankdir == "TB", vertical layout can be showed
    """
    keras.utils.plot_model(model,
        to_file=filepath + "//model.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB')

def acc_loss_paint(history, filepath="./"):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['acc', "val_acc"], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['loss', "val_loss"], loc='lower right')
    plt.tight_layout()
    plt.savefig(filepath + "//acc_loss.png")
