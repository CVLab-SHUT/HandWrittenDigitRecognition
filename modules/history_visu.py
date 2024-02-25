from IPython.display import HTML, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

display(HTML("""
<style>
#output-body {
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
"""))

def visualize_loss_accuary(history):
    # extract loss values
    epoch = history.epoch
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # styling
    sns.set_style("darkgrid")
    sns.set_style({
    "axes.grid": True,
    "grid.color": "0.8",
    "grid.linewidth": 1
    })

    accuracy_type = np.array(['Training accuracy'] * len(epoch))
    accuracy_type[val_accuracy < val_accuracy] = 'Validation accuracy'

    data = {'epoch': epoch, 'accuracy': train_accuracy}
    df1 = pd.DataFrame(data)
    data = {'epoch': epoch, 'accuracy': val_accuracy}
    df2 = pd.DataFrame(data)

    sns.lineplot(x='epoch', y='accuracy', data=df1, hue=accuracy_type, palette=['darkblue'],
                linewidth=1.5, linestyle='solid', label='Training accuracy')
    sns.lineplot(x='epoch', y='accuracy', data=df2, hue=accuracy_type, palette=['orange'],
                linewidth=1.5, linestyle='solid', label='Validation accuracy')

    # format the y-axis label to display accuracy with 2 decimal places
    plt.gca().yaxis.set_major_formatter('{:.2f}'.format)
    # set the x-axis limits to start from -1


    # display the plot
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #-------------------------------------------------------------------------------
    # styling
    sns.set_style("darkgrid")
    sns.set_style({
    "axes.grid": True,
    "grid.color": "0.8",
    "grid.linewidth": 1
    })

    # plot the results
    loss_type = np.array(['Training loss'] * len(epoch))
    loss_type[val_loss < train_loss] = 'Validation loss'

    sns.lineplot(x=epoch, y=train_loss, hue=loss_type, palette=['darkblue'],
                linewidth=1.5, linestyle='solid', label='Training loss')

    sns.lineplot(x=epoch, y=val_loss, hue=loss_type, palette=['orange'],
                linewidth=1.5, linestyle='solid', label='Validation loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()