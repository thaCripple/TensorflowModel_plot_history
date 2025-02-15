def plot_history(history, figsize=(12,8)):
  """
  Plots the Accuracy and Loss history collected during model training.
  
  Parameters:
    history: <history.History>
      Object returned by a models' .fit() method
    
    figsize: tuple(width, height); default=(12,8)
      Size of the matplotlib figure to be drawn

  Raises:
    Exception: when the passed history object doesn't contain 
    exactly 4 entries

  Returns: None

  """
  
  # Check if matplotlib has been imported
  if "plt" not in globals():
    import matplotlib.pyplot as plt

  # Extract a dictionary from the history object
  history = history.history
  # Check if history is of correct size
  history_items = list(history.items())
  if len(history_items) != 4:
    raise Exception("History must contain 4 entries")

  n_epochs = range(1, len(history_items[0][1]) + 1)
  fig, axes = plt.subplots(ncols=2, figsize=figsize)

  # Plot Accuracy
  axes[0].plot(n_epochs, history_items[0][1], label=history_items[0][0], color="blue")
  axes[0].plot(n_epochs, history_items[2][1], label=history_items[2][0], color="green")
  axes[0].set(xlabel="Epoch", ylabel="Accuracy")
  axes[0].set_title("Training and Validation Accuracy", fontsize="x-large")
  axes[0].grid(visible=True)
  axes[0].legend()

  # Plot Loss
  axes[1].plot(n_epochs, history_items[1][1], label=history_items[1][0], color="blue")
  axes[1].plot(n_epochs, history_items[3][1], label=history_items[3][0], color="green")
  axes[1].set(xlabel="Epoch", ylabel="Loss")
  axes[1].set_title("Training and Validation Loss", fontsize="x-large")
  axes[1].grid(visible=True)
  axes[1].legend()

  fig.suptitle("Training History", fontsize="xx-large", y=1.02)
  fig.show()
