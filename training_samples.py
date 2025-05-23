# Malota SP 41370015
import torch # Import the PyTorch library, fundamental for neural network operations.
import torch.nn as nn # Import the neural network module from PyTorch.
import torch.optim as optim # Import the optimization module from PyTorch.
from torch.utils.data import DataLoader # Import DataLoader for efficient batching of datasets.
from preprocessing_images import IntelImageDataset, get_transforms # Import custom dataset and transformation functions from a local module.


train_dir = "C:/Users/malot/OneDrive/Documents/AI/intel-image-ml-project/data/seg_train" # Define the path to the training dataset directory.
test_dir = "C:/Users/malot/OneDrive/Documents/AI/intel-image-ml-project/data/seg_test" # Define the path to the testing dataset directory.
try:
    train_dataset = IntelImageDataset(train_dir, transform=get_transforms(train=True)) # Initialize the training dataset with training transformations.
    test_dataset = IntelImageDataset(test_dir, transform=get_transforms(train=False)) # Initialize the testing dataset with testing transformations.
except Exception as e:
    print(f"Error loading datasets: {e}") # Print an error message if dataset loading fails.
    raise # Re-raise the exception.

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Create a DataLoader for the training dataset, enabling batching and shuffling.
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Create a DataLoader for the testing dataset, enabling batching but no shuffling.
class SimpleCNN(nn.Module): # Define a simple Convolutional Neural Network (CNN) class, inheriting from nn.Module.
    def __init__(self, num_classes): # Constructor for the SimpleCNN, taking the number of output classes as an argument.
        super(SimpleCNN, self).__init__() # Call the constructor of the parent class (nn.Module).
        self.features = nn.Sequential( # Define the convolutional feature extraction layers.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # First convolutional layer: 3 input channels (RGB), 32 output channels, 3x3 kernel, 1-pixel padding.
            nn.ReLU(), # ReLU activation function.
            nn.MaxPool2d(2, 2),  # Max pooling layer: 2x2 window, stride 2.

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Second convolutional layer.
            nn.ReLU(), # ReLU activation function.
            nn.MaxPool2d(2, 2),  # Max pooling layer.

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Third convolutional layer.
            nn.ReLU(), # ReLU activation function.
            nn.MaxPool2d(2, 2),  # Max pooling layer.
        )
        self.classifier = nn.Sequential( # Define the classification layers.
            nn.Flatten(), # Flatten the output from the convolutional layers into a 1D vector.
            nn.Linear(128 * 18 * 18, 256), # First fully connected (linear) layer. Input size is calculated based on the output of the last max pooling layer.
            nn.ReLU(), # ReLU activation function.
            nn.Dropout(0.5), # Dropout layer with a dropout probability of 0.5 to prevent overfitting.
            nn.Linear(256, num_classes) # Second fully connected (linear) layer, outputting scores for each class.
        )

    def forward(self, x): # Define the forward pass of the network.
        x = self.features(x) # Pass the input through the feature extraction layers.
        x = self.classifier(x) # Pass the result through the classification layers.
        return x # Return the final output.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set the device to GPU if available, otherwise CPU.
num_classes = len(train_dataset.class_names) # Get the number of classes from the training dataset.
model = SimpleCNN(num_classes=num_classes).to(device) # Initialize the SimpleCNN model and move it to the specified device.
criterion = nn.CrossEntropyLoss() # Define the loss function as Cross-Entropy Loss, suitable for multi-class classification.
optimizer = optim.Adam(model.parameters(), lr=0.001) # Define the optimizer as Adam, with a learning rate of 0.001.
def train_model(num_epochs=10): # Define the training function.
    for epoch in range(num_epochs): # Loop through each epoch.
        model.train() # Set the model to training mode.
        running_loss = 0.0 # Initialize running loss for the current epoch.
        correct = 0 # Initialize count of correctly predicted samples.
        total = 0 # Initialize total number of samples processed.

        for images, labels in train_loader: # Iterate through batches in the training data loader.
            try:
                images, labels = images.to(device), labels.to(device) # Move images and labels to the specified device.
                outputs = model(images) # Perform a forward pass to get model outputs.
                loss = criterion(outputs, labels) # Calculate the loss.
                optimizer.zero_grad() # Zero the gradients before backpropagation.
                loss.backward() # Perform backpropagation to calculate gradients.
                optimizer.step() # Update model parameters.

                running_loss += loss.item() # Add the current batch's loss to running loss.
                _, predicted = torch.max(outputs, 1) # Get the predicted class with the highest probability.
                total += labels.size(0) # Update total samples.
                correct += (predicted == labels).sum().item() # Update correctly predicted samples.
            except Exception as e:
                print(f"Error during training batch: {e}") # Print an error if an issue occurs during a batch.

        accuracy = 100 * correct / total if total > 0 else 0 # Calculate training accuracy for the epoch.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%") # Print epoch-wise training statistics.
def evaluate_model(): # Define the model evaluation function.
    model.eval() # Set the model to evaluation mode.
    correct = 0 # Initialize correct predictions count.
    total = 0 # Initialize total samples count.
    try:
        with torch.no_grad(): # Disable gradient calculation for evaluation.
            for images, labels in test_loader: # Iterate through batches in the testing data loader.
                images, labels = images.to(device), labels.to(device) # Move images and labels to the specified device.
                outputs = model(images) # Perform a forward pass.
                _, predicted = torch.max(outputs.data, 1) # Get predicted classes.
                total += labels.size(0) # Update total samples.
                correct += (predicted == labels).sum().item() # Update correct predictions.
        print(f"Test Accuracy: {100 * correct / total:.2f}%") # Print the overall test accuracy.
    except Exception as e:
        print(f"Error during evaluation: {e}") # Print an error if an issue occurs during evaluation.
def show_per_class_results(): # Define a function to show per-class accuracy and error rates.
    model.eval() # Set the model to evaluation mode.
    class_names = train_dataset.class_names # Get the list of class names.
    class_to_idx = {name: idx for idx, name in enumerate(class_names)} # Create a mapping from class name to index.
    idx_to_class = {idx: name for name, idx in class_to_idx.items()} # Create a mapping from index to class name.
    results = {cls: {'correct': 0, 'incorrect': 0, 'total': 0} for cls in class_names} # Initialize a dictionary to store per-class results.

    with torch.no_grad(): # Disable gradient calculation.
        for images, labels in test_loader: # Iterate through test data.
            images, labels = images.to(device), labels.to(device) # Move data to device.
            outputs = model(images) # Get model outputs.
            _, predicted = torch.max(outputs, 1) # Get predicted classes.
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()): # Iterate through true labels and predictions.
                class_name = idx_to_class[label] # Get the class name for the current label.
                results[class_name]['total'] += 1 # Increment total count for the class.
                if label == pred: # Check if prediction is correct.
                    results[class_name]['correct'] += 1 # Increment correct count.
                else:
                    results[class_name]['incorrect'] += 1 # Increment incorrect count.

    print("\nPer-class accuracy and error rates on test set:") # Print header for per-class results.
    for cls in class_names: # Iterate through each class.
        total = results[cls]['total'] # Get total samples for the class.
        correct = results[cls]['correct'] # Get correct predictions for the class.
        incorrect = results[cls]['incorrect'] # Get incorrect predictions for the class.
        correct_pct = 100 * correct / total if total > 0 else 0 # Calculate correct percentage.
        incorrect_pct = 100 * incorrect / total if total > 0 else 0 # Calculate incorrect percentage.
        print(f"Class '{cls}': {correct} correct ({correct_pct:.2f}%), {incorrect} incorrect ({incorrect_pct:.2f}%), {total} total") # Print per-class statistics.
from sklearn.metrics import ( # Import various metrics from scikit-learn for comprehensive evaluation.
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt # Import matplotlib for plotting.
import numpy as np # Import numpy for numerical operations.

def evaluate_all_metrics(model, data_loader, device, num_classes, class_names): # Define a function to evaluate all relevant classification metrics.
    model.eval() # Set the model to evaluation mode.
    all_labels = [] # List to store all true labels.
    all_preds = [] # List to store all predicted labels.
    all_probs = [] # List to store all predicted probabilities.

    with torch.no_grad(): # Disable gradient calculation.
        for images, labels in data_loader: # Iterate through data.
            images = images.to(device) # Move images to device.
            labels = labels.to(device) # Move labels to device.
            outputs = model(images) # Get model outputs.
            probs = torch.softmax(outputs, dim=1) # Calculate probabilities using softmax.
            _, preds = torch.max(outputs, 1) # Get predicted classes.
            all_labels.extend(labels.cpu().numpy()) # Add true labels to list.
            all_preds.extend(preds.cpu().numpy()) # Add predicted labels to list.
            all_probs.extend(probs.cpu().numpy()) # Add predicted probabilities to list.

    all_labels = np.array(all_labels) # Convert lists to numpy arrays.
    all_preds = np.array(all_preds) # Convert lists to numpy arrays.
    all_probs = np.array(all_probs) # Convert lists to numpy arrays.
    acc = accuracy_score(all_labels, all_preds) # Calculate overall accuracy.
    print(f"Accuracy: {acc:.4f}") # Print accuracy.
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) # Calculate macro-averaged precision.
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) # Calculate macro-averaged recall.
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) # Calculate macro-averaged F1-score.
    print(f"Precision (macro): {precision:.4f}") # Print precision.
    print(f"Recall (macro): {recall:.4f}") # Print recall.
    print(f"F1-Score (macro): {f1:.4f}") # Print F1-score.
    cm = confusion_matrix(all_labels, all_preds) # Calculate the confusion matrix.
    print("Confusion Matrix:") # Print confusion matrix header.
    print(cm) # Print the confusion matrix.
    if num_classes == 2: # Check if it's a binary classification problem.
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1]) # Calculate False Positive Rate (FPR) and True Positive Rate (TPR) for ROC curve.
        roc_auc = auc(fpr, tpr) # Calculate Area Under the Curve (AUC) for ROC.
        plt.figure() # Create a new plot figure.
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})') # Plot ROC curve.
        plt.plot([0, 1], [0, 1], 'k--') # Plot the random classifier line.
        plt.xlabel('False Positive Rate') # Set x-axis label.
        plt.ylabel('True Positive Rate') # Set y-axis label.
        plt.title('ROC Curve') # Set plot title.
        plt.legend(loc="lower right") # Show legend.
        plt.show() # Display the plot.
        print(f"AUC: {roc_auc:.4f}") # Print AUC.
    else: # For multi-class classification.
        from sklearn.preprocessing import label_binarize # Import label_binarize for one-hot encoding labels.
        y_test_bin = label_binarize(all_labels, classes=range(num_classes)) # Binarize true labels.
        roc_auc = roc_auc_score(y_test_bin, all_probs, average="macro", multi_class="ovr") # Calculate macro-averaged ROC AUC for multi-class.
        print(f"Multiclass AUC (macro): {roc_auc:.4f}") # Print multi-class AUC.
    if num_classes == 2: # Check if it's a binary classification problem.
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs[:, 1]) # Calculate precision-recall curve.
        pr_auc = auc(recall_curve, precision_curve) # Calculate AUC for precision-recall curve.
        plt.figure() # Create a new plot figure.
        plt.plot(recall_curve, precision_curve, label=f'PR curve (area = {pr_auc:.2f})') # Plot PR curve.
        plt.xlabel('Recall') # Set x-axis label.
        plt.ylabel('Precision') # Set y-axis label.
        plt.title('Precision-Recall Curve') # Set plot title.
        plt.legend(loc="lower left") # Show legend.
        plt.show() # Display the plot.
        print(f"PR AUC: {pr_auc:.4f}") # Print PR AUC.
    else: # For multi-class classification.
        from sklearn.preprocessing import label_binarize # Import label_binarize.
        y_test_bin = label_binarize(all_labels, classes=range(num_classes)) # Binarize true labels.
        pr_auc = 0 # Initialize PR AUC.
        for i in range(num_classes): # Iterate for each class.
            precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i]) # Calculate PR curve for each class (one-vs-rest).
            pr_auc += auc(recall_curve, precision_curve) # Add AUC for current class.
        pr_auc /= num_classes # Average PR AUC across classes.
        print(f"Multiclass PR AUC (macro): {pr_auc:.4f}") # Print multi-class PR AUC.
    total_params = sum(p.numel() for p in model.parameters()) # Calculate the total number of trainable parameters in the model.
    print(f"Model parameters: {total_params:,}") # Print the total number of model parameters.
if __name__ == "__main__": # Check if the script is being run directly.
    try:
        train_model(num_epochs=10) # Call the training function with 10 epochs.
        evaluate_model() # Call the evaluation function to get overall accuracy.
        show_per_class_results() # Call the function to display per-class results.
        evaluate_all_metrics(model, test_loader, device, num_classes, train_dataset.class_names) # Call the function to evaluate and display all metrics.
    except Exception as e:
        print(f"Error in main execution: {e}") # Print an error if an issue occurs during main execution.