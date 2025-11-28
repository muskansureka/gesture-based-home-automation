import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("gesture_results.csv")

# Extract true and predicted labels
y_true = df["gesture_actual"]
y_pred = df["gesture_detected"]

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred) * 100
report = classification_report(y_true, y_pred, output_dict=True)
avg_time = df["detection_time"].mean()

# Save text report (ASCII only)
with open("analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("===== Gesture Recognition Accuracy Report =====\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write(f"Average Detection Time: {avg_time:.3f} seconds\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(classification_report(y_true, y_pred))

print("Analysis complete! Results saved to analysis_report.txt")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = sorted(df["gesture_actual"].unique())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Gesture")
plt.ylabel("Actual Gesture")
plt.title("Confusion Matrix for Gesture Recognition")
plt.savefig("confusion_matrix.png")
plt.close()

# Gesture-wise accuracy
gesture_acc = {}
for gesture in labels:
    correct = df[(df["gesture_actual"] == gesture) &
                 (df["gesture_detected"] == gesture)]
    total = df[df["gesture_actual"] == gesture]
    gesture_acc[gesture] = (len(correct) / len(total)) * 100

plt.figure(figsize=(6, 4))
plt.bar(gesture_acc.keys(), gesture_acc.values(), color='green')
plt.xlabel("Gesture")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Gesture")
plt.ylim(0, 100)
plt.savefig("gesture_accuracy.png")
plt.close()

print("Confusion matrix and gesture accuracy chart saved successfully!")