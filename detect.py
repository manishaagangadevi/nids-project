import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time # We'll use this to time the training

print("--- AI NIDS Project ---")
print("Libraries imported.")

# --- 1. Data Collection & Feature Extraction ---
# We need to give names to all the columns.
# The NSL-KDD dataset website specifies these 42 columns.
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
    "num_compromised", "root_shell", "su_attempted", "num_root", 
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

print("Loading dataset file...")
# Try to load the data file from your folder
try:
    # Read the .txt file using pandas
    data = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("File 'KDDTrain+.txt' not found.")
    print("Please make sure the file is in the same folder as your detect.py script.")
    print("-------------")
    exit()

# --- *** CORRECTED SECTION *** ---

# 1. First, separate the labels (our "answer") from the rest of the data.
labels = data['label']

# 2. Now, create the features DataFrame by dropping the columns we don't need
#    ('label' because it's our answer, 'difficulty' because it's not a feature)
features_data = data.drop(['label', 'difficulty'], axis=1)

print("Converting text data to numbers (feature extraction)...")
# 3. Find all text columns *only in the features data*
text_columns = features_data.select_dtypes(['object']).columns

# 4. Convert *only the feature data* to numbers
features_numeric = pd.get_dummies(features_data, columns=text_columns)

# --- *** END OF CORRECTION *** ---


print(f"Total connections in dataset: {len(labels)}")
print(f"Normal connections: {labels.value_counts()['normal']}")
# Calculate total attacks by subtracting 'normal' from the total count
print(f"Attack connections: {len(labels) - labels.value_counts()['normal']}")

# --- 2. Training the AI Model ---
# We split our data: 80% for training the AI, 20% for testing it
# We use 'features_numeric' (our X) and 'labels' (our y)
X_train, X_test, y_train, y_test = train_test_split(
    features_numeric, labels, test_size=0.2, random_state=42
)

print("\nAI 'Guard' is starting to train...")
start_time = time.time()

# Create the AI model. We'll use Random Forest, as in your plan.
# This is like hiring a team of 50 "smart guards" (n_estimators=50)
ai_guard = RandomForestClassifier(n_estimators=50, random_state=42)

# The AI "learns" the patterns from the 80% training data
ai_guard.fit(X_train, y_train)

end_time = time.time()
print(f"Training complete! It took {end_time - start_time:.2f} seconds.")

# --- 3. Detection (Demo) ---
print("\n--- AI NIDS DEMO ---")
print("AI 'Guard' is now monitoring the 20% test traffic...")
# Use the trained AI to make predictions on the 20% of data it has *never* seen
predictions = ai_guard.predict(X_test)

# Compare the AI's guesses (predictions) to the real answers (y_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print(f"\nDetection Accuracy: {accuracy * 100:.2f}%")
print("This shows how often the AI was correct.")

# Show a detailed "classification report"
# This breaks down how well it did on 'normal' vs. 'attack'
print("\n--- Detailed Report ---")
# We set zero_division=0 to prevent warnings if a class has 0 samples in the test set
print(metrics.classification_report(y_test, predictions, zero_division=0))

# Show 15 example predictions for your demo
print("\n--- Live Demo Examples ---")
print("Showing 15 random examples from the test data:")
print("-----------------------------------------------")
print("   AI Prediction   |   Real Answer")
print("-----------------------------------------------")

# Get 15 random samples to display
demo_sample = X_test.sample(15, random_state=42)
demo_predictions = ai_guard.predict(demo_sample)
demo_real_answers = labels.loc[demo_sample.index]

for pred, real in zip(demo_predictions, demo_real_answers):
    # This just makes the output line up nicely
    print(f"   {pred:<15} |   {real}") 
print("-----------------------------------------------")