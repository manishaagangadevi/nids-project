import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# --- FIX: Changed SMOTE to RandomOverSampler ---
from imblearn.over_sampling import RandomOverSampler 

# --- Page Setup ---
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")
st.title("🚀 AI Network Intrusion Detection System (NIDS)")

# --- Caching the Model ---
@st.cache_resource
# --- FIX: Renamed parameter to 'use_oversampling' for clarity ---
def train_model(use_oversampling=False): 
    # --- 1. Data Collection & Feature Extraction ---
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
    
    try:
        data = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
    except FileNotFoundError:
        st.error("FATAL ERROR: 'KDDTrain+.txt' not found. Please put the dataset in the same folder.")
        return None, None, None, None, None, None, None

    labels = data['label']
    features_data = data.drop(['label', 'difficulty'], axis=1)
    text_columns = features_data.select_dtypes(['object']).columns
    features_numeric = pd.get_dummies(features_data, columns=text_columns)

    # --- 2. Training the AI Model ---
    X_train, X_test, y_train, y_test = train_test_split(
        features_numeric, labels, test_size=0.2, random_state=42
    )

    # --- FIX: Replaced SMOTE block with RandomOverSampler ---
    if use_oversampling:
        with st.spinner("Applying RandomOverSampler (photocopying rare attacks)..."):
            # Create the RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            # IMPORTANT: We only apply this to the *training* data, never the test data!
            X_train, y_train = ros.fit_resample(X_train, y_train)
            st.toast("Over-sampling applied. Training on new, balanced dataset.")
    # --- End of fix ---

    ai_guard = RandomForestClassifier(n_estimators=50, random_state=42)
    ai_guard.fit(X_train, y_train)
    
    # --- 3. Detection ---
    predictions = ai_guard.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    report = metrics.classification_report(y_test, predictions, zero_division=0, output_dict=True)
    
    # --- Get Feature Importance ---
    importances = ai_guard.feature_importances_
    feature_names = features_numeric.columns
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    return ai_guard, X_test, y_test, predictions, accuracy, report, feature_importance_df

# --- Main Dashboard ---
st.header("Project Overview")
st.markdown("""
This project demonstrates an AI-based NIDS using a **Random Forest Classifier**.
- The AI is trained on the **NSL-KDD dataset**.
- It learns to distinguish between 'normal' network traffic and various types of attacks.
""")

# --- FIX: Updated Checkbox Text ---
st.subheader("Training Options")
use_oversampling_checkbox = st.checkbox("Use Random Over-sampling (to fix rare attacks)")
st.markdown("""
*(Check this box to improve detection for **rare attacks** like 'land', 'spy', etc. 
The AI will "photocopy" rare attacks to get more examples to learn from.)*
""")
# --- End of fix ---


# Create the button
if st.button("Train AI and Run Detection Demo"):
    
    # --- FIX: Get value from the updated checkbox ---
    use_oversampling_value = use_oversampling_checkbox
    
    st.info("Loading data and training AI 'Guard'... Please wait.")
    
    with st.spinner('Training in progress... This may take a few seconds...'):
        # --- FIX: Pass the new value to the function ---
        (ai_guard, X_test, y_test, 
         predictions, accuracy, report, 
         feature_importance_df) = train_model(use_oversampling=use_oversampling_value)

    if ai_guard is None:
        st.error("Training failed. Please check the error message above.")
    else:
        st.success("Training Complete! AI 'Guard' is now active.")
        
        # --- Display Results ---
        st.header(f"Detection Accuracy: {accuracy * 100:.2f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Live Demo Examples")
            st.markdown("Random examples from the test data:")
            
            demo_sample = X_test.sample(15, random_state=42)
            demo_predictions = ai_guard.predict(demo_sample)
            demo_real_answers = y_test.loc[demo_sample.index]
            
            demo_df = pd.DataFrame({
                "AI Prediction": demo_predictions,
                "Real Answer": demo_real_answers
            })
            st.dataframe(demo_df)

        with col2:
            st.subheader("Detailed Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

        # --- Display Confusion Matrix (Full Width) ---
        st.header("Visual Report: Confusion Matrix")
        
        with st.spinner("Generating confusion matrix..."):
            all_labels = sorted(list(set(y_test) | set(predictions)))
            cm = confusion_matrix(y_test, predictions, labels=all_labels)
            
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels, cmap='viridis', ax=ax)
            ax.set_title('AI NIDS Confusion Matrix')
            ax.set_ylabel('True Label (The Real Answer)')
            ax.set_xlabel('Predicted Label (AI\'s Guess)')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            st.pyplot(fig)

        # --- Display Feature Importance ---
        st.header("How the AI 'Thinks': Feature Importance")
        st.markdown("""
        This chart shows which network features the AI "learned" were the
        most important for detecting an attack.
        """)
        
        top_20_features = feature_importance_df.head(20)
        st.bar_chart(top_20_features.set_index('feature')['importance'])