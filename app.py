import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler 

# --- Page Setup ---
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")
st.title("🚀 AI Network Intrusion Detection System (NIDS)")

# --- Caching the Model ---
@st.cache_resource
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
        return None, None, None, None, None, None, None, None, None

    labels = data['label']
    features_data = data.drop(['label', 'difficulty'], axis=1)
    text_columns = features_data.select_dtypes(['object']).columns
    features_numeric = pd.get_dummies(features_data, columns=text_columns)
    
    # --- SIMULATOR: Save the list of all column names in order ---
    all_feature_names = features_numeric.columns.tolist()

    # --- 2. Training the AI Model ---
    X_train, X_test, y_train, y_test = train_test_split(
        features_numeric, labels, test_size=0.2, random_state=42
    )

    # --- SIMULATOR: Get a 'normal' row from the test set to use as a template ---
    try:
        normal_index = y_test[y_test == 'normal'].index[0]
        template_row = X_test.loc[normal_index].copy()
    except IndexError:
        st.warning("Could not find a 'normal' row in the test set for the simulator. Using the first row instead.")
        template_row = X_test.iloc[0].copy()


    if use_oversampling:
        with st.spinner("Applying RandomOverSampler (photocopying rare attacks)..."):
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            st.toast("Over-sampling applied. Training on new, balanced dataset.")

    ai_guard = RandomForestClassifier(n_estimators=50, random_state=42)
    ai_guard.fit(X_train, y_train)
    
    # --- 3. Detection ---
    predictions = ai_guard.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    report = metrics.classification_report(y_test, predictions, zero_division=0, output_dict=True)
    
    # --- Get Feature Importance ---
    importances = ai_guard.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # --- SIMULATOR: Return the new objects ---
    return (ai_guard, X_test, y_test, predictions, accuracy, report, 
            feature_importance_df, template_row, all_feature_names)

# --- Main Dashboard ---
st.header("Project Overview")
st.markdown("""
This project demonstrates an AI-based NIDS using a **Random Forest Classifier**.
- The AI is trained on the **NSL-KDD dataset**.
- It learns to distinguish between 'normal' network traffic and various types of attacks.
""")

st.subheader("Training Options")
use_oversampling_checkbox = st.checkbox("Use Random Over-sampling (to fix rare attacks)")
st.markdown("""
*(Check this box to improve detection for **rare attacks** like 'land', 'spy', etc. 
The AI will "photocopy" rare attacks to get more examples to learn from.)*
""")

# Create the button
if st.button("Train AI and Run Detection Demo"):
    
    use_oversampling_value = use_oversampling_checkbox
    
    st.info("Loading data and training AI 'Guard'... Please wait.")
    
    with st.spinner('Training in progress... This may take a few seconds...'):
        # --- SIMULATOR: Unpack all the new objects ---
        (ai_guard, X_test, y_test, 
         predictions, accuracy, report, 
         feature_importance_df, template_row,
         all_feature_names) = train_model(use_oversampling=use_oversampling_value)

        # --- SIMULATOR: Save the trained model and template to the session ---
        # This keeps them loaded even when we interact with sliders
        st.session_state.ai_guard = ai_guard
        st.session_state.template_row = template_row
        st.session_state.all_feature_names = all_feature_names
        st.session_state.y_test = y_test # Store these for the demo later
        st.session_state.predictions = predictions
        st.session_state.report = report
        st.session_state.accuracy = accuracy
        st.session_state.feature_importance_df = feature_importance_df
        st.session_state.X_test = X_test
        # --- End Simulator ---


# --- Check if the model is trained and in memory ---
if 'ai_guard' in st.session_state:
    
    # Retrieve all items from the session state
    ai_guard = st.session_state.ai_guard
    accuracy = st.session_state.accuracy
    report = st.session_state.report
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    predictions = st.session_state.predictions
    feature_importance_df = st.session_state.feature_importance_df

    st.success("Training Complete! AI 'Guard' is now active.")
    
    # --- SIMULATOR: New Section ---
    st.header("Real-Time Attack Simulator 🕹️")
    st.markdown("Use the controls below to simulate a network connection and see if the AI flags it as an attack.")
    
    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.subheader("Simulate Traffic Features")
        
        # Get the template row (a 'normal' connection)
        sim_template = st.session_state.template_row.copy()
        
        # Create sliders and inputs based on important features
        # We use the 'normal' row's values as the default
        
        # --- These are key features for 'neptune' (DoS) attacks ---
        st.write("**Denial of Service (DoS) Features**")
        sim_count = st.slider("Connection Count (count)", 0, 511, int(sim_template['count']))
        sim_srv_count = st.slider("Service Connection Count (srv_count)", 0, 511, int(sim_template['srv_count']))
        sim_dst_host_srv_count = st.slider("Dest. Host Service Count (dst_host_srv_count)", 0, 255, int(sim_template['dst_host_srv_count']))

        # --- These are key features for 'smurf' (Flooding) attacks ---
        st.write("**Flooding / Probing Features**")
        sim_src_bytes = st.number_input("Source Bytes (src_bytes)", 0, 1500000000, int(sim_template['src_bytes']), step=1000)
        sim_dst_bytes = st.number_input("Destination Bytes (dst_bytes)", 0, 1500000000, int(sim_template['dst_bytes']), step=1000)
        
        # We need to know which 'service' column to set to 1
        # This is a simple way to simulate a common service
        sim_service_http = st.checkbox("Service: HTTP (service_http)", value=bool(sim_template['service_http']))

    
    with sim_col2:
        st.subheader("AI Prediction")
        
        if st.button("Predict Simulated Traffic"):
            # 1. Update the template row with new values
            sim_template['count'] = sim_count
            sim_template['srv_count'] = sim_srv_count
            sim_template['dst_host_srv_count'] = sim_dst_host_srv_count
            sim_template['src_bytes'] = sim_src_bytes
            sim_template['dst_bytes'] = sim_dst_bytes

            # 2. Handle the 'service_http' feature (one-hot encoded)
            # This is a bit of a trick: we set ALL service columns to 0, then set http to 1 if checked.
            service_cols = [col for col in st.session_state.all_feature_names if col.startswith('service_')]
            for col in service_cols:
                sim_template[col] = 0 # Reset all service columns
            
            if sim_service_http:
                if 'service_http' in sim_template:
                    sim_template['service_http'] = 1
                else:
                    st.warning("Feature 'service_http' not found in model columns.")

            # 3. Put the single row into a DataFrame, ensuring column order
            prediction_data = pd.DataFrame([sim_template], columns=st.session_state.all_feature_names)
            
            # 4. Run prediction
            prediction = ai_guard.predict(prediction_data)
            prediction_proba = ai_guard.predict_proba(prediction_data)

            # 5. Display result
            result = prediction[0]
            proba = prediction_proba.max() * 100

            if result == 'normal':
                st.success(f"✅ Prediction: NORMAL")
                st.write(f"Confidence: {proba:.2f}%")
                st.write("The AI believes this traffic is safe.")
            else:
                st.error(f"🚨 Prediction: ATTACK (Type: {result})")
                st.write(f"Confidence: {proba:.2f}%")
                st.write("The AI has flagged this traffic as malicious!")

    # --- End Simulator Section ---

    st.divider() # Add a line to separate sections
    
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