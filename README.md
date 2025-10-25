# AI-Based Network Intrusion Detection System (NIDS)

This is a project for a computer networks class.

It uses a Python script (`detect.py`) and the **NSL-KDD dataset** to build an AI model that can detect network attacks.

### What it Does
1.  **Loads** the `KDDTrain+.txt` dataset.
2.  **Cleans** the data and converts text features into numbers.
3.  **Trains** a "Random Forest" (a type of AI) to learn the patterns of "normal" vs. "attack" traffic.
4.  **Tests** the AI on data it has never seen before and prints a report showing its accuracy.

### How to Run It

This project is an interactive web app built with Streamlit.

1.  Make sure you have all the libraries installed (pandas, scikit-learn, streamlit, etc.).
2.  Open your terminal in the project folder.
3.  Run the following command:

  ```bash
  streamlit run app.pypy`
4. A new tab will automatically open in your web browser with the dashboard.