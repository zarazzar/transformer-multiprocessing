import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Read CSV
print("Reading CSV file...")
df = pd.read_csv("https://raw.githubusercontent.com/zarazzar/sentimen-analisis-boikot-produk-sirewel/main/data_15k.csv")
df = df[["id", "author", "description", "clean"]]

# Set pretrained model
pretrained = "mdhugol/indonesia-bert-sentiment-classification"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

# Create sentiment classifier using huggingface pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Ensure 'clean' column is of type str and handle NaNs
df['clean'] = df['clean'].astype(str).fillna('')

# Set Label Index
label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

def get_sentiment(text):
    sentiment = sentiment_analysis(text)[0]['label']
    return label_index[sentiment]

def plot_sentiment_distribution(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='sentiment', hue='sentiment', data=df, order=['positive', 'neutral', 'negative'], 
                       palette={'positive': 'blue', 'neutral': 'gray', 'negative': 'red'}, legend=False)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.show()

def main(execution_time, num_cores):
    root = tk.Tk()
    root.title("Sentiment Analysis Results")

    # Create frame for plot
    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create plot in tkinter
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='sentiment', hue='sentiment', data=df, order=['positive', 'neutral', 'negative'], 
                  palette={'positive': 'blue', 'neutral': 'gray', 'negative': 'red'}, ax=ax, legend=False)
    ax.set_title("Analisis Sentimen")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")

    # Display plot in tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create frame for labels
    label_frame = tk.Frame(root)
    label_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

    # Display execution time and number of logical processors
    time_label = tk.Label(label_frame, text=f"Execution time: {execution_time:.2f} seconds", font=("Helvetica", 12))
    time_label.pack(side=tk.LEFT, padx=10, pady=10)
    cores_label = tk.Label(label_frame, text=f"Number of logical processors: {num_cores}", font=("Helvetica", 12))
    cores_label.pack(side=tk.LEFT, padx=10, pady=10)

    # Display GUI
    root.mainloop()

if __name__ == '__main__':
    # Measure time before processing
    start_time = time.time()

    # Apply sentiment analysis sequentially
    df['sentiment'] = df['clean'].apply(get_sentiment)

    # Measure time after processing
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Display GUI with the last run's sentiment distribution and execution time
    main(execution_time, 1)
