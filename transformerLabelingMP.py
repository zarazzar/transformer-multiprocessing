# 1 Install necessary packages (jalankan ini di terminal, bukan dalam script Python)
# pip install huggingface transformers pandas seaborn torch torchvision torchaudio

# 2 Load Packages
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import time  

# 3 Load Modules
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 4 Read CSV
print("Reading CSV file...")
df = pd.read_csv("https://github.com/zarazzar/sentimen-analisis-boikot-produk-sirewel/raw/main/dataset_boikot_cleaned_2k.csv")
df = df[["id", "author", "description", "clean"]]

# 5 Set pretrained model
print("Setting pretrained model...")
pretrained = "mdhugol/indonesia-bert-sentiment-classification"

# 6 Set Model and Tokenizer
print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

# 7 Create sentiment classifier using huggingface pipeline
print("Creating sentiment analysis pipeline...")
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 8 Ensure 'clean' column is of type str and handle NaNs
print("Ensuring 'clean' column is of type str and handling NaNs...")
df['clean'] = df['clean'].astype(str).fillna('')

# 10 Set Label Index
label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

# 9 Function to get sentiment label
def get_sentiment(text):
    sentiment = sentiment_analysis(text)[0]['label']
    return label_index[sentiment]

# 9 Multiprocessing for sentiment analysis
def apply_sentiment_analysis(data_chunk):
    print("Applying sentiment analysis to a data chunk...")
    return data_chunk.apply(lambda x: get_sentiment(x))

if __name__ == '__main__':
    num_partitions = mp.cpu_count()  # Number of partitions to split the dataframe
    num_cores = mp.cpu_count()  # Number of cores on your machine
    print(f"Number of processors used: {num_cores}")

    # Split dataframe into partitions
    print("Splitting dataframe into partitions...")
    df_split = [df['clean'][i::num_partitions] for i in range(num_partitions)]

    # Create a multiprocessing pool
    print("Creating multiprocessing pool...")
    pool = mp.Pool(num_cores)

    # Measure time before processing
    start_time = time.time()
    print("Starting sentiment analysis...")

    # Apply sentiment analysis to each partition in parallel
    result = pool.map(apply_sentiment_analysis, df_split)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Measure time after processing
    end_time = time.time()

    # Combine the results into a single series
    print("Combining results into a single dataframe...")
    df['sentiment'] = pd.concat(result)

    # 11 Replace the values in the sentiment column
    print("Replacing sentiment labels with readable values...")
    df['sentiment'] = df['sentiment'].replace(label_index)

    # 12 Show Comment with sentiment
    print("Displaying final dataframe with sentiments...")
    print(df[["id", "author", "description", "clean", "sentiment"]])

     # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution time transformer: {execution_time} seconds")
