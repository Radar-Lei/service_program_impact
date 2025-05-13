import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
import os
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

# Directory path
directory_path = 'senti_results_SZ'
all_docs = []

# Define the Shenzhen metro station names
sz_metro_stations = [
    "机场东", "机场", "宝安中心", "前海湾", "鲤鱼门", "大新", "桃园", "深大", "科苑", "白石洲", 
    "世界之窗", "华侨城", "侨城北", "香蜜湖", "车公庙", "竹子林", "招商银行大厦", "深康", "黄贝岭", 
    "黄贝", "新秀", "莲塘", "梧桐山南", "梧桐山", "盐田路", "沙头角", "海山", "罗湖", "国贸", 
    "老街", "大剧院", "科学馆", "华强路", "岗厦", "会展中心", "香蜜", "深南香蜜", "红树湾", "后海", 
    "南山", "科技园", "大学城", "桃源村", "龙珠", "龙华", "清湖", "碧海湾", "铁路公园", "西丽湖"
]

# Define keywords for filtering
keywords = ["深圳", "深铁"] + sz_metro_stations

# List to store all filtered DataFrames and their file paths
all_filtered_dfs = []
all_file_paths = []

# First, collect all filtered DataFrames and their file paths
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        
        # Use pandas to read the CSV and preserve row information
        df = pd.read_csv(file_path)
        
        # Check if "微博正文" column exists
        if "微博正文" not in df.columns:
            print(f"Warning: '微博正文' column not found in {file_path}. Skipping file.")
            continue
        
        # Filter rows where "微博正文" contains any of the keywords
        filtered_df = df[df["微博正文"].apply(lambda text: 
            any(keyword in str(text) for keyword in keywords) if pd.notna(text) else False
        )]
        
        if not filtered_df.empty:
            all_filtered_dfs.append(filtered_df)
            all_file_paths.append(file_path)

# Collect all texts for LDA
all_texts = []
for df in all_filtered_dfs:
    all_texts.extend(df["微博正文"].fillna("").astype(str).tolist())

print(f"Collected {len(all_texts)} texts for LDA topic modeling")

# Apply LDA topic modeling
# Initialize parameters with better settings for Chinese text
n_features = 5000  # Increased features to capture more vocabulary
n_topics = 15  # Increased number of topics for more granularity
max_df = 0.90  # Slightly reduced to filter out more common terms
min_df = 3  # Slightly increased to filter more rare terms

# Add custom Chinese stopwords
chinese_stopwords = set([
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "这", "那", "有", "我", "你", "他", "她", 
    "它", "们", "为", "以", "在", "上", "下", "被", "把", "于", "中", "或", "由", "一", "要", "些",
    "吗", "吧", "啊", "呢", "还", "好", "该", "如", "这么", "那么", "什么", "怎么", "样", "嗯", "到",
    "会", "对", "能", "可以", "不", "没", "去", "来", "很", "但", "所以", "因为", "因此", "所", "啦",
    "哦", "哈", "呀", "啦", "吧"
])

def preprocess_chinese_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs, numbers, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fff\s]', ' ', text)  # Keep only Chinese characters
    
    # Segment Chinese text
    seg_list = jieba.cut(text)
    
    # Remove stopwords and short words (likely not meaningful)
    filtered_words = [word for word in seg_list 
                     if word not in chinese_stopwords 
                     and len(word.strip()) > 1 
                     and not word.isdigit()]
    
    return " ".join(filtered_words)

# Load custom dictionary for metro-related terms to improve word segmentation
# Add metro station names to jieba dictionary for better segmentation
for station in sz_metro_stations:
    jieba.add_word(station)
jieba.add_word("深圳地铁")
jieba.add_word("深铁")

# Preprocess all texts
print("Preprocessing Chinese text with stopword removal and custom dictionary...")
preprocessed_texts = []
for text in all_texts:
    processed = preprocess_chinese_text(text)
    # Only keep texts that have meaningful content after preprocessing
    if len(processed.strip()) > 5:  # Filter out very short texts
        preprocessed_texts.append(processed)
    
print(f"After preprocessing: {len(preprocessed_texts)} documents remain")

# First get the term frequency
print("Fitting CountVectorizer...")
tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=n_features)
tf = tf_vectorizer.fit_transform(preprocessed_texts)
print(f"Vectorizer fitted with {tf.shape[1]} features")

# Then transform to TF-IDF
print("Applying TF-IDF transformation...")
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf = tfidf_transformer.fit_transform(tf)
print("TF-IDF transformation complete")

# Try different random seeds and pick the best model
best_lda = None
best_perplexity = float('inf')
random_seeds = [0, 42, 123, 555, 999]  # Try multiple random initializations

for seed in random_seeds:
    print(f"\nTrying LDA with random_state={seed}")
    # Create and fit LDA model
    lda_trial = LatentDirichletAllocation(
        n_components=n_topics, 
        max_iter=75,  # Further increased for better convergence
        learning_method='online',
        learning_offset=75.,  # Increased learning offset
        batch_size=256,  # Increased batch size for better learning from more data at once
        doc_topic_prior=0.05,  # Adjusted prior for document-topic density (alpha)
        topic_word_prior=0.005,  # Adjusted prior for topic-word density (beta)
        verbose=1,  # Verbosity for tracking progress
        evaluate_every=5,  # Evaluate perplexity every 5 iterations
        n_jobs=-1,  # Use all available cores
        random_state=seed
    )
    
    lda_trial.fit(tfidf)  # Use TF-IDF instead of raw counts
    
    # Calculate perplexity
    current_perplexity = lda_trial.perplexity(tfidf)
    print(f"Perplexity with seed {seed}: {current_perplexity:.2f}")
    
    # Keep the best model
    if current_perplexity < best_perplexity:
        best_perplexity = current_perplexity
        best_lda = lda_trial
        print(f"New best model found with seed {seed} (perplexity: {best_perplexity:.2f})")

lda = best_lda  # Use the best model found
print(f"\nLDA model fitting completed. Best perplexity: {best_perplexity:.2f}")

# Further improve model interpretability by only keeping relevant topics
# Get coherence scores or other metrics to evaluate topics
topic_word_matrix = lda.components_
topic_word_scores = np.sum(topic_word_matrix, axis=1)
print("\nTopic importance (sum of word probabilities):")
for idx, score in enumerate(topic_word_scores):
    print(f"Topic {idx}: {score:.4f}")

# Function to get the top terms for each topic with better formatting
def get_top_words_per_topic(model, feature_names, n_top_words=8):
    topic_terms = []
    for topic_idx, topic in enumerate(model.components_):
        top_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_terms.append(' '.join(top_terms))
    return topic_terms

# Get feature names
feature_names = tf_vectorizer.get_feature_names_out()
top_terms_per_topic = get_top_words_per_topic(lda, feature_names)

# Display topics
for idx, terms in enumerate(top_terms_per_topic):
    print(f"Topic {idx}: {terms}")

# Transform all texts to get their topic distributions
topic_distributions = lda.transform(tfidf)  # Use the TF-IDF matrix for consistency

# Assign topics to each text
text_topics = np.argmax(topic_distributions, axis=1)

# Create a mapping between all_texts and preprocessed_texts/topic distributions
# This is needed because we filtered some texts during preprocessing
original_to_processed = {}
processed_idx = 0
all_text_idx = 0

for df in all_filtered_dfs:
    for _, row in df.iterrows():
        text = row["微博正文"]
        if pd.notna(text):
            text = str(text)
            processed = preprocess_chinese_text(text)
            if len(processed.strip()) > 5:  # This text was included in LDA
                original_to_processed[all_text_idx] = processed_idx
                processed_idx += 1
        all_text_idx += 1

# Now add the topic column to each filtered dataframe and create documents
all_text_idx = 0
for i, filtered_df in enumerate(all_filtered_dfs):
    file_path = all_file_paths[i]
    
    # Create new columns for topics in the DataFrame
    filtered_df["topic"] = None
    filtered_df["topic_id"] = None
    filtered_df["topic_strength"] = None
    
    # Assign topics to each row in the DataFrame
    for idx, row in filtered_df.iterrows():
        if all_text_idx in original_to_processed:
            # This text was processed and has a topic
            proc_idx = original_to_processed[all_text_idx]
            topic_id = text_topics[proc_idx]
            topic_terms = top_terms_per_topic[topic_id]
            
            # Calculate topic strength (confidence)
            topic_strength = topic_distributions[proc_idx, topic_id]
            
            # Update the DataFrame
            filtered_df.at[idx, "topic"] = topic_terms
            filtered_df.at[idx, "topic_id"] = int(topic_id)
            filtered_df.at[idx, "topic_strength"] = float(topic_strength)
        else:
            # This text was too short or otherwise filtered out
            filtered_df.at[idx, "topic"] = "unknown_topic"
            filtered_df.at[idx, "topic_id"] = -1
            filtered_df.at[idx, "topic_strength"] = 0.0
        
        all_text_idx += 1
    
    # Convert each filtered row to a Document with appropriate metadata
    for index, row in filtered_df.iterrows():
        # Use the topic as content instead of the original text
        content = row["topic"] if pd.notna(row["topic"]) else "unknown_topic"
        
        # Create metadata with file source and row index
        metadata = {
            'source': file_path,
            'row': index,  # Keep original index from source file
            'original_text': row["微博正文"],  # Store original text in metadata
            'topic_id': row["topic_id"] if pd.notna(row["topic_id"]) else -1,
            'topic_strength': row["topic_strength"] if pd.notna(row["topic_strength"]) else 0.0
        }
        doc = Document(page_content=content, metadata=metadata)
        all_docs.append(doc)

print(f"Loaded {len(all_docs)} documents (rows) from CSVs after filtering for Shenzhen metro related content")

# Split documents for the vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
texts_all = text_splitter.split_documents(all_docs)

# Initialize embeddings
hg_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Create vector store
persist_directory = 'docs/chroma_rag/'
posts_langchain_chroma = Chroma.from_documents(
    documents=texts_all,
    collection_name="SZ_posts_senti",
    embedding=hg_embeddings,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

# Create directory for output if it doesn't exist
output_dir = 'service_program_matches_SZ'
os.makedirs(output_dir, exist_ok=True)

# Load the service program CSV file
service_program_df = pd.read_csv('service_program_data/SPD_SZ_zh.csv')

# Create a cache for the source dataframes to avoid reloading them
source_df_cache = {}

# Iterate through each row of the CSV
for index, row in service_program_df.iterrows():
    # Concatenate the first three columns as service_dimension
    service_dimension = ' '.join([str(row[col]) for col in service_program_df.columns[1:3]])
    
    # Search for similar documents in the vector database
    docs = posts_langchain_chroma.similarity_search_with_relevance_scores(service_dimension, k=300, score_threshold=0.35)
    # Create a dictionary to group matched rows by source file
    matches_by_source = {}
    
    # Process each matched document
    # each doc is a tuple
    for doc in docs:
        source = doc[0].metadata.get('source', None)
        row_index = doc[0].metadata.get('row', None)
        
        if source and row_index is not None:
            # Convert row to integer if it's a numeric string
            try:
                row_index = int(row_index)
            except (ValueError, TypeError):
                continue
                
            if source not in matches_by_source:
                matches_by_source[source] = []
            
            matches_by_source[source].append(row_index)
    
    # Create a list to store all matching rows from original sources
    all_matched_rows = []
    
    # For each source file, get the matching rows
    for source, row_indices in matches_by_source.items():
        # Load the source file if not in cache
        if source not in source_df_cache:
            try:
                source_df_cache[source] = pd.read_csv(source)
            except Exception as e:
                print(f"Error loading {source}: {e}")
                continue
        
        source_df = source_df_cache[source]
        
        # Get rows from original source file
        for row_idx in row_indices:
            try:
                if 0 <= row_idx < len(source_df):
                    row_data = source_df.iloc[row_idx].to_dict()
                    row_data['original_source'] = source
                    row_data['original_row'] = row_idx
                    all_matched_rows.append(row_data)
            except Exception as e:
                print(f"Error accessing row {row_idx} in {source}: {e}")
    
    # Convert to DataFrame
    if all_matched_rows:
        matches_df = pd.DataFrame(all_matched_rows)
        
        # Create a filename for this service program
        safe_filename = f"service_program_{index}_matches.csv"
        
        # Save to CSV
        matches_df.to_csv(os.path.join(output_dir, safe_filename), index=False)
    
    # Print progress
    if index % 10 == 0:
        print(f"Processed {index} rows")

print(f"Completed! All match files saved to {output_dir} directory.")

# Display summary of what was created
print(f"Created CSV files with matching posts:")
for i, filename in enumerate(sorted(os.listdir(output_dir))):
    try:
        match_df = pd.read_csv(os.path.join(output_dir, filename))
        print(f"- {filename}: {len(match_df)} matched original posts")
    except:
        print(f"- {filename}: Could not read file")
