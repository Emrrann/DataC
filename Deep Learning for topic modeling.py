import pandas as pd
from bertopic import BERTopic

# Step 1: Load the Dataset
def load_data(file_path):
    # Load the dataset from CSV
    data = pd.read_csv("/Users/emran/Desktop/chatdata.csv")
    return data

# Step 2: Preprocess Data
def preprocess_data(data):
    # Ensure all values in the message column are strings and replace NaN with an empty string
    data['message'] = data['message'].fillna("").astype(str)
    
    # Combine chat messages for context (by session, group, period)
    data['combined_text'] = data.groupby(['sessionid', 'group', 'period'])['message'].transform(lambda x: ' '.join(x))
    
    # Drop duplicates and retain essential columns
    data = data[['sessionid', 'group', 'period', 'N', 'combined_text']].drop_duplicates()
    return data

def perform_topic_modeling(data):
    # Ensure all combined text is valid and drop empty entries
    data = data[data['combined_text'].notnull()]  # Remove rows where combined_text is NaN
    data['combined_text'] = data['combined_text'].fillna("").astype(str)
    data = data[data['combined_text'].str.strip() != ""]  # Remove rows with empty strings
    
    # Extract documents (combined chat text)
    docs = data['combined_text'].tolist()

    # Initialize and fit BERTopic
    topic_model = BERTopic(language="english", calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(docs)

    # Add topic assignments to the dataset
    data['Topic'] = topics

    return topic_model, data

# Step 4: Analyze and Visualize Topics
def analyze_topics(topic_model):
    # Display topic overview
    topic_info = topic_model.get_topic_info()
    print("Topic Overview:\n", topic_info)

    # Print top keywords for each topic
    for topic in topic_info.Topic.unique():
        print(f"\nTopic {topic}: {topic_model.get_topic(topic)}")

    # Visualize topics
    topic_model.visualize_topics().show()
    topic_model.visualize_barchart().show()
    topic_model.visualize_hierarchy().show()

# Step 5: Save Results
def save_results(data, topic_model, output_path, model_path):
    # Save the dataset with topic assignments
    data.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Save the topic model
    topic_model.save(model_path)
    print(f"Topic model saved to {model_path}")

# Main Function
if __name__ == "__main__":
    # Input file path
    input_file = "path_to_your_chatdata.csv"  # Replace with your file path

    # Output paths
    output_csv = "/Users/emran/Documents/chat_data_with_topics.csv"
    model_file = "/Users/emran/Documents/bertopic_model"

    # Load and preprocess data
    print("Loading and preprocessing data...")
    chat_data = load_data(input_file)
    processed_data = preprocess_data(chat_data)

    # Perform topic modeling
    print("Performing topic modeling...")
    topic_model, processed_data_with_topics = perform_topic_modeling(processed_data)

    # Analyze topics
    print("Analyzing topics...")
    analyze_topics(topic_model)

    # Save results
    print("Saving results...")
    save_results(processed_data_with_topics, topic_model, output_csv, model_file)

    print("Topic modeling completed successfully!")